import os
import tempfile
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import gradio as gr

# ============================================================
# Labour Intelligence Dashboard — Singapore
# v9.4 (OOM-safe / HF-friendly)
# - Memory-safe CSV loading (usecols + category casting)
# - Filter dropdown choices are lazy-populated on drawer open
# - Keeps: Bloomberg UI, overlay drawer (no squashing), KPI block,
#          downloads for CSV + PDF (no file_name kwarg)
# ============================================================

DATA_PATH = "SGJobData_cleaned.csv"

SALARY_BANDS = ["<$3k", "$3k - $6k", "$7k - $10k", "$11k - $15k", ">$16k"]
COMPOSITE_ALPHA = 0.4

pio.templates.default = "plotly_dark"

HOW_A_BULLETS = f"""
- **Job Postings (Demand)** = sum of `vacancies` per sector.
- **Composite Talent Shortage Index (TSI, 0–100)** blends demand volume and labour tightness pressure:
  - Demand component: `D = minmax_norm(postings)`
  - Pressure component: `P = minmax_norm( postings / (applications + 1) )`
  - Composite: `C = {COMPOSITE_ALPHA:.1f} * D + {1-COMPOSITE_ALPHA:.1f} * P`
  - Final index (0–100): `TSI = 100 * minmax_norm(C)`
"""

HOW_B_BULLETS = """
- For each **sector × salary band** cell:
  - **Postings (this band)** = sum of `vacancies` within that sector and salary band.
- Bubble **size** and **color** are both mapped to `postings_band`.
"""

HOW_C_BULLETS = """
- **Median salary per sector** = median of `salary` values within the sector.
- Sectors are ordered by **demand (postings)** from highest → lowest (within the filtered slice).
"""

HOW_D_BULLETS = """
- Heatmap cell value (**Postings**) = sum of `vacancies` for each **sector × salary band**.
- Only rows with a valid `salary_band` are included.
"""


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_col(existing_cols, candidates) -> str | None:
    cols = set([str(c).strip().lower() for c in existing_cols])
    for c in candidates:
        c = c.lower()
        if c in cols:
            return c
    return None


def _ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip(),
        errors="coerce",
    )


def _minmax_norm(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    return (arr - mn) / (mx - mn + 1e-9)


def _coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def _to_str(s: pd.Series, unknown="Unknown") -> pd.Series:
    return s.astype(str).where(~s.isna(), other=unknown).fillna(unknown)


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    # IMPORTANT (HF / OOM-safe):
    # - We want to read only needed columns (usecols) to reduce memory.
    # - Pandas `usecols` must match the ORIGINAL CSV header names exactly.
    # - Therefore, read header first and build a mapping: normalized_name -> original_name.

    header_raw = pd.read_csv(DATA_PATH, nrows=0)
    orig_cols = list(header_raw.columns)
    lower_map = {str(c).strip().lower(): c for c in orig_cols}
    lower_cols = list(lower_map.keys())

    sector_key = _pick_col(lower_cols, ["sector", "industry", "jobsector", "job_sector", "sector_name"])
    vacancies_key = _pick_col(lower_cols, ["vacancies", "numberofvacancies", "number_of_vacancies", "jobpostings", "postings"])
    applications_key = _pick_col(
        lower_cols,
        ["applications", "metadata_totalnumberjobapplication", "totalapplications", "total_number_job_application"],
    )
    salary_key = _pick_col(lower_cols, ["salary", "avg_salary", "average_salary", "salary_average"])
    salary_min_key = _pick_col(lower_cols, ["salary_min", "salary_minimum", "minimum_salary"])
    salary_max_key = _pick_col(lower_cols, ["salary_max", "salary_maximum", "maximum_salary"])
    emp_key = _pick_col(lower_cols, ["employmenttypes", "employment_type", "employmenttype", "employment"])
    pos_key = _pick_col(lower_cols, ["positionlevels", "position_level", "positionlevel", "level"])
    date_key = _pick_col(
        lower_cols,
        ["date_posted", "posting_date", "dateofposting", "posted_date", "created_at", "createdat", "timestamp"],
    )
    title_key = _pick_col(lower_cols, ["job_title", "title", "jobtitle", "position", "role", "occupation"])

    if not sector_key:
        raise ValueError("Missing required column: sector (or equivalent).")
    if not vacancies_key:
        raise ValueError("Missing required column: vacancies (or equivalent).")
    if not applications_key:
        raise ValueError("Missing required column: applications (or equivalent).")
    if not salary_key and not (salary_min_key or salary_max_key):
        raise ValueError("Missing salary fields. Expected salary/avg_salary OR salary_min/salary_max.")

    # Build ORIGINAL header names for usecols (must match CSV exactly)
    usecols = []
    for k in [sector_key, vacancies_key, applications_key, salary_key, salary_min_key, salary_max_key, emp_key, pos_key, date_key, title_key]:
        if k and k in lower_map:
            usecols.append(lower_map[k])
    # de-dup while preserving order
    seen = set()
    usecols = [c for c in usecols if not (c in seen or seen.add(c))]

    df0 = pd.read_csv(DATA_PATH, usecols=usecols)
    df0 = _normalize_cols(df0)

    # After normalization, all columns are lowercase. Use the *keys* (already lowercase) as column selectors.
    sector_col = sector_key
    vacancies_col = vacancies_key
    applications_col = applications_key
    salary_col = salary_key
    salary_min_col = salary_min_key
    salary_max_col = salary_max_key
    emp_col = emp_key
    pos_col = pos_key
    date_col = date_key
    title_col = title_key

    out = pd.DataFrame(index=df0.index)
    out["sector"] = _to_str(df0[sector_col], unknown="Unknown")
    out["vacancies"] = _ensure_numeric(df0[vacancies_col]).fillna(0)
    out["applications"] = _ensure_numeric(df0[applications_col]).fillna(0)

    if salary_col and salary_col in df0.columns:
        out["salary"] = _ensure_numeric(df0[salary_col])
    else:
        smin = _ensure_numeric(df0[salary_min_col]) if (salary_min_col and salary_min_col in df0.columns) else pd.Series(np.nan, index=df0.index)
        smax = _ensure_numeric(df0[salary_max_col]) if (salary_max_col and salary_max_col in df0.columns) else pd.Series(np.nan, index=df0.index)
        if salary_min_col and salary_max_col and (salary_min_col in df0.columns) and (salary_max_col in df0.columns):
            out["salary"] = (smin + smax) / 2.0
        elif salary_min_col and (salary_min_col in df0.columns):
            out["salary"] = smin
        elif salary_max_col and (salary_max_col in df0.columns):
            out["salary"] = smax
        else:
            out["salary"] = pd.Series(np.nan, index=df0.index)

    out["salary"] = out["salary"].where(out["salary"] > 0, np.nan)
    bins = [-np.inf, 3000, 6000, 10000, 15000, np.inf]
    out["salary_band"] = pd.cut(out["salary"], bins=bins, labels=SALARY_BANDS, right=True)

    out["employmenttypes"] = _to_str(df0[emp_col], unknown="Unknown") if (emp_col and emp_col in df0.columns) else "Unknown"
    out["positionlevels"] = _to_str(df0[pos_col], unknown="Unknown") if (pos_col and pos_col in df0.columns) else "Unknown"
    out["job_title"] = _to_str(df0[title_col], unknown="Unknown") if (title_col and title_col in df0.columns) else "Unknown"
    out["date_posted"] = _coerce_datetime(df0[date_col]) if (date_col and date_col in df0.columns) else pd.NaT

    # Compress memory
    out["sector"] = out["sector"].astype("category")
    out["salary_band"] = pd.Categorical(out["salary_band"], categories=SALARY_BANDS, ordered=True)
    out["employmenttypes"] = pd.Series(out["employmenttypes"]).astype("category")
    out["positionlevels"] = pd.Series(out["positionlevels"]).astype("category")

    # Downcast numerics
    out["vacancies"] = pd.to_numeric(out["vacancies"], downcast="integer", errors="coerce").fillna(0).astype("int32")
    out["applications"] = pd.to_numeric(out["applications"], downcast="integer", errors="coerce").fillna(0).astype("int32")
    out["salary"] = pd.to_numeric(out["salary"], downcast="float", errors="coerce")

    return out


@lru_cache(maxsize=8)
def default_slice() -> pd.DataFrame:
    df = load_data()
    top15 = (
        df.groupby("sector", observed=True)["vacancies"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    return df[df["sector"].isin(top15)].copy()


def compute_metrics_for_df(dff: pd.DataFrame):
    if len(dff) == 0:
        postings = pd.DataFrame(columns=["sector", "postings"])
        apps_sector = pd.DataFrame(columns=["sector", "applications_sector"])
        tsi_df = pd.DataFrame(columns=["sector", "tsi"])
        med_salary = pd.DataFrame(columns=["sector", "median_salary"])
        postings_band = pd.DataFrame(columns=["sector", "salary_band", "postings_band"])
        demand_order = []
        return postings, apps_sector, tsi_df, med_salary, postings_band, demand_order

    postings = (
        dff.groupby("sector", observed=True)["vacancies"]
        .sum()
        .reset_index(name="postings")
        .sort_values("postings", ascending=False)
    )

    apps_sector = (
        dff.groupby("sector", observed=True)["applications"]
        .sum()
        .reset_index(name="applications_sector")
    )

    joined = postings.merge(apps_sector, on="sector", how="left").copy()
    pressure = joined["postings"] / (joined["applications_sector"] + 1.0)

    D = _minmax_norm(joined["postings"])
    P = _minmax_norm(pressure)
    C = COMPOSITE_ALPHA * D + (1.0 - COMPOSITE_ALPHA) * P
    tsi = 100.0 * _minmax_norm(C)

    tsi_df = pd.DataFrame({"sector": joined["sector"], "tsi": tsi})
    med_salary = dff.groupby("sector", observed=True)["salary"].median().reset_index(name="median_salary")

    dff_band = dff.dropna(subset=["salary_band"]).copy()
    postings_band = (
        dff_band.groupby(["sector", "salary_band"], observed=True)["vacancies"]
        .sum()
        .reset_index(name="postings_band")
    )

    demand_order = postings["sector"].astype(str).tolist()
    return postings, apps_sector, tsi_df, med_salary, postings_band, demand_order


def apply_filters(sectors, salary_bands, date_start, date_end, employment_types, position_levels) -> pd.DataFrame:
    df = load_data()

    if not sectors:
        raise gr.Error("Please select **1–3 sectors** for deep-dive analysis, then click Generate.")
    if len(sectors) > 3:
        raise gr.Error("Please select at most **3 sectors** for deep-dive analysis.")

    dff = df[df["sector"].astype(str).isin([str(s) for s in sectors])].copy()

    if salary_bands:
        dff = dff[dff["salary_band"].astype(str).isin([str(b) for b in salary_bands])]

    if employment_types:
        dff = dff[dff["employmenttypes"].astype(str).isin([str(e) for e in employment_types])]

    if position_levels:
        dff = dff[dff["positionlevels"].astype(str).isin([str(p) for p in position_levels])]

    if "date_posted" in dff.columns and dff["date_posted"].notna().any():
        if date_start is not None:
            ds = pd.to_datetime(date_start)
            dff = dff[dff["date_posted"] >= ds]
        if date_end is not None:
            de = pd.to_datetime(date_end)
            dff = dff[dff["date_posted"] <= de]

    return dff.copy()


def _apply_bloomberg_layout(fig: go.Figure, title: str, x_title: str | None = None, y_title: str | None = None):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", y=0.99, yanchor="top"),
        margin=dict(l=75, r=35, t=110, b=70),
        height=470,
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
    )
    if x_title is not None:
        fig.update_xaxes(title_text=f"<b>{x_title}</b>", title_standoff=18)
    if y_title is not None:
        fig.update_yaxes(title_text=f"<b>{y_title}</b>")
    return fig


def _kpi_html(dff: pd.DataFrame, label: str) -> str:
    total_rows = int(len(dff))
    total_vac = float(dff["vacancies"].sum()) if "vacancies" in dff.columns else 0.0
    total_apps = float(dff["applications"].sum()) if "applications" in dff.columns else 0.0
    sectors = int(dff["sector"].astype(str).nunique()) if "sector" in dff.columns else 0
    emp = int(dff["employmenttypes"].astype(str).nunique()) if "employmenttypes" in dff.columns else 0
    lvl = int(dff["positionlevels"].astype(str).nunique()) if "positionlevels" in dff.columns else 0

    date_min = ""
    date_max = ""
    if "date_posted" in dff.columns and dff["date_posted"].notna().any():
        mn = pd.to_datetime(dff["date_posted"], errors="coerce").min()
        mx = pd.to_datetime(dff["date_posted"], errors="coerce").max()
        if pd.notna(mn) and pd.notna(mx):
            date_min = mn.strftime("%Y-%m-%d")
            date_max = mx.strftime("%Y-%m-%d")

    date_line = f"{date_min} → {date_max}" if date_min and date_max else "N/A"

    return f"""
    <div class="kpi-wrap">
      <div class="kpi-label">{label}</div>
      <div class="kpi-grid">
        <div class="kpi-item"><div class="kpi-k">Rows</div><div class="kpi-v">{total_rows:,}</div></div>
        <div class="kpi-item"><div class="kpi-k">Total Vacancies</div><div class="kpi-v">{total_vac:,.0f}</div></div>
        <div class="kpi-item"><div class="kpi-k">Total Applications</div><div class="kpi-v">{total_apps:,.0f}</div></div>
        <div class="kpi-item"><div class="kpi-k">Sectors</div><div class="kpi-v">{sectors:,}</div></div>
        <div class="kpi-item"><div class="kpi-k">Emp Types</div><div class="kpi-v">{emp:,}</div></div>
        <div class="kpi-item"><div class="kpi-k">Levels</div><div class="kpi-v">{lvl:,}</div></div>
        <div class="kpi-item kpi-wide"><div class="kpi-k">Date Range</div><div class="kpi-v">{date_line}</div></div>
      </div>
    </div>
    """


def _context_summary_a(combo: pd.DataFrame) -> str:
    top_tsi = combo.sort_values("tsi", ascending=False).iloc[0]
    top_post = combo.sort_values("postings", ascending=False).iloc[0]
    return (
        f"**Executive Summary (Singapore context):**\n\n"
        f"- **Shortage pressure:** **{top_tsi['sector']}** leads (TSI **{top_tsi['tsi']:.1f}/100**). "
        f"This often corresponds to longer time-to-fill and stronger competition for job-ready profiles.\n"
        f"- **Demand load:** **{top_post['sector']}** leads in postings (**{top_post['postings']:,.0f}**). "
        f"Even with moderate TSI, sheer hiring volume can strain screening and interview throughput.\n"
        f"- **Implication:** Prioritise speed (process), precision (requirements), and targeted pay/benefits; "
        f"if pressure persists, internal mobility + reskilling tends to be the most scalable response.\n"
    )


def _context_summary_b(bubble: pd.DataFrame) -> str:
    top_cell = bubble.sort_values("postings_band", ascending=False).iloc[0]
    return (
        f"**Executive Summary (Singapore context):**\n\n"
        f"- **Demand concentration:** **{top_cell['sector']}** in **{top_cell['salary_band']}** "
        f"(**{top_cell['postings_band']:,.0f}** postings).\n"
        f"- **Interpretation:** High-band clustering tends to signal specialist scarcity; mid-band clustering is usually "
        f"about pipeline throughput and assessment speed.\n"
        f"- **Implication:** Match sourcing strategy to band (specialist networks vs. volume channels) and calibrate "
        f"comp offers to reduce drop-off.\n"
    )


def _context_summary_c(cdf: pd.DataFrame) -> str:
    top_pay = cdf.sort_values("median_salary", ascending=False).iloc[0]
    if pd.isna(top_pay["median_salary"]):
        return (
            "**Executive Summary (Singapore context):**\n\n"
            "- Median salary is unavailable for this slice (missing salary values). "
            "If salary insight matters, enforce structured salary fields upstream.\n"
        )
    return (
        f"**Executive Summary (Singapore context):**\n\n"
        f"- **Pay signal:** **{top_pay['sector']}** leads (**${top_pay['median_salary']:,.0f}**/month).\n"
        f"- **Demand-pay mismatch:** High demand with weaker pay often drives hiring friction or retention pressure unless "
        f"offset by progression/benefits.\n"
        f"- **Implication:** Use this to decide whether to compete on pay, speed, or role design.\n"
    )


def _context_summary_d(heat_df: pd.DataFrame) -> str:
    band_totals = heat_df.groupby("salary_band", observed=True)["postings_band"].sum().sort_values(ascending=False)
    top_band = str(band_totals.index[0]) if len(band_totals) else "N/A"
    return (
        f"**Executive Summary (Singapore context):**\n\n"
        f"- **Band concentration:** strongest in **{top_band}**.\n"
        f"- **Interpretation:** When pressure is band-localised, broad averages won’t fix shortages.\n"
        f"- **Implication:** Prioritise training supply and budgets on the concentrated bands.\n"
    )


def build_outputs(dff: pd.DataFrame, title_prefix: str):
    postings, apps_sector, tsi_df, med_salary, postings_band, demand_order = compute_metrics_for_df(dff)

    if len(postings) == 0:
        empty = go.Figure()
        empty.update_layout(title=dict(text=f"<b>{title_prefix} — No data</b>", x=0.5, xanchor="center"))
        empty.update_layout(height=470, margin=dict(l=40, r=20, t=85, b=40))
        msg = "**Executive Summary:** No postings matched your current filter selections."
        return empty, msg, empty, msg, empty, msg, empty, msg

    combo = postings.merge(tsi_df, on="sector", how="left").sort_values("postings", ascending=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=combo["sector"].astype(str),
        x=combo["postings"],
        name="Job Postings",
        orientation="h",
        hovertemplate="<b>%{y}</b><br>Postings: %{x:,.0f}<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        y=combo["sector"].astype(str),
        x=combo["tsi"],
        name="Composite TSI (0–100)",
        mode="lines+markers",
        xaxis="x2",
        hovertemplate="<b>%{y}</b><br>Composite TSI: %{x:.1f}/100<extra></extra>",
    ))
    fig1.update_layout(
        xaxis2=dict(title=None, overlaying="x", side="top", range=[0, 100], showgrid=False, zeroline=False, ticks="outside", showticklabels=True),
        yaxis=dict(title="", automargin=True),
    )
    _apply_bloomberg_layout(fig1, f"{title_prefix}: Demand (Postings) vs Composite TSI (0–100)", x_title="Job Postings")
    s1 = _context_summary_a(combo)

    if len(postings_band) == 0:
        fig2 = go.Figure()
        _apply_bloomberg_layout(fig2, f"{title_prefix}: Job Postings by Salary Band (no salary bands)", x_title="Salary band", y_title="Sector")
        s2 = "**Executive Summary:** Salary-band charts are unavailable because salary bands are missing for the selected slice."
    else:
        bubble = postings_band.merge(postings, on="sector", how="left").merge(apps_sector, on="sector", how="left")
        bubble["salary_band"] = pd.Categorical(bubble["salary_band"], categories=SALARY_BANDS, ordered=True)
        fig2 = px.scatter(bubble, x="salary_band", y=bubble["sector"].astype(str), size="postings_band", color="postings_band", size_max=36)
        fig2.update_traces(hovertemplate="<b>%{y}</b><br>Salary band: %{x}<br>Postings: %{marker.size:,.0f}<extra></extra>")
        _apply_bloomberg_layout(fig2, f"{title_prefix}: Job Postings by Salary Band", x_title="Salary band", y_title="Sector")
        s2 = _context_summary_b(bubble)

    cdf = postings.merge(med_salary, on="sector", how="left").sort_values("postings", ascending=False)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=cdf["sector"].astype(str),
        y=cdf["median_salary"],
        mode="lines+markers",
        name="Median Salary",
        customdata=np.stack([cdf["postings"]], axis=1),
        hovertemplate="<b>%{x}</b><br>Median salary: $%{y:,.0f}<br>Postings: %{customdata[0]:,.0f}<extra></extra>",
    ))
    fig3.update_xaxes(tickangle=30)
    _apply_bloomberg_layout(fig3, f"{title_prefix}: Median Salary (Ordered by Demand)", x_title="Sector (ranked by demand)", y_title="Median salary (SGD/month)")
    s3 = _context_summary_c(cdf)

    if len(postings_band) == 0:
        fig4 = go.Figure()
        _apply_bloomberg_layout(fig4, f"{title_prefix}: Vacancies Heatmap (no salary bands)", x_title="Salary band", y_title="Sector")
        s4 = "**Executive Summary:** Heatmap is unavailable because salary bands are missing for the selected slice."
    else:
        heat_df = postings_band.copy()
        heat_df["salary_band"] = pd.Categorical(heat_df["salary_band"], categories=SALARY_BANDS, ordered=True)
        heat_pivot = (
            heat_df.pivot(index="sector", columns="salary_band", values="postings_band")
            .reindex(index=pd.Index(demand_order, name="sector"), fill_value=0)
            .reindex(columns=SALARY_BANDS, fill_value=0)
            .fillna(0)
        )
        fig4 = go.Figure(data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.astype(str),
            y=heat_pivot.index.astype(str),
            colorbar=dict(title="Postings"),
            hovertemplate="<b>%{y}</b><br>Salary band: %{x}<br>Postings: %{z:,.0f}<extra></extra>",
        ))
        _apply_bloomberg_layout(fig4, f"{title_prefix}: Vacancies Heatmap — Sector vs Salary Band", x_title="Salary band", y_title="Sector")
        s4 = _context_summary_d(heat_df)

    return fig1, s1, fig2, s2, fig3, s3, fig4, s4


def build_table_preview(dff: pd.DataFrame) -> pd.DataFrame:
    cols = ["job_title", "sector", "employmenttypes", "positionlevels", "salary_band", "salary", "vacancies", "applications"]
    if "date_posted" in dff.columns and dff["date_posted"].notna().any():
        cols.append("date_posted")
    cols = [c for c in cols if c in dff.columns]
    out = dff[cols].copy()
    if "date_posted" in out.columns:
        out["date_posted"] = pd.to_datetime(out["date_posted"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.head(10)


def to_csv_tempfile(dff: pd.DataFrame) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="labour_dashboard_")
    path = os.path.join(tmp_dir, "labour_dashboard_export.csv")
    dff.to_csv(path, index=False)
    return path


def _try_write_plotly_png(fig: go.Figure, path: str) -> bool:
    try:
        fig.write_image(path, scale=2)
        return True
    except Exception:
        return False


def generate_board_report_pdf(figs, summaries, title: str) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    import textwrap

    tmp_dir = tempfile.mkdtemp(prefix="board_report_")
    pdf_path = os.path.join(tmp_dir, "board_report.pdf")

    w, h = A4
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setTitle(title)

    left = 2.0 * cm
    right = w - 2.0 * cm
    col_gap = 0.6 * cm
    col_w = (right - left - col_gap) / 2.0
    img_h = 6.0 * cm

    tmp_imgs = []
    ok = True
    for i, fig in enumerate(figs, start=1):
        img_path = tempfile.mktemp(prefix=f"chart_{i}_", suffix=".png")
        ok = _try_write_plotly_png(fig, img_path)
        tmp_imgs.append(img_path)
        if not ok:
            break

    def draw_page_header(main_title: str, subtitle: str | None = None):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(left, h - 2.0 * cm, main_title)
        if subtitle:
            c.setFont("Helvetica", 9)
            c.drawString(left, h - 2.6 * cm, subtitle)

    # Page 1
    draw_page_header(title, "Generated from Singapore Labour Intelligence Dashboard (selected filters).")
    y_top = h - 3.4 * cm
    y_img = y_top - 0.4 * cm - img_h

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y_top, "Chart A")
    c.drawString(left + col_w + col_gap, y_top, "Chart B")

    if ok:
        c.drawImage(ImageReader(tmp_imgs[0]), left, y_img, width=col_w, height=img_h, preserveAspectRatio=True, mask="auto")
        c.drawImage(ImageReader(tmp_imgs[1]), left + col_w + col_gap, y_img, width=col_w, height=img_h, preserveAspectRatio=True, mask="auto")
    else:
        c.setFont("Helvetica", 9)
        c.drawString(left, y_img + img_h / 2, "Chart rendering unavailable (install kaleido for embedded charts).")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, y_img - 0.5 * cm, "Interpretation (Board View)")
    c.setFont("Helvetica", 9)

    y_text = y_img - 0.9 * cm
    for block in (summaries[0], summaries[1]):
        text = block.replace("**", "").replace("\n\n", "\n").replace("\n", " ")
        for line in textwrap.wrap(text, width=120):
            if y_text < 2.0 * cm:
                break
            c.drawString(left, y_text, line)
            y_text -= 11
        y_text -= 6

    c.showPage()

    # Page 2
    draw_page_header("Board Report — Continued")
    y_top2 = h - 3.2 * cm
    y_img2 = y_top2 - 0.4 * cm - img_h

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y_top2, "Chart C")
    c.drawString(left + col_w + col_gap, y_top2, "Chart D")

    if ok:
        c.drawImage(ImageReader(tmp_imgs[2]), left, y_img2, width=col_w, height=img_h, preserveAspectRatio=True, mask="auto")
        c.drawImage(ImageReader(tmp_imgs[3]), left + col_w + col_gap, y_img2, width=col_w, height=img_h, preserveAspectRatio=True, mask="auto")
    else:
        c.setFont("Helvetica", 9)
        c.drawString(left, y_img2 + img_h / 2, "Chart rendering unavailable (install kaleido for embedded charts).")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, y_img2 - 0.5 * cm, "Interpretation (Board View)")
    c.setFont("Helvetica", 9)

    y_text2 = y_img2 - 0.9 * cm
    for block in (summaries[2], summaries[3]):
        text = block.replace("**", "").replace("\n\n", "\n").replace("\n", " ")
        for line in textwrap.wrap(text, width=120):
            if y_text2 < 2.0 * cm:
                break
            c.drawString(left, y_text2, line)
            y_text2 -= 11
        y_text2 -= 6

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(left, 1.4 * cm, "Note: Directional interpretation of postings-based indicators; validate with hiring outcomes and wage data.")
    c.save()

    return pdf_path


def cb_init():
    dff_full = load_data()
    dff = default_slice()

    fig1, s1, fig2, s2, fig3, s3, fig4, s4 = build_outputs(dff, title_prefix="Top 15 Sectors")
    table = build_table_preview(dff)
    kpi = _kpi_html(dff_full, "Dataset (Full)") + _kpi_html(dff, "View (Default Top 15)")
    return fig1, s1, fig2, s2, fig3, s3, fig4, s4, table, gr.update(visible=False), False, kpi


def cb_generate(sectors, salary_bands, date_start, date_end, employment_types, position_levels):
    dff_full = load_data()
    dff = apply_filters(sectors, salary_bands, date_start, date_end, employment_types, position_levels)
    fig1, s1, fig2, s2, fig3, s3, fig4, s4 = build_outputs(dff, title_prefix="Filtered View")
    table = build_table_preview(dff)
    kpi = _kpi_html(dff_full, "Dataset (Full)") + _kpi_html(dff, "View (Filtered)")
    return fig1, s1, fig2, s2, fig3, s3, fig4, s4, table, gr.update(visible=False), False, kpi


def cb_default():
    dff_full = load_data()
    dff = default_slice()
    fig1, s1, fig2, s2, fig3, s3, fig4, s4 = build_outputs(dff, title_prefix="Top 15 Sectors")
    table = build_table_preview(dff)
    kpi = _kpi_html(dff_full, "Dataset (Full)") + _kpi_html(dff, "View (Default Top 15)")

    return (
        fig1, s1, fig2, s2, fig3, s3, fig4, s4,
        table,
        gr.update(value=[]),
        gr.update(value=[]),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=[]),
        gr.update(value=[]),
        gr.update(visible=False),
        False,
        kpi,
    )


def cb_download_csv(sectors, salary_bands, date_start, date_end, employment_types, position_levels):
    if not sectors:
        dff = default_slice()
    else:
        dff = apply_filters(sectors, salary_bands, date_start, date_end, employment_types, position_levels)
    return to_csv_tempfile(dff)


def cb_board_report(sectors, salary_bands, date_start, date_end, employment_types, position_levels):
    if not sectors:
        dff = default_slice()
        title_prefix = "Top 15 Sectors"
        report_title = "Board Report — Singapore Labour Intelligence (Default Top 15)"
    else:
        dff = apply_filters(sectors, salary_bands, date_start, date_end, employment_types, position_levels)
        title_prefix = "Filtered View"
        report_title = "Board Report — Singapore Labour Intelligence (Filtered View)"

    fig1, s1, fig2, s2, fig3, s3, fig4, s4 = build_outputs(dff, title_prefix=title_prefix)
    return generate_board_report_pdf([fig1, fig2, fig3, fig4], [s1, s2, s3, s4], report_title)


def cb_populate_filter_choices():
    df = load_data()
    # Categories are already built; this is low-overhead vs unique()
    sectors = df["sector"].cat.categories.astype(str).tolist() if hasattr(df["sector"], "cat") else sorted(df["sector"].astype(str).unique().tolist())
    emp = df["employmenttypes"].cat.categories.astype(str).tolist() if hasattr(df["employmenttypes"], "cat") else sorted(df["employmenttypes"].astype(str).unique().tolist())
    lvl = df["positionlevels"].cat.categories.astype(str).tolist() if hasattr(df["positionlevels"], "cat") else sorted(df["positionlevels"].astype(str).unique().tolist())
    return (
        gr.update(choices=sectors),
        gr.update(choices=emp),
        gr.update(choices=lvl),
    )


CSS = """
:root {
  --bb-bg: #0b0f14;
  --bb-panel: #0f1620;
  --bb-border: rgba(255,255,255,0.10);
  --bb-text: #e5e7eb;
  --bb-muted: #9ca3af;
}

body, .gradio-container {
  background: var(--bb-bg) !important;
  color: var(--bb-text) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}

.bb-card {
  background: var(--bb-panel) !important;
  border: 1px solid var(--bb-border) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
}

.bb-toolbar {
  background: transparent !important;
  border: 1px solid var(--bb-border) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.bb-toolbar .gr-button {
  width: auto !important;
  min-width: 150px;
}

.bb-btnstack { gap: 10px; }

.kpi-wrap { padding: 2px 6px; }
.kpi-label { color: var(--bb-muted); font-size: 11px; margin-bottom: 4px; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(90px, 1fr));
  gap: 8px;
}
.kpi-item {
  border: 1px solid var(--bb-border);
  border-radius: 10px;
  padding: 6px 8px;
  background: rgba(255,255,255,0.03);
}
.kpi-item.kpi-wide { grid-column: span 2; }
.kpi-k { color: var(--bb-muted); font-size: 10px; }
.kpi-v { color: var(--bb-text); font-weight: 700; font-size: 13px; }

.bb-drawer {
  position: fixed;
  left: 18px;
  top: 140px;
  width: 380px;
  max-height: 78vh;
  overflow: auto;
  z-index: 50;
  box-shadow: 0 10px 30px rgba(0,0,0,0.45);
}

.bb-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.35);
  z-index: 40;
}

.bb-actions-right {
  margin-left: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: flex-start;
}


@media (max-width: 1200px){
  .kpi-grid { grid-template-columns: repeat(3, minmax(90px, 1fr)); }
  .kpi-item.kpi-wide { grid-column: span 3; }
  .bb-drawer { width: 92vw; left: 4vw; }
}
"""


with gr.Blocks(css=CSS, theme=gr.themes.Base()) as app:
    gr.Markdown("# **Labour Intelligence Dashboard — Singapore**")
    gr.Markdown("<span style='color:#9ca3af'>Composite TSI is a directional 0–100 signal blending demand volume and tightness pressure.</span>")

    drawer_visible = gr.State(False)
    choices_loaded = gr.State(False)

    with gr.Row(elem_classes=["bb-toolbar"]):
        with gr.Column(scale=0, min_width=190, elem_classes=["bb-btnstack"]):
            btn_filters = gr.Button("☰ Filters")
            btn_generate_btn = gr.Button("Generate", variant="primary")
            btn_default_btn = gr.Button("Default")

        kpi_html = gr.HTML("")

        with gr.Column(scale=0, min_width=220, elem_classes=["bb-actions-right"]):
            csv_download = gr.DownloadButton("Download CSV")
            pdf_download = gr.DownloadButton("Board Report (PDF)")

    backdrop = gr.HTML("", visible=False, elem_classes=["bb-backdrop"])

    with gr.Column(visible=False, elem_classes=["bb-card", "bb-drawer"]) as drawer:
        gr.Markdown("## Filters (Deep-Dive)")
        gr.Markdown("Select **1–3 sectors**, refine using other filters, then click **Generate**.")

        # Lazy choices (start empty)
        sector_dd = gr.Dropdown(choices=[], multiselect=True, value=[], label="Sectors (select 1–3)", interactive=True)
        salary_dd = gr.Dropdown(choices=SALARY_BANDS, multiselect=True, value=[], label="Salary Bands", interactive=True)
        emp_dd = gr.Dropdown(choices=[], multiselect=True, value=[], label="Employment Types", interactive=True)
        pos_dd = gr.Dropdown(choices=[], multiselect=True, value=[], label="Position Levels", interactive=True)

        with gr.Row():
            date_start = gr.DateTime(label="Posting date (start)", value=None)
            date_end = gr.DateTime(label="Posting date (end)", value=None)

        close_drawer_btn = gr.Button("Close")

    with gr.Column():
        with gr.Row():
            with gr.Column(elem_classes=["bb-card"]):
                chartA = gr.Plot(show_label=False)
                sumA = gr.Markdown()
                gr.Markdown("**How we derive the numbers**")
                with gr.Accordion("Show details", open=False):
                    gr.Markdown(HOW_A_BULLETS)

            with gr.Column(elem_classes=["bb-card"]):
                chartB = gr.Plot(show_label=False)
                sumB = gr.Markdown()
                gr.Markdown("**How we derive the numbers**")
                with gr.Accordion("Show details", open=False):
                    gr.Markdown(HOW_B_BULLETS)

        with gr.Row():
            with gr.Column(elem_classes=["bb-card"]):
                chartC = gr.Plot(show_label=False)
                sumC = gr.Markdown()
                gr.Markdown("**How we derive the numbers**")
                with gr.Accordion("Show details", open=False):
                    gr.Markdown(HOW_C_BULLETS)

            with gr.Column(elem_classes=["bb-card"]):
                chartD = gr.Plot(show_label=False)
                sumD = gr.Markdown()
                gr.Markdown("**How we derive the numbers**")
                with gr.Accordion("Show details", open=False):
                    gr.Markdown(HOW_D_BULLETS)

        with gr.Column(elem_classes=["bb-card"]):
            gr.Markdown("## Postings (Preview — Top 10)")
            postings_table = gr.Dataframe(interactive=False, wrap=True)

        with gr.Column(elem_classes=["bb-card"]):
            gr.Markdown(
                "<span style=\"color:#9ca3af; font-size:12px;\">Note: Directional interpretation of postings-based indicators; validate with hiring outcomes, wage data, and policy changes. TSI is a relative signal and should not be treated as a definitive measure of labour scarcity.</span>"
            )

    app.load(
        fn=cb_init,
        inputs=[],
        outputs=[chartA, sumA, chartB, sumB, chartC, sumC, chartD, sumD, postings_table, drawer, drawer_visible, kpi_html],
    )

    def _open_drawer(is_vis: bool, loaded: bool):
        new_vis = not is_vis
        # If opening for the first time, load choices
        return gr.update(visible=new_vis), new_vis, gr.update(visible=new_vis), loaded

    btn_filters.click(fn=_open_drawer, inputs=[drawer_visible, choices_loaded], outputs=[drawer, drawer_visible, backdrop, choices_loaded]).then(
        fn=lambda vis, loaded: cb_populate_filter_choices() if (vis and not loaded) else (gr.update(), gr.update(), gr.update()),
        inputs=[drawer_visible, choices_loaded],
        outputs=[sector_dd, emp_dd, pos_dd],
    ).then(
        fn=lambda vis, loaded: True if (vis and not loaded) else loaded,
        inputs=[drawer_visible, choices_loaded],
        outputs=[choices_loaded],
    )

    close_drawer_btn.click(fn=lambda: (gr.update(visible=False), False, gr.update(visible=False)), inputs=[], outputs=[drawer, drawer_visible, backdrop])
    backdrop.click(fn=lambda: (gr.update(visible=False), False, gr.update(visible=False)), inputs=[], outputs=[drawer, drawer_visible, backdrop])

    btn_generate_btn.click(
        fn=cb_generate,
        inputs=[sector_dd, salary_dd, date_start, date_end, emp_dd, pos_dd],
        outputs=[chartA, sumA, chartB, sumB, chartC, sumC, chartD, sumD, postings_table, drawer, drawer_visible, kpi_html],
    ).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[backdrop])

    btn_default_btn.click(
        fn=cb_default,
        inputs=[],
        outputs=[
            chartA, sumA, chartB, sumB, chartC, sumC, chartD, sumD,
            postings_table,
            sector_dd, salary_dd, date_start, date_end, emp_dd, pos_dd,
            drawer, drawer_visible,
            kpi_html,
        ],
    ).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[backdrop])

    csv_download.click(
        fn=cb_download_csv,
        inputs=[sector_dd, salary_dd, date_start, date_end, emp_dd, pos_dd],
        outputs=[csv_download],
    )

    pdf_download.click(
        fn=cb_board_report,
        inputs=[sector_dd, salary_dd, date_start, date_end, emp_dd, pos_dd],
        outputs=[pdf_download],
    )


app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_api=False)
