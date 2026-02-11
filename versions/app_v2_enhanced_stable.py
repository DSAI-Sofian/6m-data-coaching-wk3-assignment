import os
import tempfile
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

# ============================================================
# Labour Intelligence Dashboard — Singapore
# Final v7 (HF Spaces + Gradio 5.x compatible)
#
# - No dict/DataFrame objects stored in gr.State
# - Filter drawer + Generate + Default
# - Table preview Top 10 + CSV export for current filters
# - PORT env var + 0.0.0.0 binding for HF Spaces
# ============================================================

DATA_PATH = "SGJobData_cleaned.csv"

SALARY_BANDS = ["<$3k", "$3k - $6k", "$7k - $10k", "$11k - $15k", ">$16k"]
COMPOSITE_ALPHA = 0.4

HOW_A = f"""### How we derive the numbers
- **Job Postings (Demand)** = sum of `vacancies` per sector.
- **Composite Talent Shortage Index (TSI, 0–100)** blends (i) demand volume and (ii) labour tightness pressure:
  - Demand component: `D = minmax_norm(postings)`
  - Pressure component: `P = minmax_norm( postings / (applications + 1) )`
  - Composite: `C = {COMPOSITE_ALPHA:.1f} * D + {1-COMPOSITE_ALPHA:.1f} * P`
  - Final index (0–100): `TSI = 100 * minmax_norm(C)`
"""

HOW_B = """### How we derive the numbers
- For each **sector × salary band** cell:
  - **Postings (this band)** = sum of `vacancies` within that sector and salary band.
- Bubble **size** and **color** are both mapped to `postings_band` (higher = larger + brighter).
"""

HOW_C = """### How we derive the numbers
- **Median salary per sector** = median of `salary` values within the sector.
- Sectors on the x-axis are ordered by **demand (postings)** from highest → lowest (within the filtered slice).
"""

HOW_D = """### How we derive the numbers
- Heatmap cell value (**Postings**) = sum of `vacancies` for each **sector × salary band**.
- Only rows with a valid `salary_band` are included (rows with missing/unparsable salary are excluded from band charts).
"""


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        c = c.lower()
        if c in df.columns:
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
    df0 = pd.read_csv(DATA_PATH)
    df0 = _normalize_cols(df0)

    sector_col = _pick_col(df0, ["sector", "industry", "jobsector", "job_sector", "sector_name"])
    if not sector_col:
        raise ValueError("Missing required column: sector (or equivalent).")

    vacancies_col = _pick_col(df0, ["vacancies", "numberofvacancies", "number_of_vacancies", "jobpostings", "postings"])
    if not vacancies_col:
        raise ValueError("Missing required column: vacancies (e.g. numberOfVacancies).")

    applications_col = _pick_col(
        df0,
        ["applications", "metadata_totalnumberjobapplication", "totalapplications", "total_number_job_application"],
    )
    if not applications_col:
        raise ValueError("Missing required column: applications (e.g. metadata_totalNumberJobApplication).")

    salary_col = _pick_col(df0, ["salary", "avg_salary", "average_salary", "salary_average"])
    salary_min_col = _pick_col(df0, ["salary_min", "salary_minimum", "minimum_salary"])
    salary_max_col = _pick_col(df0, ["salary_max", "salary_maximum", "maximum_salary"])

    # requested optional filters
    emp_col = _pick_col(df0, ["employmenttypes", "employment_type", "employmenttype", "employment"])
    pos_col = _pick_col(df0, ["positionlevels", "position_level", "positionlevel", "level"])

    # optional date column
    date_col = _pick_col(
        df0,
        ["date_posted", "posting_date", "dateofposting", "posted_date", "created_at", "createdat", "timestamp"],
    )

    # optional title column for table
    title_col = _pick_col(df0, ["job_title", "title", "jobtitle", "position", "role", "occupation"])

    out = pd.DataFrame(index=df0.index)
    out["sector"] = _to_str(df0[sector_col], unknown="Unknown")
    out["vacancies"] = _ensure_numeric(df0[vacancies_col]).fillna(0)
    out["applications"] = _ensure_numeric(df0[applications_col]).fillna(0)

    if salary_col:
        out["salary"] = _ensure_numeric(df0[salary_col])
    else:
        smin = _ensure_numeric(df0[salary_min_col]) if salary_min_col else pd.Series(np.nan, index=df0.index)
        smax = _ensure_numeric(df0[salary_max_col]) if salary_max_col else pd.Series(np.nan, index=df0.index)
        if salary_min_col and salary_max_col:
            out["salary"] = (smin + smax) / 2.0
        elif salary_min_col:
            out["salary"] = smin
        elif salary_max_col:
            out["salary"] = smax
        else:
            raise ValueError("Missing salary fields. Expected salary/avg_salary OR salary_min/salary_max.")

    out["salary"] = out["salary"].where(out["salary"] > 0, np.nan)
    bins = [-np.inf, 3000, 6000, 10000, 15000, np.inf]
    out["salary_band"] = pd.cut(out["salary"], bins=bins, labels=SALARY_BANDS, right=True)

    out["employmenttypes"] = _to_str(df0[emp_col], unknown="Unknown") if emp_col else "Unknown"
    out["positionlevels"] = _to_str(df0[pos_col], unknown="Unknown") if pos_col else "Unknown"
    out["job_title"] = _to_str(df0[title_col], unknown="Unknown") if title_col else "Unknown"
    out["date_posted"] = _coerce_datetime(df0[date_col]) if date_col else pd.NaT

    out["sector"] = out["sector"].astype("category")
    out["salary_band"] = pd.Categorical(out["salary_band"], categories=SALARY_BANDS, ordered=True)
    out["employmenttypes"] = out["employmenttypes"].astype("category")
    out["positionlevels"] = out["positionlevels"].astype("category")

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


def _choices(col: str) -> list[str]:
    df = load_data()
    return sorted(df[col].dropna().astype(str).unique().tolist())


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


def build_figs_and_summaries(dff: pd.DataFrame, title_prefix: str):
    postings, apps_sector, tsi_df, med_salary, postings_band, demand_order = compute_metrics_for_df(dff)

    if len(postings) == 0:
        empty = go.Figure()
        empty.update_layout(title=f"{title_prefix} — No data for selected filters", height=520, margin=dict(l=20, r=20, t=60, b=40))
        msg = "**Executive Summary:** No postings matched your current filter selections."
        return empty, msg, empty, msg, empty, msg, empty, msg

    combo = postings.merge(tsi_df, on="sector", how="left").sort_values("postings", ascending=True)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(y=combo["sector"].astype(str), x=combo["postings"], name="Job Postings", orientation="h",
                          hovertemplate="<b>%{y}</b><br>Postings: %{x:,.0f}<extra></extra>"))
    fig1.add_trace(go.Scatter(y=combo["sector"].astype(str), x=combo["tsi"], name="Composite TSI (0–100)",
                              mode="lines+markers", xaxis="x2",
                              hovertemplate="<b>%{y}</b><br>Composite TSI: %{x:.1f}/100<extra></extra>"))
    fig1.update_layout(
        title=dict(text=f"{title_prefix}: Demand (Postings) vs Composite TSI (0–100)", x=0.0, xanchor="left", y=0.98, yanchor="top"),
        height=520,
        margin=dict(l=170, r=70, t=110, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0),
        xaxis=dict(title="Job Postings", tickformat=",", showgrid=True, zeroline=False),
        xaxis2=dict(title=None, overlaying="x", side="top", range=[0, 100], showgrid=False, zeroline=False, ticks="outside", showticklabels=True),
        yaxis=dict(title="", automargin=True),
    )

    top_tsi = combo.sort_values("tsi", ascending=False).iloc[0]
    top_post = combo.sort_values("postings", ascending=False).iloc[0]
    s1 = f"**Executive Summary:** Highest shortage signal: **{top_tsi['sector']}** (TSI **{top_tsi['tsi']:.1f}/100**). Highest demand: **{top_post['sector']}** (**{top_post['postings']:,.0f}** postings)."

    if len(postings_band) == 0:
        fig2 = go.Figure()
        fig2.update_layout(title=f"{title_prefix}: Postings by Salary Band — no valid salary band rows", height=520, margin=dict(l=20, r=20, t=60, b=40))
        s2 = "**Executive Summary:** Salary-band charts are unavailable because salary bands are missing for the selected slice."
    else:
        bubble = postings_band.merge(postings, on="sector", how="left").merge(apps_sector, on="sector", how="left")
        bubble["salary_band"] = pd.Categorical(bubble["salary_band"], categories=SALARY_BANDS, ordered=True)
        fig2 = px.scatter(
            bubble,
            x="salary_band",
            y=bubble["sector"].astype(str),
            size="postings_band",
            color="postings_band",
            size_max=40,
            title=f"{title_prefix}: Job Postings by Salary Band (Bubble size & color = postings in band)",
        )
        fig2.update_traces(hovertemplate="<b>%{y}</b><br>Salary band: %{x}<br>Postings (this band): %{marker.size:,.0f}<br><extra></extra>")
        fig2.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20), xaxis_title="Salary band", yaxis_title="Sector")
        top_cell = bubble.sort_values("postings_band", ascending=False).iloc[0]
        s2 = f"**Executive Summary:** Strongest demand concentration: **{top_cell['sector']}** in **{top_cell['salary_band']}** with **{top_cell['postings_band']:,.0f}** postings."

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
    fig3.update_layout(
        title=f"{title_prefix}: Median Salary — Sectors Ordered by Demand (High → Low)",
        height=520,
        margin=dict(l=20, r=20, t=60, b=90),
        xaxis=dict(title="Sector (ranked by demand)", tickangle=45),
        yaxis=dict(title="Median Salary (SGD/month)", tickprefix="$", tickformat=","),
    )
    top_pay = cdf.sort_values("median_salary", ascending=False).iloc[0]
    s3 = "**Executive Summary:** Median salary is unavailable for this slice (salary values missing)." if pd.isna(top_pay["median_salary"]) else f"**Executive Summary:** Highest median salary sector: **{top_pay['sector']}** at **${top_pay['median_salary']:,.0f}**."

    if len(postings_band) == 0:
        fig4 = go.Figure()
        fig4.update_layout(title=f"{title_prefix}: Heatmap — no valid salary band rows", height=520, margin=dict(l=20, r=20, t=60, b=40))
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
        fig4.update_layout(
            title=f"{title_prefix}: Vacancies Heatmap — Sector vs Salary Band",
            height=520,
            margin=dict(l=170, r=40, t=60, b=60),
            xaxis=dict(title="Salary Band"),
            yaxis=dict(title="Sector", automargin=True),
        )
        band_totals = heat_df.groupby("salary_band", observed=True)["postings_band"].sum().sort_values(ascending=False)
        top_band = str(band_totals.index[0]) if len(band_totals) else "N/A"
        s4 = f"**Executive Summary:** Vacancy concentration is strongest in **{top_band}** for the current selection."

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
    fd, path = tempfile.mkstemp(prefix="labour_dashboard_", suffix=".csv")
    os.close(fd)
    dff.to_csv(path, index=False)
    return path


def cb_init():
    dff = default_slice()
    figs = build_figs_and_summaries(dff, title_prefix="Top 15 Sectors")
    table = build_table_preview(dff)
    return (*figs, table, gr.update(visible=False), False)


def cb_toggle_drawer(is_visible: bool):
    return gr.update(visible=not is_visible), (not is_visible)


def cb_generate(sectors, salary_bands, date_start, date_end, employment_types, position_levels):
    dff = apply_filters(sectors, salary_bands, date_start, date_end, employment_types, position_levels)
    figs = build_figs_and_summaries(dff, title_prefix="Filtered View")
    table = build_table_preview(dff)
    return (*figs, table, gr.update(visible=False), False)


def cb_default():
    dff = default_slice()
    figs = build_figs_and_summaries(dff, title_prefix="Top 15 Sectors")
    table = build_table_preview(dff)
    return (
        *figs,
        table,
        gr.update(value=[]),    # sector_dd
        gr.update(value=[]),    # salary_dd
        gr.update(value=None),  # date_start
        gr.update(value=None),  # date_end
        gr.update(value=[]),    # emp_dd
        gr.update(value=[]),    # pos_dd
        gr.update(visible=False),
        False,
    )


def cb_download(sectors, salary_bands, date_start, date_end, employment_types, position_levels):
    dff = apply_filters(sectors, salary_bands, date_start, date_end, employment_types, position_levels)
    return to_csv_tempfile(dff)


with gr.Blocks() as app:
    gr.Markdown("# Labour Intelligence Dashboard — Singapore (Composite TSI Normalized 0–100)")

    drawer_visible = gr.State(False)

    with gr.Row():
        btn_filters = gr.Button("☰ Filters", scale=1)
        btn_generate_btn = gr.Button("Generate", variant="primary", scale=1)
        btn_default_btn = gr.Button("Default", scale=1)

    with gr.Row():
        with gr.Column(visible=False) as drawer:
            gr.Markdown("## Filters (Deep-Dive)")
            gr.Markdown("Select **1–3 sectors**, refine using other filters, then click **Generate**.")

            sector_dd = gr.Dropdown(choices=_choices("sector"), multiselect=True, value=[], label="Sectors (select 1–3)", interactive=True)
            salary_dd = gr.Dropdown(choices=SALARY_BANDS, multiselect=True, value=[], label="Salary Bands", interactive=True)
            emp_dd = gr.Dropdown(choices=_choices("employmenttypes"), multiselect=True, value=[], label="Employment Types", interactive=True)
            pos_dd = gr.Dropdown(choices=_choices("positionlevels"), multiselect=True, value=[], label="Position Levels", interactive=True)

            with gr.Row():
                date_start = gr.DateTime(label="Posting date (start)", value=None)
                date_end = gr.DateTime(label="Posting date (end)", value=None)

        with gr.Column():
            with gr.Group():
                chartA = gr.Plot(label="Chart A — Demand vs Talent Shortage Index")
                sumA = gr.Markdown()
                with gr.Accordion("How we derive the numbers", open=False):
                    gr.Markdown(HOW_A)

            with gr.Group():
                chartB = gr.Plot(label="Chart B — Postings by Salary Band")
                sumB = gr.Markdown()
                with gr.Accordion("How we derive the numbers", open=False):
                    gr.Markdown(HOW_B)

            with gr.Group():
                chartC = gr.Plot(label="Chart C — Median Salary Trend")
                sumC = gr.Markdown()
                with gr.Accordion("How we derive the numbers", open=False):
                    gr.Markdown(HOW_C)

            with gr.Group():
                chartD = gr.Plot(label="Chart D — Vacancy Heatmap")
                sumD = gr.Markdown()
                with gr.Accordion("How we derive the numbers", open=False):
                    gr.Markdown(HOW_D)

            gr.Markdown("## Postings (Preview — Top 10)")
            postings_table = gr.Dataframe(interactive=False, wrap=True)

            download_btn = gr.DownloadButton("Download CSV (based on current filters)")

    app.load(
        fn=cb_init,
        inputs=[],
        outputs=[chartA, sumA, chartB, sumB, chartC, sumC, chartD, sumD, postings_table, drawer, drawer_visible],
    )

    btn_filters.click(fn=cb_toggle_drawer, inputs=[drawer_visible], outputs=[drawer, drawer_visible])

    btn_generate_btn.click(
        fn=cb_generate,
        inputs=[sector_dd, salary_dd, date_start, date_end, emp_dd, pos_dd],
        outputs=[chartA, sumA, chartB, sumB, chartC, sumC, chartD, sumD, postings_table, drawer, drawer_visible],
    )

    download_btn.click(
        fn=cb_download,
        inputs=[sector_dd, salary_dd, date_start, date_end, emp_dd, pos_dd],
        outputs=[download_btn],
    )

    btn_default_btn.click(
        fn=cb_default,
        inputs=[],
        outputs=[
            chartA, sumA, chartB, sumB, chartC, sumC, chartD, sumD,
            postings_table,
            sector_dd, salary_dd, date_start, date_end, emp_dd, pos_dd,
            drawer,
            drawer_visible,
        ],
    )

# HF Spaces: bind to 0.0.0.0 and use injected PORT. Do NOT use share=True.
app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), show_api=False)
