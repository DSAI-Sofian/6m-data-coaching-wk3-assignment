import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from functools import lru_cache

DATA_PATH = "SGJobData_cleaned.csv"

SALARY_BANDS = ["<$3k", "$3k - $6k", "$7k - $10k", "$11k - $15k", ">$16k"]

# Composite TSI weight (0–1):
# alpha closer to 1 => more weight on demand volume; closer to 0 => more weight on tightness pressure (vacancies/applications).
COMPOSITE_ALPHA = 0.4


# -----------------------------
# Methodology text blocks
# -----------------------------
HOW_A = f"""### How we derive the numbers
- **Job Postings (Demand)** = sum of `vacancies` per sector (Top 15 sectors by total postings).
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
- Sectors on the x-axis are ordered by **demand (postings)** from highest → lowest.
"""

HOW_D = """### How we derive the numbers
- Heatmap cell value (**Postings**) = sum of `vacancies` for each **sector × salary band**.
- Only rows with a valid `salary_band` are included (rows with missing/unparsable salary are excluded from band charts).
"""


# -----------------------------
# Helpers: robust column mapping
# -----------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        c = c.lower()
        if c in df.columns:
            return c
    return None


def _ensure_numeric(s: pd.Series) -> pd.Series:
    # Handles "$5,000" / "5000" / " 5,000 " robustly
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _minmax_norm(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    return (arr - mn) / (mx - mn + 1e-9)


# -----------------------------
# Cached data loading
# -----------------------------
@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = _normalize_cols(df)

    # sector
    sector_col = _pick_col(df, ["sector", "industry", "jobsector", "job_sector", "sector_name"])
    if not sector_col:
        raise ValueError("Missing required column: sector (or equivalent).")

    # vacancies/postings
    vacancies_col = _pick_col(
        df,
        ["vacancies", "numberofvacancies", "number_of_vacancies", "jobpostings", "postings"],
    )
    if not vacancies_col:
        raise ValueError("Missing required column: vacancies (e.g. numberOfVacancies).")

    # applications
    applications_col = _pick_col(
        df,
        ["applications", "metadata_totalnumberjobapplication", "totalapplications", "total_number_job_application"],
    )
    if not applications_col:
        raise ValueError("Missing required column: applications (e.g. metadata_totalNumberJobApplication).")

    # salary (try avg, else min/max)
    salary_col = _pick_col(df, ["salary", "avg_salary", "average_salary", "salary_average"])
    salary_min_col = _pick_col(df, ["salary_min", "salary_minimum", "minimum_salary"])
    salary_max_col = _pick_col(df, ["salary_max", "salary_maximum", "maximum_salary"])

    out = pd.DataFrame()
    out["sector"] = df[sector_col].astype(str).fillna("Unknown")
    out["vacancies"] = _ensure_numeric(df[vacancies_col]).fillna(0)
    out["applications"] = _ensure_numeric(df[applications_col]).fillna(0)

    if salary_col:
        out["salary"] = _ensure_numeric(df[salary_col])
    else:
        smin = _ensure_numeric(df[salary_min_col]) if salary_min_col else pd.Series(np.nan, index=df.index)
        smax = _ensure_numeric(df[salary_max_col]) if salary_max_col else pd.Series(np.nan, index=df.index)

        if salary_min_col and salary_max_col:
            out["salary"] = (smin + smax) / 2.0
        elif salary_min_col:
            out["salary"] = smin
        elif salary_max_col:
            out["salary"] = smax
        else:
            raise ValueError(
                "Missing salary fields. Expected one of: salary/average_salary OR salary_minimum/salary_maximum."
            )

    # keep only positive salaries as valid
    out["salary"] = out["salary"].where(out["salary"] > 0, np.nan)

    # salary bands (fixed 5 bands)
    bins = [-np.inf, 3000, 6000, 10000, 15000, np.inf]
    out["salary_band"] = pd.cut(out["salary"], bins=bins, labels=SALARY_BANDS, right=True)

    # ordering + faster groupby
    out["sector"] = out["sector"].astype("category")
    out["salary_band"] = pd.Categorical(out["salary_band"], categories=SALARY_BANDS, ordered=True)

    return out


# -----------------------------
# Cached aggregations
# -----------------------------
@lru_cache(maxsize=64)
def compute_metrics():
    df = load_data()

    # Top 15 sectors by total vacancies (demand proxy)
    top15 = (
        df.groupby("sector", observed=True)["vacancies"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    dft = df[df["sector"].isin(top15)].copy()

    # Total postings per sector
    postings = (
        dft.groupby("sector", observed=True)["vacancies"]
        .sum()
        .reset_index(name="postings")
        .sort_values("postings", ascending=False)
    )

    # Total applications per sector (context + pressure denominator)
    apps_sector = (
        dft.groupby("sector", observed=True)["applications"]
        .sum()
        .reset_index(name="applications_sector")
    )

    # ---- Composite TSI (0–100) ----
    # Pressure component: postings / (applications + 1)
    joined = postings.merge(apps_sector, on="sector", how="left").copy()
    pressure = joined["postings"] / (joined["applications_sector"] + 1.0)

    D = _minmax_norm(joined["postings"])
    P = _minmax_norm(pressure)
    C = COMPOSITE_ALPHA * D + (1.0 - COMPOSITE_ALPHA) * P
    tsi = 100.0 * _minmax_norm(C)

    tsi_df = pd.DataFrame({"sector": joined["sector"], "tsi": tsi})

    # Median salary (sector)
    med_salary = (
        dft.groupby("sector", observed=True)["salary"]
        .median()
        .reset_index(name="median_salary")
    )

    # Band-based charts require valid salary_band
    dft_band = dft.dropna(subset=["salary_band"]).copy()

    # Postings by sector & band  ✅ used for Chart 2 + Chart 4
    postings_band = (
        dft_band.groupby(["sector", "salary_band"], observed=True)["vacancies"]
        .sum()
        .reset_index(name="postings_band")
    )

    return postings, apps_sector, tsi_df, med_salary, postings_band


# -----------------------------
# Dashboard builder
# -----------------------------
def build_dashboard():
    postings, apps_sector, tsi_df, med_salary, postings_band = compute_metrics()

    # Order sectors by demand (high -> low)
    demand_order = postings["sector"].astype(str).tolist()

    # =========================
    # (A) Combo chart (horizontal): postings + Composite TSI (0–100)
    # =========================
    combo = postings.merge(tsi_df, on="sector", how="left").copy()
    combo = combo.sort_values("postings", ascending=True)  # ascending for horizontal bars

    fig1 = go.Figure()

    fig1.add_trace(
        go.Bar(
            y=combo["sector"].astype(str),
            x=combo["postings"],
            name="Job Postings",
            orientation="h",
            hovertemplate="<b>%{y}</b><br>Postings: %{x:,.0f}<extra></extra>",
        )
    )

    fig1.add_trace(
        go.Scatter(
            y=combo["sector"].astype(str),
            x=combo["tsi"],
            name="Composite TSI (0–100)",
            mode="lines+markers",
            xaxis="x2",
            hovertemplate="<b>%{y}</b><br>Composite TSI: %{x:.1f}/100<extra></extra>",
        )
    )

    # Layout tuned to prevent top text overlap (title/legend/xaxis2)
    fig1.update_layout(
        title=dict(
            text="Top 15 Sectors: Demand (Postings) vs Composite Talent Shortage Index (0–100)",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        height=560,
        margin=dict(l=170, r=70, t=120, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="left",
            x=0.0,
        ),
        xaxis=dict(
            title="Job Postings",
            tickformat=",",
            showgrid=True,
            zeroline=False,
        ),
        xaxis2=dict(
            title=None,  # remove to avoid overlap
            overlaying="x",
            side="top",
            range=[0, 100],
            showgrid=False,
            zeroline=False,
            ticks="outside",
            showticklabels=True,
        ),
        yaxis=dict(title="", automargin=True),
    )

    top_tsi = combo.sort_values("tsi", ascending=False).iloc[0]
    top_post = combo.sort_values("postings", ascending=False).iloc[0]
    s1 = (
        f"**Executive Summary:** Highest composite shortage signal: **{top_tsi['sector']}** "
        f"(Composite TSI **{top_tsi['tsi']:.1f}/100**). Highest demand volume: **{top_post['sector']}** "
        f"(**{top_post['postings']:,.0f}** postings)."
    )

    # =========================
    # (B) Bubble chart: postings by sector & salary band
    # =========================
    bubble = postings_band.merge(postings, on="sector", how="left").merge(apps_sector, on="sector", how="left")
    bubble["salary_band"] = pd.Categorical(bubble["salary_band"], categories=SALARY_BANDS, ordered=True)

    fig2 = px.scatter(
        bubble,
        x="salary_band",
        y=bubble["sector"].astype(str),
        size="postings_band",
        color="postings_band",
        size_max=40,
        title="Job Postings Volume by Salary Band (Bubble size & color = postings in band)",
    )
    fig2.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Salary band: %{x}<br>"
            "Postings (this band): %{marker.size:,.0f}<br>"
            "<extra></extra>"
        )
    )
    fig2.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Salary band",
        yaxis_title="Sector",
    )

    top_cell = bubble.sort_values("postings_band", ascending=False).iloc[0]
    s2 = (
        f"**Executive Summary:** Strongest demand concentration: **{top_cell['sector']}** in **{top_cell['salary_band']}** "
        f"with **{top_cell['postings_band']:,.0f}** postings. Large bubbles at higher bands indicate senior/specialist demand."
    )

    # =========================
    # (C) Line plot: median salary ordered by demand (high -> low)
    # =========================
    cdf = postings.merge(med_salary, on="sector", how="left").copy()
    cdf = cdf.sort_values("postings", ascending=False)

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=cdf["sector"].astype(str),
            y=cdf["median_salary"],
            mode="lines+markers",
            name="Median Salary",
            customdata=np.stack([cdf["postings"]], axis=1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Median salary: $%{y:,.0f}<br>"
                "Postings (demand): %{customdata[0]:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    fig3.update_layout(
        title="Median Salary (Line) — Top 15 Sectors Ordered by Demand (High → Low)",
        height=560,
        margin=dict(l=20, r=20, t=60, b=90),
        xaxis=dict(title="Sector (ranked by demand)", tickangle=45),
        yaxis=dict(title="Median Salary (SGD/month)", tickprefix="$", tickformat=","),
    )

    top_pay = cdf.sort_values("median_salary", ascending=False).iloc[0]
    s3 = (
        f"**Executive Summary:** Highest median salary sector: **{top_pay['sector']}** at **${top_pay['median_salary']:,.0f}**. "
        f"Compare pay levels against demand ordering to spot mismatches (high demand but low pay, or vice versa)."
    )

    # =========================
    # (D) Heatmap: postings by sector vs salary band (fixed axes)
    # =========================
    heat_df = postings_band.copy()
    heat_df["salary_band"] = pd.Categorical(heat_df["salary_band"], categories=SALARY_BANDS, ordered=True)

    heat_pivot = (
        heat_df.pivot(index="sector", columns="salary_band", values="postings_band")
        .reindex(index=pd.Index(demand_order, name="sector"), fill_value=0)
        .reindex(columns=SALARY_BANDS, fill_value=0)
        .fillna(0)
    )

    fig4 = go.Figure(
        data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.astype(str),
            y=heat_pivot.index.astype(str),
            colorbar=dict(title="Postings"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Salary band: %{x}<br>"
                "Postings: %{z:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    fig4.update_layout(
        title="Vacancies Heatmap — Sector vs Salary Band",
        height=560,
        margin=dict(l=170, r=40, t=60, b=60),
        xaxis=dict(title="Salary Band"),
        yaxis=dict(title="Sector", automargin=True),
    )

    band_totals = heat_df.groupby("salary_band", observed=True)["postings_band"].sum().sort_values(ascending=False)
    top_band = str(band_totals.index[0]) if len(band_totals) else "N/A"
    s4 = (
        f"**Executive Summary:** Vacancy concentration is strongest in **{top_band}**. "
        f"Use band-level concentration to calibrate sourcing intensity and compensation strategy."
    )

    return fig1, s1, fig2, s2, fig3, s3, fig4, s4


# -----------------------------
# Gradio UI (2×2)
# -----------------------------
with gr.Blocks() as app:
    gr.Markdown("# Labour Intelligence Dashboard — Singapore (Composite TSI Normalized 0–100)")

    fig1, s1, fig2, s2, fig3, s3, fig4, s4 = build_dashboard()

    with gr.Row():
        with gr.Column():
            gr.Plot(fig1)
            gr.Markdown(s1)
            gr.Markdown(HOW_A)
        with gr.Column():
            gr.Plot(fig2)
            gr.Markdown(s2)
            gr.Markdown(HOW_B)

    with gr.Row():
        with gr.Column():
            gr.Plot(fig3)
            gr.Markdown(s3)
            gr.Markdown(HOW_C)
        with gr.Column():
            gr.Plot(fig4)
            gr.Markdown(s4)
            gr.Markdown(HOW_D)

app.launch(server_name="0.0.0.0", server_port=7860)
