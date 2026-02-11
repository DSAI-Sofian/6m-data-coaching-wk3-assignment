---
title: Labour Intelligence Dashboard (SG)
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# Labour Intelligence Dashboard — Singapore

Interactive 2×2 dashboard built with **Gradio + Plotly** using `SGJobData_cleaned.csv`.

## What you get (2×2 layout)
1. **Combo chart**: Top 15 sectors by job postings (bar) + **Talent Shortage Index (TSI)** (line)
2. **Bubble/Scatter**: Applications (Top 15 sectors) vs Salary band (bubble size = postings)
3. **Median salary**: Median salary for Top 15 demand sectors
4. **Heatmap**: Vacancies (Top 15 sectors) vs Salary band

## Talent Shortage Index (0–100)
A Singapore-style shortage pressure proxy is used:

- **Raw pressure ratio**: `vacancies / (applications + 1)` (aggregated by sector)
- **Normalization (0–100)**: Min-Max scaling across the Top 15 sectors  
  `TSI = 100 * (x - min(x)) / (max(x) - min(x))`

Higher TSI = higher shortage pressure relative to other sectors in the dashboard.

## Salary bands (fixed)
- `<$3k`
- `$3k - $6k`
- `$7k - $10k`
- `$11k - $15k`
- `>$16k`

## Run locally
```bash
pip install -r requirements.txt
python app.py
