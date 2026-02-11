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

# Singapore Labour Intelligence Dashboard

**Production-Grade Labour Market Analytics Platform (Python / Gradio / Plotly)**

---

## Overview

The **Singapore Labour Intelligence Dashboard** is a production-ready analytics system designed to extract actionable labour market intelligence from large-scale job vacancy datasets (~1M+ rows).

The platform enables **real-time labour diagnostics, workforce pressure detection, and strategic manpower insight generation** using a Composite Talent Shortage Index (TSI) and multi-dimensional labour analytics.

Built for **policy analysis, workforce planning, recruitment intelligence, and macro labour monitoring**.

---

## Key Capabilities

### Core Labour Intelligence Engine

* Composite Talent Shortage Index (0–100)

  * Demand Pressure + Talent Tightness
* Structural shortage detection
* Wage vs demand imbalance analysis
* Hiring pressure & labour stress signals

---

### 4-Panel Analytics Dashboard

1. **Demand vs Composite TSI**

   * Sector hiring pressure visualization
   * Shortage signal extraction

2. **Salary Band Distribution**

   * Market wage structure & compression analysis

3. **Median Salary Trend**

   * Wage movement vs labour tightness

4. **Vacancy Heatmap**

   * Structural demand concentration mapping

---

### Bloomberg-Style Professional UI

* Dark theme labour intelligence layout
* Center-aligned chart architecture
* KPI header panel
* Overlay filter drawer
* Clean statistical visualization
* Boardroom-ready display

---

### KPI Statistics Panel

Shows both:

* Full dataset metrics
* Filtered view metrics

Includes:

* Total vacancies
* Median salary
* Composite TSI
* Sector distribution
* Demand pressure indicators

---

### Data & Performance Architecture

* Lazy-loaded dataset (~1M rows)
* Memory-safe loading
* Optimized filtering pipeline
* Fast redraw for analytics
* HuggingFace Spaces compatible
* Gradio 5.x supported

---

### Filters

* Sector (Multi-select, up to 3)
* Salary Band
* Employment Type
* Position Level
* Posting Date Range

All charts and KPIs dynamically update.

---

### Export Features

* CSV export (filtered dataset)
* Board-level PDF report

  * Embedded charts
  * KPI snapshot
  * Professional formatting

---

## Labour Intelligence Use Cases

* Sector manpower outlook
* Structural shortage detection
* Wage-demand mismatch identification
* Hiring pressure monitoring
* Workforce planning insights
* Policy & labour market analysis
* Recruitment intelligence
* Economic manpower diagnostics

---

## Technology Stack

* Python
* Pandas
* Plotly
* Gradio 5.x
* Matplotlib (PDF rendering)
* NumPy

---

## Deployment

### HuggingFace Spaces (Recommended)

Compatible with:

* CPU environments
* Large dataset handling
* Gradio 5.x runtime

---

## Performance Optimizations

* Lazy dataset loading
* Memory-safe filtering
* Chart redraw optimization
* Export caching
* UI alignment stabilization
* Margin and layout tuning

---

## Roadmap / Intelligence Expansion

* Predictive labour shortage model
* Wage inflation signal
* Hiring pressure index
* Sector outlook engine
* Workforce planning analytics
* Policy intelligence module
* Scenario simulation (labour shocks)

---

## Disclaimer

This dashboard is intended for **labour market intelligence and analytical purposes only**.
Indicators such as Composite TSI are derived metrics and should be interpreted alongside broader economic and policy context.

---

## Author

Developed through iterative engineering between S. Sofian and ChatGPT, UX refinement, and labour intelligence modelling across multiple production branches.
