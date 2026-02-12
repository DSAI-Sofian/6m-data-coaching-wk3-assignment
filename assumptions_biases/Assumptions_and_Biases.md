
# Assumptions and Biases — Singapore Labour Intelligence Dashboard

## Purpose
This document outlines the key **assumptions, limitations, and potential biases** underlying the construction of the Labour Intelligence Dashboard. Understanding these factors is essential for correct interpretation of indicators such as Demand, Composite Talent Shortage Index (TSI), Salary Metrics, and Vacancy Distribution.

---

## Core Assumptions

### 1. Job Postings Represent Labour Demand
- The dashboard assumes **number of vacancies (job postings)** is a valid proxy for employer demand.
- It assumes postings reflect **real hiring intent**, not duplicated, evergreen, or exploratory listings.

**Implication:** Demand may be overstated in sectors with repeated postings or recruitment churn.

---

### 2. Applications Represent Labour Supply
- Total job applications are used as a proxy for **available labour supply**.
- Assumes applications reflect **genuine, qualified jobseekers**.

**Implication:** Supply may be overstated where:
- Applicants apply to many roles simultaneously
- Low-quality / mismatched applications inflate counts

---

### 3. Composite Talent Shortage Index (TSI) Reflects Labour Tightness
The dashboard defines shortage as a blend of:
- **Demand intensity** (vacancy volume)
- **Supply tightness** (vacancies relative to applications)

TSI is normalized across sectors (0–100), meaning:
- It is a **relative**, not absolute, measure.
- A sector with TSI=100 has the strongest shortage *compared to other sectors*, not in absolute terms.

**Implication:** TSI does not measure:
- Absolute workforce shortage
- Skill mismatch directly
- Structural vs cyclical labour gaps

---

### 4. Salary Represents Market Price for Labour
- Salary is derived from available fields (average or midpoint of min/max).
- Median salary is used to reduce skew from outliers.

**Implication:** Salary may not reflect:
- Bonus / variable pay
- Total compensation
- Seniority mix within sectors

---

### 5. Salary Bands Represent Labour Segmentation
Salary bands are assumed to reflect different labour strata:
- Entry-level
- Mid-skilled
- Professional
- Senior / Specialist

**Implication:** Band boundaries are fixed and may not perfectly match real labour market segmentation across sectors.

---

## Data Limitations

### 1. No Unemployment Data by Sector
The dashboard cannot compute MOM-style **Vacancy-to-Unemployed ratio (V/U)**, so applications are used as a proxy for labour supply.

### 2. No Skill-Level Matching
The dashboard does not distinguish:
- Qualified vs unqualified applicants
- Skill mismatches
- Role seniority differences

### 3. Static Snapshot (No Time Dimension)
Unless extended, the dashboard reflects a **cross-sectional view**, not trends over time.

---

## Potential Biases

### Supply-Side Bias
- High-application sectors may appear less tight even if skills mismatch exists.

### Demand Duplication Bias
- Sectors with repeated or persistent postings may appear to have stronger demand.

### Salary Reporting Bias
- Incomplete salary disclosure can skew median salary calculations.

### Sector Aggregation Bias
- Aggregating diverse roles within a sector may mask internal shortages or surpluses.

### Normalization Bias
- Min–max normalization compresses mid-range sectors and exaggerates extremes.

---

## Interpretation Guidance

The dashboard should be used to identify:
- Relative labour tightness
- Demand concentration
- Salary positioning
- Sectoral imbalance patterns

It should **not be used alone** to conclude:
- Exact labour shortage size
- Skill-gap severity
- Workforce productivity
- Structural unemployment

---

## Future Improvements (Recommended)
- Add unemployment proxy or labour force estimates
- Introduce skill/role granularity
- Add time-series trend analysis
- Detect structural vs cyclical shortage
- Incorporate workforce transformation indicators

---

**End of Document**
