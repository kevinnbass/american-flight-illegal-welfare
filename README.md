# Domestic Migration vs Welfare Benefits for Illegal Immigrants

This repository analyzes the relationship between state-level welfare benefits for illegal immigrants and net domestic migration patterns in the United States.

## Key Findings

### Primary Result
**States that offer welfare benefits to illegal immigrants experience significantly higher population outflow than states that don't.**

| Metric | Value |
|--------|-------|
| Spearman correlation | rho = -0.57 (p < 0.0001) |
| Pearson correlation | r = -0.45 (p = 0.001) |
| Point-biserial correlation | r = -0.55 (p < 0.0001) |

### Inflow Rates by Benefit Status
- **States with zero benefits:** 72% have net population inflow (23 of 32)
- **States with any benefits:** 24% have net population inflow (4 of 17)

### Top 10 Analysis
- **9 of 10** top outflow states offer benefits to illegal immigrants
- **1 of 10** top inflow states offers benefits to illegal immigrants
- Fisher's exact test: **p = 0.0011**

### Top 10 Outflow States
| State | Migration % | Benefits |
|-------|-------------|----------|
| New York | -0.90% | Yes |
| California | -0.88% | Yes |
| Louisiana | -0.68% | No |
| Illinois | -0.65% | Yes |
| Maryland | -0.53% | Yes |
| Massachusetts | -0.52% | Yes |
| New Jersey | -0.52% | Yes |
| District of Columbia | -0.38% | Yes |
| Rhode Island | -0.30% | Yes |
| Washington | -0.22% | Yes |

### Top 10 Inflow States
| State | Migration % | Benefits |
|-------|-------------|----------|
| South Carolina | +1.49% | No |
| Delaware | +0.96% | No |
| North Carolina | +0.91% | No |
| Tennessee | +0.85% | No |
| Florida | +0.81% | No |
| Montana | +0.80% | No |
| Idaho | +0.75% | No |
| Maine | +0.69% | Yes |
| Texas | +0.63% | No |
| Alabama | +0.59% | No |

## Data Sources

### Migration Data
- **Source:** US Census Bureau Population Estimates 2023
- **Metric:** Net domestic migration (interstate moves only, excludes international migration)
- **Year:** 2023

### Welfare Policy Data
Five benefit categories are tracked for illegal immigrant eligibility:

| Benefit | Source |
|---------|--------|
| Health coverage (children) | KFF, NILC |
| Health coverage (adults) | KFF, NILC |
| Health coverage (seniors 65+) | KFF, NILC |
| Food assistance (SNAP-like) | NILC |
| State EITC (ITIN filers) | ITEP |

## Methodology

### Scope
- **Continental US only** (excludes Alaska and Hawaii)
- 49 states + District of Columbia = 49 observations

### Analysis Approach
1. Merged state welfare policies with Census migration data
2. Calculated migration as percentage of state population
3. Created total benefit score (0-5) for each state
4. Computed correlations and statistical tests
5. Analyzed top 10 inflow/outflow states

### Statistical Tests
- **Spearman correlation:** Non-parametric rank correlation (robust to outliers)
- **Pearson correlation:** Linear correlation coefficient
- **Point-biserial correlation:** Binary (any benefit vs none) correlation
- **Fisher's exact test:** Contingency table analysis for top 10 comparison

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── state_policies.csv      # Welfare policy data by state
│   └── net_domestic_migration_2023.csv
├── src/
│   └── analysis.py             # Self-contained analysis script
└── output/
    ├── scatter_benefits_migration.png
    ├── scatter_binary_migration.png
    ├── bar_chart_migration.png
    ├── map_migration_benefits.png
    └── social_graphic.png
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python src/analysis.py
```

This generates all figures and prints statistics to the console.

## Output Figures

### Scatter Plot: Benefits vs Migration
Shows relationship between total number of benefit types (0-5) and net migration percentage.

### Binary Scatter: Any Benefits vs Migration
Compares states with any benefits to states with none, showing mean migration for each group.

### Bar Chart: Migration by Benefit Level
Groups states into 0, 1-4, and 5 benefits categories showing average migration.

### Map: Geographic Distribution
Choropleth map showing migration patterns with welfare benefit symbols overlaid.

### Social Graphic
Summary graphic for social media showing top 10 inflow/outflow state comparisons.

## Limitations

1. **Correlation is not causation.** This analysis shows association, not causal relationship.
2. **Confounding variables.** Many factors affect migration (cost of living, climate, jobs, taxes, etc.).
3. **Single year data.** Uses 2023 migration only; trends may vary year to year.
4. **Binary classification.** Welfare policies have nuances not captured by 0/1 coding.
5. **Population weighting.** Analysis treats all states equally regardless of population size.

## License

MIT License
