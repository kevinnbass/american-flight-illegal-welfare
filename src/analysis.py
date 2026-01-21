"""
Domestic Migration vs Welfare Benefits for Illegal Immigrants
=============================================================

This script analyzes the relationship between state-level welfare benefits
for illegal immigrants and net domestic migration patterns.

Data Sources:
- Migration: US Census Bureau Population Estimates 2023
- Welfare policies: KFF, NILC (Health Coverage); NILC (Food Assistance); ITEP (State EITC)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'output'


def set_style():
    """Set consistent matplotlib style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_data():
    """Load and merge policy and migration data."""
    # Load state policies
    policies = pd.read_csv(DATA_DIR / 'state_policies.csv')

    # Load migration data
    migration = pd.read_csv(DATA_DIR / 'net_domestic_migration_2023.csv')

    # Merge
    df = policies.merge(migration, on=['state', 'abbrev'], how='left')

    # Calculate total benefits (0-5)
    benefit_cols = ['health_children', 'health_adults', 'health_seniors', 'food', 'eitc']
    df['total_benefits'] = df[benefit_cols].sum(axis=1)

    # Binary: has any benefit
    df['has_any_benefit'] = (df['total_benefits'] > 0).astype(int)

    # Load population for percentage calculation
    # Population data (2023 estimates in thousands)
    pop_2023 = {
        'AL': 5108, 'AK': 733, 'AZ': 7431, 'AR': 3067, 'CA': 38965, 'CO': 5877,
        'CT': 3617, 'DE': 1031, 'FL': 22975, 'GA': 11029, 'HI': 1435, 'ID': 1964,
        'IL': 12516, 'IN': 6862, 'IA': 3207, 'KS': 2937, 'KY': 4526, 'LA': 4573,
        'ME': 1395, 'MD': 6180, 'MA': 7001, 'MI': 10037, 'MN': 5737, 'MS': 2939,
        'MO': 6196, 'MT': 1133, 'NE': 1978, 'NV': 3194, 'NH': 1402, 'NJ': 9290,
        'NM': 2114, 'NY': 19571, 'NC': 10835, 'ND': 783, 'OH': 11785, 'OK': 4053,
        'OR': 4240, 'PA': 12972, 'RI': 1110, 'SC': 5373, 'SD': 919, 'TN': 7126,
        'TX': 30503, 'UT': 3417, 'VT': 648, 'VA': 8683, 'WA': 7812, 'WV': 1770,
        'WI': 5910, 'WY': 584, 'DC': 678
    }

    df['population_2023'] = df['abbrev'].map(pop_2023) * 1000
    df['migration_pct'] = (df['net_domestic_migration_2023'] / df['population_2023']) * 100

    return df


def get_continental(df):
    """Exclude Alaska and Hawaii."""
    return df[~df['abbrev'].isin(['AK', 'HI'])].copy()


def create_scatter_plot(df, output_path):
    """Create scatter plot: total benefits vs migration %."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {0: '#deebf7', 1: '#9ecae1', 2: '#6baed6', 3: '#4292c6', 4: '#2171b5', 5: '#084594'}

    np.random.seed(42)
    for score in sorted(df['total_benefits'].unique()):
        subset = df[df['total_benefits'] == score]
        jitter = np.random.uniform(-0.15, 0.15, len(subset))

        ax.scatter(
            subset['total_benefits'] + jitter,
            subset['migration_pct'],
            s=100,
            c=colors.get(int(score), '#084594'),
            alpha=0.7,
            edgecolors='white',
            linewidth=1,
            label=f'{int(score)} benefits (n={len(subset)})'
        )

        # Label notable states
        for idx, row in subset.iterrows():
            if abs(row['migration_pct']) > 0.5:
                ax.annotate(
                    row['abbrev'],
                    (score + jitter[list(subset.index).index(idx)], row['migration_pct']),
                    fontsize=8, ha='center', va='bottom'
                )

    ax.axhline(y=0, color='#666666', linestyle='--', linewidth=1, alpha=0.5)

    # Trend line
    z = np.polyfit(df['total_benefits'], df['migration_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(-0.2, 5.2, 100)
    pearson_r, pearson_p = stats.pearsonr(df['total_benefits'], df['migration_pct'])
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend (r={pearson_r:.2f})')

    ax.set_xlabel('Total Number of Benefit Types (0-5)', fontsize=12)
    ax.set_ylabel('Net Domestic Migration (% of Population)', fontsize=12)
    ax.set_title('Net Domestic Migration vs Total Welfare Benefits for Illegal Immigrants\n'
                 '(Continental US - Excludes AK & HI)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(-0.5, 5.5)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.legend(loc='upper right', fontsize=9, frameon=True)

    # Stats
    spearman_r, spearman_p = stats.spearmanr(df['total_benefits'], df['migration_pct'])
    stats_text = f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})\nSpearman rho = {spearman_r:.3f} (p = {spearman_p:.4f})"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.text(0.02, -0.02, 'Sources: Census Bureau Population Estimates (Migration), NCSL/KFF/NILC/ITEP (Welfare)',
             fontsize=9, color='#888888', style='italic')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {output_path.name}')


def create_binary_scatter(df, output_path):
    """Create scatter plot: binary benefit status vs migration %."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    np.random.seed(42)

    for has_benefit in [0, 1]:
        subset = df[df['has_any_benefit'] == has_benefit]
        jitter = np.random.uniform(-0.15, 0.15, len(subset))

        color = '#084594' if has_benefit else '#deebf7'
        label = f"Has Benefits (n={len(subset)})" if has_benefit else f"No Benefits (n={len(subset)})"

        ax.scatter(
            subset['has_any_benefit'] + jitter,
            subset['migration_pct'],
            s=100, c=color, alpha=0.7,
            edgecolors='white', linewidth=1, label=label
        )

        for idx, row in subset.iterrows():
            if abs(row['migration_pct']) > 0.5:
                ax.annotate(row['abbrev'],
                           (has_benefit + jitter[list(subset.index).index(idx)], row['migration_pct']),
                           fontsize=8, ha='center', va='bottom')

    # Mean lines
    for has_benefit in [0, 1]:
        subset = df[df['has_any_benefit'] == has_benefit]
        mean_val = subset['migration_pct'].mean()
        ax.hlines(mean_val, has_benefit - 0.3, has_benefit + 0.3, colors='red', linewidth=3,
                  label=f'Mean: {mean_val:+.2f}%')

    ax.axhline(y=0, color='#666666', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Benefits', 'Has Benefits'])
    ax.set_xlabel('Welfare Benefit Status', fontsize=12)
    ax.set_ylabel('Net Domestic Migration (% of Population)', fontsize=12)
    ax.set_title('Net Domestic Migration by Welfare Benefit Status\n(Continental US - Excludes AK & HI)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Point-biserial correlation
    r, p = stats.pointbiserialr(df['has_any_benefit'], df['migration_pct'])
    ax.text(0.02, 0.02, f'Point-biserial r = {r:.3f} (p = {p:.4f})', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {output_path.name}')


def create_bar_chart(df, output_path):
    """Create bar chart: 0, 1-4, 5 benefits grouping."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    def categorize(n):
        if n == 0: return '0'
        elif n <= 4: return '1-4'
        else: return '5'

    df['benefit_group'] = df['total_benefits'].apply(categorize)

    group_order = ['0', '1-4', '5']
    grouped = df.groupby('benefit_group').agg({
        'migration_pct': 'mean',
        'state': 'count'
    }).reindex(group_order).reset_index()
    grouped.columns = ['benefit_group', 'avg_migration', 'n_states']

    colors = ['#deebf7', '#4292c6', '#084594']
    x = np.arange(len(grouped))
    bars = ax.bar(x, grouped['avg_migration'], width=0.5, color=colors, edgecolor='#333333', linewidth=1)

    y_min, y_max = grouped['avg_migration'].min(), grouped['avg_migration'].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.35, y_max + y_range * 0.3)

    for bar, row in zip(bars, grouped.itertuples()):
        height = bar.get_height()
        y_pos = height + y_range * 0.05 if height >= 0 else height - y_range * 0.05
        va = 'bottom' if height >= 0 else 'top'
        ax.annotate(f'{height:+.2f}%\n(n={row.n_states})',
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    ha='center', va=va, fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.5)
    ax.set_ylabel('Avg Net Migration (% of Pop)', fontsize=11)
    ax.set_title('Average Net Domestic Migration by Welfare Benefit Level\n(Continental US - Excludes AK & HI)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['No Benefits', '1-4 Benefits', '5 Benefits'], fontsize=11)
    ax.set_xlabel('Welfare Benefit Level', fontsize=11)

    spearman_r, spearman_p = stats.spearmanr(df['total_benefits'], df['migration_pct'])
    ax.text(0.02, 0.05, f'Spearman rho = {spearman_r:.3f} (p = {spearman_p:.4f})',
            transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {output_path.name}')


def create_map(df, output_path):
    """Create choropleth map with welfare symbols."""
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not installed, skipping map generation")
        return

    set_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    us_states_url = 'https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_5m.zip'
    states_gdf = gpd.read_file(us_states_url)
    states_gdf = states_gdf[states_gdf['STATEFP'].astype(int) <= 56]
    states_gdf = states_gdf.merge(df, left_on='STUSPS', right_on='abbrev', how='left')

    for col in ['health_children', 'health_adults', 'health_seniors', 'food', 'eitc', 'migration_pct']:
        states_gdf[col] = states_gdf[col].fillna(0)

    states_gdf = states_gdf.to_crs('ESRI:102003')

    def get_color(pct):
        if pct >= 0: return '#d1e5f0'
        elif pct >= -0.3: return '#fddbc7'
        elif pct >= -0.6: return '#ef8a62'
        else: return '#b2182b'

    def get_symbols(row):
        symbols = []
        if row.get('health_children', 0) == 1: symbols.append('Hc')
        if row.get('health_adults', 0) == 1: symbols.append('Ha')
        if row.get('health_seniors', 0) == 1: symbols.append('Hs')
        if row.get('food', 0) == 1: symbols.append('F')
        if row.get('eitc', 0) == 1: symbols.append('E')
        return ' '.join(symbols) if symbols else ''

    continental = states_gdf[~states_gdf['STUSPS'].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP'])]

    for idx, row in continental.iterrows():
        pct = row['migration_pct'] if pd.notna(row['migration_pct']) else 0
        color = get_color(pct)
        continental[continental.index == idx].plot(ax=ax, color=color, edgecolor='white', linewidth=0.5)

    for idx, row in continental.iterrows():
        centroid = row.geometry.centroid
        symbols = get_symbols(row)
        if symbols:
            ax.annotate(symbols, xy=(centroid.x, centroid.y), ha='center', va='center',
                        fontsize=7, fontweight='bold', color='#333333')

    ax.set_title('Net Domestic Migration (% of Population) & Welfare Benefits for Illegal Immigrants',
                 fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')

    # Legends
    welfare_elements = [
        Line2D([0], [0], marker='$Hc$', color='w', markersize=9, markerfacecolor='#333', label='Hc = Health (Children)'),
        Line2D([0], [0], marker='$Ha$', color='w', markersize=9, markerfacecolor='#333', label='Ha = Health (Adults)'),
        Line2D([0], [0], marker='$Hs$', color='w', markersize=9, markerfacecolor='#333', label='Hs = Health (Seniors 65+)'),
        Line2D([0], [0], marker='$F$', color='w', markersize=8, markerfacecolor='#333', label='F = Food Assistance'),
        Line2D([0], [0], marker='$E$', color='w', markersize=8, markerfacecolor='#333', label='E = EITC (ITIN filers)'),
    ]
    legend1 = ax.legend(handles=welfare_elements, loc='lower left', fontsize=8, frameon=True,
                        title='Benefits for Illegal Immigrants', title_fontsize=9)
    ax.add_artist(legend1)

    migration_elements = [
        mpatches.Patch(facecolor='#d1e5f0', edgecolor='#666', label='Net Inflow (>= 0%)'),
        mpatches.Patch(facecolor='#fddbc7', edgecolor='#666', label='Small outflow (0 to -0.3%)'),
        mpatches.Patch(facecolor='#ef8a62', edgecolor='#666', label='Moderate outflow (-0.3% to -0.6%)'),
        mpatches.Patch(facecolor='#b2182b', edgecolor='#666', label='Large outflow (< -0.6%)'),
    ]
    ax.legend(handles=migration_elements, loc='lower right', fontsize=8, frameon=True,
              title='Net Domestic Migration (% of Pop)', title_fontsize=9)

    fig.text(0.5, 0.02, 'Sources: US Census Bureau Population Estimates 2023 (Migration); KFF, NILC (Health Coverage); NILC (Food Assistance); ITEP (State EITC)',
             ha='center', fontsize=10, color='#444444', style='italic')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {output_path.name}')


def create_social_graphic(df, output_path):
    """Create social media graphic with state lists."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor('#000000')
    ax.set_facecolor('#000000')
    ax.axis('off')

    # Headline
    ax.text(0.5, 0.92, 'AMERICANS ARE', fontsize=32, color='white', ha='center', fontweight='bold')
    ax.text(0.5, 0.84, 'FLEEING', fontsize=64, color='#ff4466', ha='center', fontweight='bold')
    ax.text(0.5, 0.76, 'ILLEGAL IMMIGRANT', fontsize=28, color='white', ha='center', fontweight='bold')
    ax.text(0.5, 0.71, 'WELFARE STATES', fontsize=28, color='white', ha='center', fontweight='bold')

    # Get top 10 outflow and inflow
    top_outflow = df.nsmallest(10, 'migration_pct')
    top_inflow = df.nlargest(10, 'migration_pct')

    outflow_with_benefits = (top_outflow['total_benefits'] > 0).sum()
    inflow_with_benefits = (top_inflow['total_benefits'] > 0).sum()

    # Left: Outflow
    ax.text(0.25, 0.62, 'TOP 10 OUTFLOW', fontsize=14, color='#ff4466', ha='center', fontweight='bold')
    ax.text(0.25, 0.585, f'({outflow_with_benefits} of 10 offer benefits)', fontsize=11, color='#888888', ha='center')

    for i, (_, row) in enumerate(top_outflow.iterrows()):
        y = 0.54 - i * 0.038
        has_ben = row['total_benefits'] > 0
        color = '#ff4466' if has_ben else '#555555'
        marker = '*' if has_ben else ''
        ax.text(0.25, y, f"{row['abbrev']}{marker}", fontsize=13, color=color, ha='center', fontweight='bold')

    # Right: Inflow
    ax.text(0.75, 0.62, 'TOP 10 INFLOW', fontsize=14, color='#00ff88', ha='center', fontweight='bold')
    ax.text(0.75, 0.585, f'({inflow_with_benefits} of 10 offers benefits)', fontsize=11, color='#888888', ha='center')

    for i, (_, row) in enumerate(top_inflow.iterrows()):
        y = 0.54 - i * 0.038
        has_ben = row['total_benefits'] > 0
        color = '#00ff88' if has_ben else '#555555'
        marker = '*' if has_ben else ''
        ax.text(0.75, y, f"{row['abbrev']}{marker}", fontsize=13, color=color, ha='center', fontweight='bold')

    ax.axvline(x=0.5, ymin=0.15, ymax=0.65, color='#333333', linewidth=2)

    ax.text(0.5, 0.12, '* = offers welfare benefits to illegal immigrants',
            fontsize=11, color='#888888', ha='center', style='italic')

    # Fisher's exact test
    table = [[outflow_with_benefits, 10 - outflow_with_benefits],
             [inflow_with_benefits, 10 - inflow_with_benefits]]
    _, pval = stats.fisher_exact(table)

    rect = FancyBboxPatch((0.3, 0.04), 0.4, 0.055, boxstyle="round,pad=0.02",
                           facecolor='#ff4466', edgecolor='none')
    ax.add_patch(rect)
    ax.text(0.5, 0.0675, f'p = {pval:.4f}', fontsize=18, color='white', ha='center', fontweight='bold')

    ax.text(0.5, 0.01, 'Census Bureau 2023 Migration Data', fontsize=10, color='#555555', ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor='#000000', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f'Saved {output_path.name}')


def print_statistics(df):
    """Print key statistics."""
    print("\n" + "=" * 60)
    print("KEY STATISTICS (Continental US - Excludes AK & HI)")
    print("=" * 60)

    # Correlations
    pearson_r, pearson_p = stats.pearsonr(df['total_benefits'], df['migration_pct'])
    spearman_r, spearman_p = stats.spearmanr(df['total_benefits'], df['migration_pct'])
    pb_r, pb_p = stats.pointbiserialr(df['has_any_benefit'], df['migration_pct'])

    print(f"\nCorrelations (total benefits vs migration %):")
    print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
    print(f"  Spearman rho = {spearman_r:.3f} (p = {spearman_p:.4f})")
    print(f"  Point-biserial r = {pb_r:.3f} (p = {pb_p:.4f})")

    # Inflow rates
    no_benefits = df[df['total_benefits'] == 0]
    has_benefits = df[df['total_benefits'] > 0]

    inflow_no = (no_benefits['migration_pct'] > 0).sum()
    inflow_yes = (has_benefits['migration_pct'] > 0).sum()

    print(f"\nInflow rates:")
    print(f"  States with 0 benefits: {inflow_no} of {len(no_benefits)} ({100*inflow_no/len(no_benefits):.0f}%) have net inflow")
    print(f"  States with benefits: {inflow_yes} of {len(has_benefits)} ({100*inflow_yes/len(has_benefits):.0f}%) have net inflow")

    # Top 10 analysis
    top_outflow = df.nsmallest(10, 'migration_pct')
    top_inflow = df.nlargest(10, 'migration_pct')

    outflow_with = (top_outflow['total_benefits'] > 0).sum()
    inflow_with = (top_inflow['total_benefits'] > 0).sum()

    print(f"\nTop 10 analysis:")
    print(f"  Top 10 outflow states with benefits: {outflow_with} of 10")
    print(f"  Top 10 inflow states with benefits: {inflow_with} of 10")

    # Fisher's exact
    table = [[outflow_with, 10 - outflow_with], [inflow_with, 10 - inflow_with]]
    _, fisher_p = stats.fisher_exact(table)
    print(f"  Fisher's exact test p-value: {fisher_p:.4f}")

    # List states
    print(f"\nTop 10 OUTFLOW states:")
    for _, row in top_outflow.iterrows():
        benefits = "Yes" if row['total_benefits'] > 0 else "No"
        print(f"  {row['state']} ({row['abbrev']}): {row['migration_pct']:.2f}% - Benefits: {benefits}")

    print(f"\nTop 10 INFLOW states:")
    for _, row in top_inflow.iterrows():
        benefits = "Yes" if row['total_benefits'] > 0 else "No"
        print(f"  {row['state']} ({row['abbrev']}): {row['migration_pct']:+.2f}% - Benefits: {benefits}")

    # Mean migration by group
    print(f"\nMean migration % by benefit level:")
    for n in sorted(df['total_benefits'].unique()):
        subset = df[df['total_benefits'] == n]
        print(f"  {int(n)} benefits: {subset['migration_pct'].mean():+.2f}% (n={len(subset)})")


def main():
    """Run full analysis."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("Filtering to continental US...")
    df_cont = get_continental(df)

    print("\nGenerating figures...")
    create_scatter_plot(df_cont, OUTPUT_DIR / 'scatter_benefits_migration.png')
    create_binary_scatter(df_cont, OUTPUT_DIR / 'scatter_binary_migration.png')
    create_bar_chart(df_cont, OUTPUT_DIR / 'bar_chart_migration.png')
    create_map(df, OUTPUT_DIR / 'map_migration_benefits.png')
    create_social_graphic(df_cont, OUTPUT_DIR / 'social_graphic.png')

    print_statistics(df_cont)

    print("\n" + "=" * 60)
    print("Analysis complete! Figures saved to output/")
    print("=" * 60)


if __name__ == '__main__':
    main()
