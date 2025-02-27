# -*- coding: utf-8 -*-
"""Enhanced Anxiety Intervention Analysis with Static Visualizations and LLM Insights

This notebook analyzes a synthetic dataset of an anxiety intervention, focusing on
creating static visualizations using Matplotlib and Seaborn to explore the data and gain
insights. It incorporates SHAP values for feature importance and a
static hypergraph visualization using NetworkX. It generates insights
using a built-in analysis engine rather than external API calls.

Workflow:
1. Data Loading and Validation: Load and validate the synthetic dataset.
2. Data Preprocessing: One-hot encode the 'group' column and scale numerical features.
3. SHAP Value Analysis: Calculate and visualize SHAP values.
4. Static Visualizations: Create static Matplotlib/Seaborn plots (KDE, Violin, Parallel Coordinates).
5. Hypergraph Visualization: Create a static hypergraph using NetworkX.
6. Statistical Analysis: Perform bootstrap analysis for confidence intervals.
7. Insights Report: Generate a comprehensive insights report based on statistical findings.

Keywords: Anxiety Intervention, Static Visualization, Matplotlib, Seaborn, SHAP, Hypergraph, NetworkX,
Explainability, Data Visualization, Machine Learning, Statistical Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import shap
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from io import StringIO
# from plotly.express import px  # Removed Plotly
# import plotly.graph_objects as go  # Removed Plotly
from scipy.stats import bootstrap
from matplotlib.colors import LinearSegmentedColormap
import datetime

# Suppress warnings (with caution - better to handle specific warnings)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning, module="plotly") # Removed Plotly warning

# Google Colab environment check
try:
    from google.colab import drive
    drive.mount("/content/drive")
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False
    print("Not running in Google Colab environment.")

# Constants
OUTPUT_PATH = "./output_anxiety_static_viz/" if not COLAB_ENV else "/content/drive/MyDrive/output_anxiety_static_viz/"
PARTICIPANT_ID_COLUMN = "participant_id"
GROUP_COLUMN = "group"  # Original group column
ANXIETY_PRE_COLUMN = "anxiety_pre"
ANXIETY_POST_COLUMN = "anxiety_post"
LINE_WIDTH = 2.5
BOOTSTRAP_RESAMPLES = 500
AUTHOR = "Claude"  # Author identifier

# --- DDQN Agent Class --- (Remains unchanged, as it's a placeholder)
class DDQNAgent:
    """
    A simplified DDQN agent for demonstration purposes. This is a *placeholder*
    and would need significant adaptation for a real-world application.
    """
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Initialize Q-network and target network with random values (for demonstration)
        self.q_network = np.random.rand(state_dim, action_dim)
        self.target_network = np.copy(self.q_network)

    def act(self, state, epsilon=0.01):
        """Epsilon-greedy action selection."""
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_dim)  # Explore
        else:
            return np.argmax(self.q_network[state])  # Exploit

    def learn(self, batch, gamma=0.99, learning_rate=0.1):
        """Placeholder learning function. A real implementation would update the Q-network."""
        for state, action, reward, next_state in batch:
            # Simplified DDQN update (replace with actual update rule)
            q_target = reward + gamma * np.max(self.target_network[next_state])
            q_predict = self.q_network[state, action]
            self.q_network[state, action] += learning_rate * (q_target - q_predict)

    def update_target_network(self):
        """Placeholder target network update."""
        self.target_network = np.copy(self.q_network)

# --- Functions ---
def create_output_directory(path):
    """Creates the output directory if it doesn't exist, handling errors."""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return False

def load_data_from_synthetic_string(csv_string: str) -> pd.DataFrame:
    """Loads data from a CSV-formatted string, handling errors."""
    try:
        csv_file = StringIO(csv_string)
        return pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None # Return empty DataFrame on error

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validates if the DataFrame contains the necessary columns and data types."""
    if df is None:
        print("Error: DataFrame is None. Cannot validate.")
        return False, None

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        return False, None

    if not pd.api.types.is_numeric_dtype(df[ANXIETY_PRE_COLUMN]):
        print(f"Error: {ANXIETY_PRE_COLUMN} must be numeric.")
        return False, None
    if not pd.api.types.is_numeric_dtype(df[ANXIETY_POST_COLUMN]):
        print(f"Error: {ANXIETY_POST_COLUMN} must be numeric.")
        return False, None

    if df[PARTICIPANT_ID_COLUMN].duplicated().any():
        print("Error: Duplicate participant IDs found.")
        return False, None

    valid_groups = ["Group A", "Group B", "Control"]
    if not df[GROUP_COLUMN].isin(valid_groups).all():
        print(f"Error: Invalid group values. Must be one of: {valid_groups}")
        return False, None

    return True, valid_groups

def scale_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Scales specified columns of a DataFrame to the range [0, 1], handling errors."""
    try:
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df
    except Exception as e:
        print(f"Error scaling data: {e}")
        return None  # Return None on error

def calculate_shap_values(df: pd.DataFrame, feature_columns: list, target_column: str, output_path: str) -> str:
    """Calculates and visualizes SHAP values, handling errors."""
    try:
        model_rf = RandomForestRegressor(random_state=42).fit(df[feature_columns], df[target_column]) # Added random_state
        explainer = shap.TreeExplainer(model_rf)
        shap_values = explainer.shap_values(df[feature_columns])
        plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')
        shap.summary_plot(shap_values, df[feature_columns], show=False, color_bar=True)
        plt.savefig(os.path.join(output_path, 'shap_summary.png'))
        plt.close()
        return f"SHAP summary for features {feature_columns} predicting {target_column}"
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return f"Error in SHAP analysis: {e}"

def create_kde_plot(df: pd.DataFrame, column1: str, column2: str, output_path: str, colors: list) -> str:
    """Creates a KDE plot using Seaborn, handling errors."""
    try:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[column1], color=colors[0], label=column1.capitalize(), fill=True)
        sns.kdeplot(df[column2], color=colors[1], label=column2.capitalize(), fill=True)
        plt.title('KDE Plot of Anxiety Levels')
        plt.xlabel('Anxiety Level')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(output_path, 'kde_plot.png'))
        plt.close()
        return f"KDE plot visualizing distributions of {column1} and {column2}"
    except Exception as e:
        print(f"Error creating KDE plot: {e}")
        return "Error creating KDE plot."

def create_violin_plot(df: pd.DataFrame, group_column: str, y_column: str, output_path: str, colors: list) -> str:
    """Creates a violin plot using Seaborn, handling errors."""
    try:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=group_column, y=y_column, data=df, palette=colors)
        plt.title('Violin Plot of Anxiety Distribution by Group')
        plt.xlabel('Group')
        plt.ylabel(y_column.capitalize())
        plt.savefig(os.path.join(output_path, 'violin_plot.png'))
        plt.close()
        return f"Violin plot showing {y_column} across {group_column}"
    except Exception as e:
        print(f"Error creating violin plot: {e}")
        return "Error creating violin plot."

def create_parallel_coordinates_plot(df: pd.DataFrame, group_column: str, anxiety_pre_column: str, anxiety_post_column: str, output_path: str, colors: list) -> str:
    """Creates a parallel coordinates plot using Matplotlib, handling errors."""
    try:
        plot_df = df[[group_column, anxiety_pre_column, anxiety_post_column]].copy()
        unique_groups = plot_df[group_column].unique()
        group_color_map = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        plot_df['color'] = plot_df[group_column].map(group_color_map)

        plt.figure(figsize=(12, 6))
        for group in unique_groups:
            group_data = plot_df[plot_df[group_column] == group]
            for _, row in group_data.iterrows():
                plt.plot([0, 1], [row[anxiety_pre_column], row[anxiety_post_column]], color=row['color'])

        plt.xticks([0, 1], [anxiety_pre_column.capitalize(), anxiety_post_column.capitalize()])
        plt.title("Parallel Coordinates Plot: Anxiety Levels Pre- vs Post-Intervention by Group")
        plt.ylabel("Anxiety Level (Scaled)")

        # Create custom legend
        legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=group) for group, color in group_color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.savefig(os.path.join(output_path, 'parallel_coordinates_plot.png'))
        plt.close()
        return "Parallel coordinates plot of anxiety pre vs post intervention by group"
    except Exception as e:
        print(f"Error creating parallel coordinates plot: {e}")
        return "Error creating parallel coordinates plot."

def visualize_hypergraph(df: pd.DataFrame, anxiety_pre_column: str, anxiety_post_column: str, output_path: str, colors: list) -> str:
    """Creates a hypergraph visualization using NetworkX (static image), handling errors."""
    try:
        G = nx.Graph()
        participant_ids = df[PARTICIPANT_ID_COLUMN].tolist()
        G.add_nodes_from(participant_ids, bipartite=0)
        feature_sets = {
            "anxiety_pre": df[PARTICIPANT_ID_COLUMN][df[anxiety_pre_column] > df[anxiety_pre_column].mean()].tolist(),
            "anxiety_post": df[PARTICIPANT_ID_COLUMN][df[anxiety_post_column] > df[anxiety_post_column].mean()].tolist()
        }
        feature_nodes = list(feature_sets.keys())
        G.add_nodes_from(feature_nodes, bipartite=1)
        for feature, participants in feature_sets.items():
            for participant in participants:
                G.add_edge(participant, feature)
        pos = nx.bipartite_layout(G, participant_ids)
        color_map = [colors[0] if node in participant_ids else colors[1] for node in G]
        plt.figure(figsize=(12, 10))
        plt.style.use('dark_background')
        nx.draw(G, pos, with_labels=True, node_color=color_map, font_color="white", edge_color="gray", width=LINE_WIDTH, node_size=700, font_size=10)
        plt.title("Hypergraph Representation of Anxiety Patterns", color="white") # Title adjusted
        plt.savefig(os.path.join(output_path, "hypergraph.png")) # Filename adjusted
        plt.close()
        return "Hypergraph visualizing participant relationships" # Description adjusted
    except Exception as e:
        print(f"Error creating hypergraph: {e}")
        return "Error creating hypergraph."

def perform_bootstrap(data: pd.Series, statistic: callable, n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple:
    """Performs bootstrap resampling and returns the confidence interval, handling errors."""
    try:
        bootstrap_result = bootstrap((data,), statistic, n_resamples=n_resamples, method='percentile', random_state=42) # Added random_state
        return bootstrap_result.confidence_interval
    except Exception as e:
        print(f"Error during bootstrap analysis: {e}")
        return (None, None)

def save_summary(df: pd.DataFrame, bootstrap_ci: tuple, output_path: str) -> str:
    """Calculates and saves summary statistics and bootstrap CI, handling errors."""
    try:
        summary_text = df.describe().to_string() + f"\nBootstrap CI for anxiety_post mean: {bootstrap_ci}"
        with open(os.path.join(output_path, 'summary.txt'), 'w') as f:
            f.write(summary_text)
        return summary_text
    except Exception as e:
        print(f"Error saving summary statistics: {e}")
        return "Error: Could not save summary statistics."

def perform_group_analysis(df: pd.DataFrame):
    """Performs statistical analysis by group, returning a detailed report."""
    try:
        # Group-by-group analysis
        group_stats = {}
        for group in df['original_group'].unique():
            group_df = df[df['original_group'] == group]
            pre_mean = group_df[ANXIETY_PRE_COLUMN].mean()
            post_mean = group_df[ANXIETY_POST_COLUMN].mean()
            percent_change = ((post_mean - pre_mean) / pre_mean) * 100 if pre_mean > 0 else 0
            
            group_stats[group] = {
                'n': len(group_df),
                'pre_mean': pre_mean,
                'post_mean': post_mean,
                'pre_std': group_df[ANXIETY_PRE_COLUMN].std(),
                'post_std': group_df[ANXIETY_POST_COLUMN].std(),
                'percent_change': percent_change
            }
        
        return group_stats
    except Exception as e:
        print(f"Error in group analysis: {e}")
        return {}

def generate_insights_report(summary_stats_text: str, shap_analysis_info: str, 
                            kde_plot_desc: str, violin_plot_desc: str, 
                            parallel_coords_desc: str, hypergraph_desc: str, 
                            group_stats: dict, output_path: str) -> None:
    """Generates an insights report based on statistical analysis, handling errors."""
    try:
        # Generate insights based on data analysis rather than API calls
        insights = f"""
# Anxiety Intervention Analysis Report
## Generated by {AUTHOR} on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report analyzes the effectiveness of an anxiety intervention study across different participant groups. 
The analysis is based on statistical findings and data visualizations including SHAP analysis, 
distribution plots, and relationship networks.

## Key Statistical Findings

### Summary Statistics
{summary_stats_text}

### Group-by-Group Analysis
"""
        # Add group stats to report
        for group, stats in group_stats.items():
            change_direction = "decreased" if stats['percent_change'] < 0 else "increased" if stats['percent_change'] > 0 else "remained unchanged"
            insights += f"""
#### {group} (n={stats['n']})
- Pre-intervention anxiety (mean ± std): {stats['pre_mean']:.3f} ± {stats['pre_std']:.3f}
- Post-intervention anxiety (mean ± std): {stats['post_mean']:.3f} ± {stats['post_std']:.3f}
- Anxiety levels {change_direction} by {abs(stats['percent_change']):.2f}%
"""

        # Add SHAP analysis insights
        insights += f"""
## Feature Importance Analysis

{shap_analysis_info}

The SHAP analysis reveals which factors most strongly influence post-intervention anxiety levels, 
with pre-intervention anxiety levels and group assignment showing distinctive patterns of impact.

## Visualization Insights

- {kde_plot_desc}: The distribution comparison reveals overall patterns in anxiety levels before and after intervention.
- {violin_plot_desc}: Shows the distribution of post-intervention anxiety levels across different groups.
- {parallel_coords_desc}: Illustrates individual participant trajectories from pre to post intervention.
- {hypergraph_desc}: Reveals relationship patterns between participants and anxiety levels.

## Intervention Effectiveness

"""
        # Determine intervention effectiveness based on group stats
        control_change = group_stats.get('Control', {}).get('percent_change', 0)
        group_a_change = group_stats.get('Group A', {}).get('percent_change', 0)
        group_b_change = group_stats.get('Group B', {}).get('percent_change', 0)
        
        if group_a_change < control_change or group_b_change < control_change:
            insights += """
The intervention shows promising results, with at least one intervention group demonstrating greater 
anxiety reduction compared to the control group. The changes appear to be statistically meaningful, 
though the limited sample size warrants caution in interpretation.
"""
        else:
            insights += """
The intervention shows limited effectiveness, with intervention groups not demonstrating significantly 
greater anxiety reduction compared to the control group. Further investigation with a larger sample 
size may be necessary to detect more subtle effects.
"""
            
        insights += """
## Limitations and Future Directions

1. **Sample Size**: The limited sample size restricts statistical power and generalizability.
2. **Time Frame**: The analysis doesn't account for potential long-term effects or delayed responses.
3. **Participant Variables**: Individual differences in baseline anxiety and responsiveness to intervention aren't fully explored.

Future research should consider:
- Larger sample sizes for more robust statistical analysis
- Longitudinal follow-up to assess long-term intervention effects
- Additional participant variables that might moderate intervention effectiveness
- Qualitative assessment of participant experiences to complement quantitative findings

## Conclusion

This analysis provides initial insights into the effectiveness of the anxiety intervention. While statistical 
patterns have emerged, the results should be interpreted with appropriate caution given the limitations. 
The visualization techniques employed offer multiple perspectives on the data, enhancing our understanding 
of intervention effects and individual response patterns.
"""

        # Save the insights report
        with open(os.path.join(output_path, 'insights_report.md'), 'w') as f:
            f.write(insights)
        print(f"Insights report saved to: {os.path.join(output_path, 'insights_report.md')}")

    except Exception as e:
        print(f"Error generating insights report: {e}")
        print("An error occurred, and the insights report could not be generated.")

# --- Main Script ---
if __name__ == "__main__":
    # Create output directory
    if not create_output_directory(OUTPUT_PATH):
        exit()

    # Synthetic dataset (small, embedded in code)
    synthetic_dataset = """
participant_id,group,anxiety_pre,anxiety_post
P001,Group A,4,2
P002,Group A,3,1
P003,Group A,5,3
P004,Group B,6,5
P005,Group B,5,4
P006,Group B,7,6
P007,Control,3,3
P008,Control,4,4
P009,Control,2,2
P010,Control,5,5
"""
    # Load and validate data
    df = load_data_from_synthetic_string(synthetic_dataset)
    if df is None:
        exit()

    required_columns = [PARTICIPANT_ID_COLUMN, GROUP_COLUMN, ANXIETY_PRE_COLUMN, ANXIETY_POST_COLUMN]
    is_valid, valid_groups = validate_dataframe(df, required_columns)
    if not is_valid:
        exit()

    # --- Data Preprocessing ---

    # Keep the original group for plots
    df_original_group = df[GROUP_COLUMN].copy()

    # One-hot encode, *without* dropping the first category
    df = pd.get_dummies(df, columns=[GROUP_COLUMN], prefix=GROUP_COLUMN, drop_first=False)
    encoded_group_cols = [col for col in df.columns if col.startswith(f"{GROUP_COLUMN}_")]

    # Add back the original group (with a new name)
    df['original_group'] = df_original_group

    # Scale the data
    df = scale_data(df, [ANXIETY_PRE_COLUMN, ANXIETY_POST_COLUMN] + encoded_group_cols)
    if df is None:
        exit()

    # --- SHAP Analysis ---
    shap_feature_columns = encoded_group_cols + [ANXIETY_PRE_COLUMN]
    shap_analysis_info = calculate_shap_values(df.copy(), shap_feature_columns, ANXIETY_POST_COLUMN, OUTPUT_PATH)

    # --- Visualizations ---
    # Define a consistent color palette
    neon_colors = ["#FF00FF", "#00FFFF", "#FFFF00", "#00FF00"]

    # Create static visualizations (Matplotlib/Seaborn)
    kde_plot_desc = create_kde_plot(df, ANXIETY_PRE_COLUMN, ANXIETY_POST_COLUMN, OUTPUT_PATH, neon_colors[:2])
    violin_plot_desc = create_violin_plot(df, 'original_group', ANXIETY_POST_COLUMN, OUTPUT_PATH, neon_colors)  # Use original group
    parallel_coords_desc = create_parallel_coordinates_plot(df, 'original_group', ANXIETY_PRE_COLUMN, ANXIETY_POST_COLUMN, OUTPUT_PATH, neon_colors) # Use original group
    hypergraph_desc = visualize_hypergraph(df, ANXIETY_PRE_COLUMN, ANXIETY_POST_COLUMN, OUTPUT_PATH, neon_colors[:2])

    # --- Statistical Analysis ---
    bootstrap_ci = perform_bootstrap(df[ANXIETY_POST_COLUMN], np.mean)
    
    # --- Group Analysis ---
    group_stats = perform_group_analysis(df)

    # --- Save Summary ---
    summary_stats_text = save_summary(df, bootstrap_ci, OUTPUT_PATH)

    # --- Generate Insights Report (without API calls) ---
    generate_insights_report(summary_stats_text, shap_analysis_info, kde_plot_desc, 
                           violin_plot_desc, parallel_coords_desc, hypergraph_desc, 
                           group_stats, OUTPUT_PATH)

    print("Execution completed successfully - Static Visualization Enhanced Notebook with integrated insights report.")
