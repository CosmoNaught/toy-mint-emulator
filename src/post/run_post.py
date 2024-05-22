import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import cycle

def assign_colors_to_nets(neural_net_types):
    """
    Assign distinct colors to each neural network type.
    """
    colors = plt.cm.get_cmap('tab10', len(neural_net_types))
    return {net: colors(i) for i, net in enumerate(neural_net_types)}

def plot_performance_metrics(data, neural_net_types, metrics, epochs_options):
    """
    Plot average performance metrics over log2 scaled training sizes for specified neural network types
    and epoch configurations.
    """
    colors = assign_colors_to_nets(neural_net_types)
    epoch_subset = [epochs_options[0], epochs_options[len(epochs_options)//2], epochs_options[-1]]
    linestyle = cycle(['-', '--', ':'])  # Different line styles for min, median, max epochs

    # Setup subplot dimensions
    num_plots = len(metrics)
    cols = 2 if num_plots > 1 else 1
    rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for ax, metric in zip(axes, metrics):
        for net_type in neural_net_types:
            for epoch, style in zip(epoch_subset, linestyle):
                subset = data[(data['neural_net_type'] == net_type) & (data['nn_epochs'] == epoch)]
                avg_metric = subset.groupby('targeted_training_size')[metric].mean().reset_index()
                avg_metric['targeted_training_size'] = np.log2(avg_metric['targeted_training_size'])

                ax.plot(avg_metric['targeted_training_size'], avg_metric[metric], label=f'{net_type} Epochs={epoch}',
                        color=colors[net_type], linestyle=style)
        
        ax.set_title(f'Average {metric.upper()} by Training Size')
        ax.set_xlabel('Training Data Size (log2)')
        ax.set_ylabel(metric.upper())
        ax.legend(title='Neural Net Type & Epochs')

    plt.tight_layout()
    plt.savefig('performance_metrics_plot.png', dpi=300)
    plt.show()

def train_models(X_train, y_train):
    """
    Train both Random Forest and XGBoost models.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Initialize and train XGBoost Regressor
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train_scaled, y_train)

    return rf, xgb, scaler

def get_feature_importance(model, feature_names):
    """
    Extract and sort feature importances from a trained model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    return sorted_feature_names, sorted_importances

def plot_feature_importances(rf_importances, xgb_importances, feature_names, title, ax):
    """
    Plot feature importances for both models on given axes.
    """
    indices = np.arange(len(feature_names))
    ax.bar(indices - 0.15, rf_importances, 0.3, label='Random Forest', color='#FF9999')
    ax.bar(indices + 0.15, xgb_importances, 0.3, label='XGBoost', color='#9999FF')
    
    ax.set_xticks(indices)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=12)
    ax.set_title(f'{title} Feature Importances', fontsize=16)
    ax.set_ylabel('Feature Importance', fontsize=14)
    ax.legend(fontsize=12)

def analyze_feature_importance(data, features, target, neural_net_types):
    """
    Prepare data, train models, and plot feature importances for specified features, target,
    and neural network types. Adjusts titles and filenames based on the target.
    """
    fig, axes = plt.subplots(len(neural_net_types), 1, figsize=(10, 5 * len(neural_net_types)), constrained_layout=True)
    if len(neural_net_types) == 1:
        axes = [axes]  # Ensure axes is iterable even for one subplot

    for ax, net_type in zip(axes, neural_net_types):
        subset = data[data['neural_net_type'] == net_type]
        X = subset[features]
        y = subset[target]
        
        # Split data
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        rf, xgb, _ = train_models(X_train, y_train)

        # Get feature importances
        rf_features, rf_importances = get_feature_importance(rf, X.columns)
        xgb_features, xgb_importances = get_feature_importance(xgb, X.columns)

        # Plot feature importances
        plot_feature_importances(rf_importances, xgb_importances, rf_features, f'{target} ({net_type})', ax)

    plt.savefig(f'feature_importances_{target}.png', dpi=300)
    plt.show()

def plot_heatmap(data, training_size):
    """
    Plot heatmap for given training data size.
    """
    # Filter data for the specific training size
    subset = data[data['targeted_training_size'] == training_size]

    # Pivot data to create a matrix of epoch size by batch size, values are average total training times
    pivot_table = subset.pivot_table(values='total_training_time', 
                                     index='nn_batch_size', 
                                     columns='nn_epochs', 
                                     aggfunc='mean')

    # Taking log2 of index and columns to scale down the epoch size and batch size
    pivot_table.index = np.log2(pivot_table.index).astype(int)
    pivot_table.columns = np.log2(pivot_table.columns).astype(int)
    
    # Heatmap plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")
    plt.title(f'Heatmap of Total Training Time for Training Size: {training_size}')
    plt.xlabel('Log2 Epoch Size')
    plt.ylabel('Log2 Batch Size')
    plt.show()

def main():
    # Load dataset
    data = pd.read_csv('/home/cosmo/Documents/PhD/Projects/toy-mint-emulator/archive/benchmark/20240503-160338-8796c163/experiment_results_final.csv')
    training_sizes = data['targeted_training_size'].unique()
    for size in training_sizes:
        plot_heatmap(data, size)

    # Define neural network types to analyze
    neural_net_types = ['FFNN', 'GRU', 'LSTM', 'BiRNN']  # Add or remove types as needed

    metrics = ['rmse', 'mse', 'mae', 'r2']
    epochs_options = [2**i for i in range(3, 8)]  # Min, median, max epochs
    plot_performance_metrics(data, neural_net_types, metrics, epochs_options)
    
    # Define features and target for analysis
    features = ['nn_epochs', 'nn_batch_size', 'targeted_training_size']
    target = 'total_training_time'
    analyze_feature_importance(data, features, target, neural_net_types)

    # Define features and target for analysis
    features = ['nn_epochs', 'targeted_training_size']
    target = 'rmse'
    analyze_feature_importance(data, features, target, neural_net_types)
    target = 'mse'
    analyze_feature_importance(data, features, target, neural_net_types)
    target = 'r2'
    analyze_feature_importance(data, features, target, neural_net_types)