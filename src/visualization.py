import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    """Plot correlation heatmap"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_model_performance(performance_df):
    """Plot bar chart for model metrics"""
    performance_df.plot(kind="bar", figsize=(8, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.show()
