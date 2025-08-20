from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model, evaluate_model, save_model
from src.visualization import plot_correlation

# Step 1: Load data
df = load_data("data/raw/your_dataset.csv")

# Step 2: Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess_data(df, target_column="target")

# Step 3: Train model
model = train_model(X_train, y_train)

# Step 4: Evaluate
metrics = evaluate_model(model, X_test, y_test)
print("Performance:", metrics)

# Step 5: Save model
save_model(model)
