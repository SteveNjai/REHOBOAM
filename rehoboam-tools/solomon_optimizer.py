import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
from scipy.optimize import minimize


# Step 1: Load and combine all CSV files
def load_data(directory):
    dfs = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, sep=',')  # Comma-delimited
            dfs.append(df)
    if not dfs:
        raise ValueError(f"No CSV files found in directory: {directory}")
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


# Define possible columns (script will use only those present in data)
possible_input_cols = ['EntryZScore', 'StopZScore', 'SL_Type', 'StopLossPercent']
possible_output_cols = ['Profit', 'Expected Payoff', 'Profit Factor', 'Recovery Factor', 'Sharpe Ratio', 'Equity DD %',
                        'Trades']

# Load data (update path)
data_dir = 'optimization files'  # Updated path to your folder
df = load_data(data_dir)

# Dynamically select available columns
input_cols = [col for col in possible_input_cols if col in df.columns]
output_cols = [col for col in possible_output_cols if col in df.columns]

if not input_cols:
    raise ValueError("No input columns found in data. Check CSV headers.")
if not output_cols:
    raise ValueError("No output columns found in data. Check CSV headers.")

print(f"Available Input Columns: {input_cols}")
print(f"Available Output Columns: {output_cols}")

# Ensure SL_Type is treated as categorical (0 or 1) if present
if 'SL_Type' in df.columns:
    df['SL_Type'] = df['SL_Type'].astype(int)

# Optional: Filter to Trades >=1 if 'Trades' is available
if 'Trades' in df.columns:
    df = df[df['Trades'] >= 1]

# Step 2: Preprocess data dynamically
# Identify numeric and categorical inputs
numeric_inputs = [col for col in input_cols if col != 'SL_Type']
categorical_inputs = ['SL_Type'] if 'SL_Type' in input_cols else []

transformers = [('num', StandardScaler(), numeric_inputs)]
if categorical_inputs:
    transformers.append(('cat', OneHotEncoder(sparse_output=False, categories=[[0, 1]]), categorical_inputs))

preprocessor = ColumnTransformer(transformers=transformers)

output_scaler = StandardScaler()
y = output_scaler.fit_transform(df[output_cols])

X = preprocessor.fit_transform(df[input_cols])

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create dataset and loader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Step 3: Define neural network model (surrogate regressor) - SOLOMON AI
class SurrogateModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


input_dim = X.shape[1]
output_dim = len(output_cols)
model = SurrogateModel(input_dim, output_dim)

# Training setup (lower LR for stability)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train the model
epochs = 300
for epoch in range(epochs):
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

print("\nPowered by SOLOMON AI")

# Step 4: Define objective function for optimization (dynamic weights)
# Base weights; use only for available outputs
base_weights = {
    'Profit': 1.0,
    'Expected Payoff': 1.0,
    'Profit Factor': 1.0,
    'Recovery Factor': 1.0,
    'Sharpe Ratio': 1.0,
    'Equity DD %': -1.0,
    'Trades': 0.5
}
weights = {col: base_weights.get(col, 0.0) for col in output_cols}  # Default 0 for missing


def post_process_predictions(pred, output_cols):
    """Enforce logical constraints: Clip/round Trades; if Trades <1, zero profit metrics."""
    pred_copy = pred.copy()
    trades_idx = output_cols.index('Trades') if 'Trades' in output_cols else -1
    if trades_idx != -1:
        pred_copy[trades_idx] = max(0, round(pred_copy[trades_idx]))  # Trades: non-negative integer
        if pred_copy[trades_idx] < 1:
            # Zero profit-related metrics if no trades
            for col in ['Profit', 'Expected Payoff', 'Profit Factor', 'Recovery Factor', 'Sharpe Ratio']:
                if col in output_cols:
                    col_idx = output_cols.index(col)
                    pred_copy[col_idx] = 0
            # Equity DD % stays (no drawdown without trades)
    return pred_copy


def objective(continuous_inputs, sl_type=None):
    # Handle SL_Type if present
    inputs = continuous_inputs.copy()
    if 'SL_Type' in input_cols:
        # Insert sl_type at position 2 (after StopZScore, before StopLossPercent)
        inputs = np.insert(inputs, 2, sl_type)

    inputs_df = pd.DataFrame([inputs], columns=input_cols)
    inputs_trans = preprocessor.transform(inputs_df)
    inputs_tensor = torch.tensor(inputs_trans, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(inputs_tensor).numpy()

    pred = output_scaler.inverse_transform(pred_scaled)[0]
    pred = post_process_predictions(pred, output_cols)  # Enforce constraints

    # Dynamic score based on available outputs
    score = sum(weights[col] * pred[output_cols.index(col)] for col in output_cols)
    return -score


# Step 5: Optimize dynamically
# Continuous inputs exclude SL_Type
continuous_input_cols = [col for col in input_cols if col != 'SL_Type']
initial_guess_continuous = df[continuous_input_cols].mean().values if continuous_input_cols else np.array([])
bounds_continuous = [(0, 10) if col in ['EntryZScore', 'StopZScore'] else (0, 5) if col == 'StopLossPercent' else (0, 1)
                     for col in continuous_input_cols]

# If SL_Type present, optimize for each value; else, single optimization
results = []
if 'SL_Type' in input_cols:
    for sl_type_val in [0, 1]:
        if len(initial_guess_continuous) > 0:
            result = minimize(lambda x: objective(x, sl_type=sl_type_val), initial_guess_continuous,
                              bounds=bounds_continuous, method='L-BFGS-B')
            best_score = -result.fun
            best_inputs_cont = result.x
            # Insert SL_Type at position 2 (after EntryZScore and StopZScore)
            best_inputs = np.insert(best_inputs_cont, 2, sl_type_val)
        else:
            # Only SL_Type
            best_inputs = np.array([sl_type_val])
            best_score = -objective([], sl_type=sl_type_val)
        results.append((best_score, best_inputs))
else:
    if len(initial_guess_continuous) > 0:
        result = minimize(lambda x: objective(x), initial_guess_continuous, bounds=bounds_continuous, method='L-BFGS-B')
        best_score = -result.fun
        best_inputs = result.x
    else:
        best_inputs = np.array([])
        best_score = 0
    results.append((best_score, best_inputs))

# Select the best result
best_score, best_inputs = max(results, key=lambda x: x[0])

# Output best inputs (map indices to columns)
print("\nBest Input Parameters:")
for i, col in enumerate(input_cols):
    if i < len(best_inputs):
        val = best_inputs[i]
        if col == 'SL_Type':
            sl_type_reverse = {1: 'percentage based stop loss', 0: 'Z-Score based stop loss'}
            print(f"{col}: {sl_type_reverse[int(val)]}")
        else:
            print(f"{col}: {val:.2f}")
    else:
        print(f"{col}: N/A")

# Predict outputs for best inputs (with constraints)
best_inputs_df = pd.DataFrame([best_inputs], columns=input_cols) if len(best_inputs) > 0 else pd.DataFrame()
if not best_inputs_df.empty:
    best_inputs_trans = preprocessor.transform(best_inputs_df)
    best_inputs_tensor = torch.tensor(best_inputs_trans, dtype=torch.float32)
    with torch.no_grad():
        best_pred_scaled = model(best_inputs_tensor).numpy()
    best_pred_raw = output_scaler.inverse_transform(best_pred_scaled)[0]
    best_pred = post_process_predictions(best_pred_raw, output_cols)
else:
    best_pred = np.zeros(len(output_cols))

print("\nPredicted Best Outputs:")
for col, val in zip(output_cols, best_pred):
    print(f"{col}: {val:.5f}")