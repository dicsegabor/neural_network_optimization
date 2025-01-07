import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLPRegression


# ------------------------------ Data Preprocessing ------------------------------


def preprocess_data(file_path="data/melb_data.csv"):
    """Load and preprocess the dataset."""
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop rows with missing target values
    df = df.dropna(subset=["Price"])

    # Define features and target
    x = df.drop(columns=["Price", "Address", "Date", "SellerG", "CouncilArea"])
    y = df["Price"]

    # Log transform and clip target variable
    y = np.log1p(y)
    y = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))

    # Identify numerical and categorical columns
    numerical_features = [
        "Rooms",
        "Distance",
        "Bedroom2",
        "Bathroom",
        "Car",
        "Landsize",
        "BuildingArea",
        "YearBuilt",
        "Lattitude",
        "Longtitude",
        "Propertycount",
    ]
    categorical_features = ["Suburb", "Type", "Method", "Postcode", "Regionname"]

    # Check for missing columns
    missing_columns = [
        col for col in numerical_features + categorical_features if col not in x.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Fill missing values for numerical features
    for col in numerical_features:
        x[col] = x[col].fillna(x[col].median())

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Transform features
    x_transformed = preprocessor.fit_transform(x)

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_transformed, y, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test, preprocessor


def prepare_dataloaders(x_train, x_test, y_train, y_test, batch_size=32):
    """Prepare PyTorch dataloaders for training, validation, and testing."""
    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train.toarray(), dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # Split into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    test_data = (x_test_tensor, y_test_tensor)

    return train_loader, val_loader, test_data


# ------------------------------ Model Training ------------------------------


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    patience,
    save_path=None,  # Default to None
):
    """Train the model with early stopping."""
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_predictions = model(x_val)
                val_loss += criterion(val_predictions, y_val).item()

        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        # Early stopping
        improvement_threshold = 1e-3  # Threshold for significant improvement
        if val_loss < best_val_loss - improvement_threshold:
            best_val_loss = val_loss
            early_stop_counter = 0

            # Save the best model only if save_path is provided
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model (only if save_path is provided)
    if save_path is not None:
        model.load_state_dict(torch.load(save_path))
    return model


def evaluate_model(model, test_data):
    """Evaluate the model on the test set."""
    x_test_tensor, y_test_tensor = test_data

    # Predict on the test set
    model.eval()
    with torch.no_grad():
        predictions = torch.expm1(model(x_test_tensor)).numpy()
        y_test_exp = torch.expm1(y_test_tensor)

    return predictions, y_test_exp


# ------------------------------ Main Workflow ------------------------------

if __name__ == "__main__":
    # Preprocess data and prepare data loaders
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data()
    train_loader, val_loader, test_data = prepare_dataloaders(
        x_train, x_test, y_train, y_test
    )

    # Initialize the model
    input_size = x_train.shape[1]
    model = MLPRegression(
        input_size=input_size, dropout_rate=0.2, hidden_layers=(32, 24)
    )

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0069519279617756054,
        weight_decay=2.1544346900318823e-05,
    )
    criterion = torch.nn.MSELoss()

    # Train the model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=100,
        patience=10,
        save_path="best_model.pth",
    )

    # Evaluate the model on the test set
    test_predictions, y_test_exp = evaluate_model(trained_model, test_data)

    # Calculate metrics and visualize
    mse = mean_squared_error(y_test_exp, test_predictions)
    mae = mean_absolute_error(y_test_exp, test_predictions)
    r2 = r2_score(
        y_test_exp.numpy() if isinstance(y_test_exp, torch.Tensor) else y_test_exp,
        test_predictions.numpy()
        if isinstance(test_predictions, torch.Tensor)
        else test_predictions,
    )

    print(f"Test Loss (MSE): {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    plt.scatter(y_test_exp, test_predictions, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs. Predicted Values")
    plt.show()
