import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch


def preprocess_data(file_path="data/melb_data.csv"):
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop rows with missing target values
    df = df.dropna(subset=["Price"])

    # Define features and target, dropping unnecessary columns
    x = df.drop(columns=["Price", "Address", "Date", "SellerG", "CouncilArea"])
    y = df["Price"]

    # Log transformation and clipping
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
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
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
    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train.toarray(), dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(
        -1, 1
    )  # Fixed: Use `.values`
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(
        -1, 1
    )  # Fixed: Use `.values`

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

    return train_loader, val_loader, test_data, x_test_tensor
