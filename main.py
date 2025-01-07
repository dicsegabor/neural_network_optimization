import random
import time

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.mlp import MLPRegression
from src.load_data import DataHandler

# Initialize random
random.seed(time.time())

# Preprocess data
data_handler = DataHandler("data/melb_data.csv")
# Preprocess data and prepare data loaders
(
    x_train,
    _,
    _,
    _,
) = DataHandler.split_traning_data(0.2)

(train_loader, val_loader, test_data) = DataHandler.prepare_dataloaders(
    32, 0.2, validation_set=True
)

# Initialize the model
input_size = x_train.shape[1]
model = MLPRegression(input_size=input_size, dropout_rate=0.2, hidden_layers=(32, 24))

# Define optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0069519279617756054,
    weight_decay=2.1544346900318823e-05,
)
criterion = torch.nn.MSELoss()

# Train the model
model.train_gradient_with_early_stop(
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=100,
    patience=10,
    show_plot=True,
    improvement_threshold=1e-3,
)

# Evaluate the model on the test set
test_predictions, y_test_exp = model.evaluate(test_data)

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
