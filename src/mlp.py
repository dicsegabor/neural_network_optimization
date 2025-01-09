import os
import tempfile

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.load_data import DataHandler


class MLPRegression(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers=(128, 64, 32),
        activation=nn.ReLU,
        dropout_rate=0.0,
        batch_norm=False,
    ):
        super(MLPRegression, self).__init__()
        layers = []
        in_features = input_size

        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(activation())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def evaluate(self, test_data):
        """Evaluate the model on the test set."""
        x_test_tensor, y_test_tensor = test_data

        # Predict on the test set
        self.model.eval()
        with torch.no_grad():
            predictions = torch.expm1(self.model(x_test_tensor)).numpy()
            y_test_exp = torch.expm1(y_test_tensor)

        return predictions, y_test_exp

    def train_gradient_with_early_stop(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        patience,
        improvement_threshold=1e-3,
        show_plot=False,
    ):
        """Train the model with early stopping using a temporary file to save the best model."""
        best_val_loss = float("inf")
        early_stop_counter = 0

        # Create a temporary file for saving the best model
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        if show_plot:
            # Initialize graph data
            train_losses = []
            val_losses = []

            # Initialize plot
            plt.ion()
            _, ax = plt.subplots()
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
            (train_loss_line,) = ax.plot([], [], label="TL", color="blue")
            (val_loss_line,) = ax.plot([], [], label="VL", color="orange")
            ax.legend()

        # Create the tqdm progress bar object
        progress_bar = tqdm(range(num_epochs), desc="Training Progress")

        for _ in progress_bar:
            # Training phase
            self.model.train()
            train_loss = 0
            total_train_batches = len(train_loader)  # Total number of training batches
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= total_train_batches  # Normalize training loss

            # Validation phase
            self.model.eval()
            val_loss = 0
            total_val_batches = len(val_loader)  # Total number of validation batches
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    val_predictions = self.model(x_val)
                    val_loss += criterion(val_predictions, y_val).item()

            val_loss /= total_val_batches  # Normalize validation los

            if show_plot:
                # Average losses
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)

                # Update graph data
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Update plot
                train_loss_line.set_data(range(len(train_losses)), train_losses)
                val_loss_line.set_data(range(len(val_losses)), val_losses)
                ax.set_xlim(0, len(train_losses))
                ax.set_ylim(0, max(*train_losses, *val_losses) * 1.1)
                plt.draw()
                plt.pause(0.01)

            # Update the tqdm description with current losses
            progress_bar.set_postfix({"TL": train_loss, "VL": val_loss})

            # Early stopping logic
            if val_loss < best_val_loss - improvement_threshold:
                best_val_loss = val_loss
                early_stop_counter = 0

                # Save the best model to the temporary file
                torch.save(self.model.state_dict(), temp_path)
            else:
                early_stop_counter += 1

            # In casse of early stop, fill the progress bar, break cycle
            if early_stop_counter >= patience:
                progress_bar.set_postfix({"Status": "Early stop"})
                progress_bar.total = progress_bar.n
                progress_bar.update(0)
                break

        # Load the best model from the temporary file
        self.model.load_state_dict(torch.load(temp_path, weights_only=False))

        # Clean up the temporary file
        os.remove(temp_path)

        if show_plot:
            # Keep the final plot open
            plt.ioff()
            plt.show()

        return self.model


def evaluate_individual(
    individual,
    batch_size,
    patience,
    test_ratio=0.2,
    validaion_ratio=0.2,
    num_epochs=50,
    improvement_threshold=1e-3,
    show_plot=False,
):
    x_train, _, _, _ = DataHandler.split_traning_data(test_ratio)

    print(f"Parameter count: {x_train.shape[1]}")

    # Initialize the model with the individual's hyperparameters
    model = MLPRegression(
        input_size=x_train.shape[1],
        dropout_rate=individual["dropout_rate"],
        hidden_layers=individual["hidden_layers"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=individual["learning_rate"],
        weight_decay=individual["weight_decay"],
    )
    criterion = torch.nn.MSELoss()

    # Prepare data loaders
    (train_loader, val_loader, test_data) = DataHandler.prepare_dataloaders(
        batch_size, test_ratio, validaion_ratio
    )

    # Train the model
    model.train_gradient_with_early_stop(
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        patience=patience,
        show_plot=show_plot,
        improvement_threshold=improvement_threshold,
    )

    # Evaluate fitness (validation loss)
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
    return test_predictions, y_test_exp, mse, mae, r2
