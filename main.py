import matplotlib.pyplot as plt

from src.load_data import DataHandler
from src.mlp import evaluate_individual

if __name__ == "__main__":
    # Preprocess data
    data_handler = DataHandler("data/melb_data.csv")

    individual = {
        "hidden_layers": (32, 24),
        "dropout_rate": 0.2,
        "learning_rate": 0.0069519279617756054,
        "weight_decay": 2.1544346900318823e-05,
    }

    test_predictions, y_test_exp, mse, mae, r2 = evaluate_individual(
        individual,
        batch_size=32,
        patience=10,
        test_ratio=0.2,
        num_epochs=50,
        improvement_threshold=1e-3,
        show_plot=True,
    )

    print(f"Test Loss (MSE): {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    plt.scatter(y_test_exp, test_predictions, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs. Predicted Values")
    plt.show()
