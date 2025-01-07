from src.load_data import DataHandler

from src.genetic_hyperparameters import genetic_algorithm

if __name__ == "__main__":
    # Preprocess data
    data_handler = DataHandler("data/melb_data.csv")

    # Run genetic algorithm
    best_hyperparameters = genetic_algorithm(
        num_generations=2,
        population_size=8,
        patience=5,
        max_workers=4,
    )
    print("\nBest Hyperparameters:", best_hyperparameters)
