from src.load_data import DataHandler

from src.genetic_hyperparameters import genetic_algorithm
from src.cli import ProgressBarManager

if __name__ == "__main__":
    # Preprocess data
    data_handler = DataHandler("data/melb_data.csv")

    num_generations = 10
    population_size = 6

    progress_manager = ProgressBarManager(num_generations, population_size)

    # Run genetic algorithm
    best_hyperparameters = genetic_algorithm(
        num_generations=num_generations,
        population_size=population_size,
        patience=10,
        max_workers=8,
        progress_manager=progress_manager,
    )
    print("\nBest Hyperparameters:", best_hyperparameters)
