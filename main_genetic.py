import random
import torch

from sklearn.metrics import mean_squared_error
import numpy as np

from main import preprocess_data, evaluate_model, prepare_dataloaders
from src.models.mlp import MLPRegression


from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import itertools


def generate_hidden_layers(
    min_layers=1, max_layers=4, min_neurons=8, max_neurons=128, step=8
):
    """Generate all possible hidden layer combinations."""
    for num_layers in range(min_layers, max_layers + 1):
        for combination in itertools.product(
            range(min_neurons, max_neurons + 1, step), repeat=num_layers
        ):
            yield combination


def generate_dropout_rates(start=0.1, stop=0.5, step=0.1):
    """Generate dropout rates."""
    return (round(x, 2) for x in np.arange(start, stop + step, step))


def generate_learning_rates(start_exp=-5, end_exp=-1, num_samples=10):
    """Generate learning rates logarithmically."""
    return np.logspace(start_exp, end_exp, num=num_samples)


def generate_weight_decays(start_exp=-6, end_exp=-2, num_samples=5):
    """Generate weight decay values logarithmically."""
    return np.logspace(start_exp, end_exp, num=num_samples)


def evaluate_individual(
    individual, x_train, x_test, y_train, y_test, batch_size, patience
):
    """Train and evaluate an individual, returning the validation MSE."""
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
    train_loader, val_loader, test_data = prepare_dataloaders(
        x_train, x_test, y_train, y_test, batch_size=batch_size
    )

    # Train the model with tqdm for epochs
    num_epochs = 50
    with tqdm(total=num_epochs, desc="Training epochs", leave=False) as pbar:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
            pbar.update(1)  # Update progress bar

    # Evaluate fitness (validation loss)
    test_predictions, y_test_exp = evaluate_model(model, test_data)
    val_mse = mean_squared_error(y_test_exp, test_predictions)
    return val_mse


def genetic_algorithm(
    x_train,
    x_test,
    y_train,
    y_test,
    num_generations=16,
    population_size=20,
    patience=5,
    batch_size=32,
    max_workers=4,
):
    """Optimize hyperparameters using a genetic algorithm with multithreading."""
    # Define the hyperparameter search space
    search_space = {
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "hidden_layers": [
            (8,),
            (16,),
            (24,),
            (32,),
            (8, 8),
            (8, 16),
            (8, 24),
            (8, 32),
            (16, 8),
            (16, 16),
            (16, 24),
            (16, 32),
            (24, 8),
            (24, 16),
            (24, 24),
            (24, 32),
            (32, 8),
            (32, 16),
            (32, 24),
            (32, 32),
        ],
        "learning_rate": [0.001, 0.0005, 0.0001, 5e-05, 1e-05],
        "weight_decay": [0.01, 0.001, 0.0001, 1e-05, 1e-06],
    }

    print("Searchspaces created")

    # Initialize the population
    population = [
        {
            "hidden_layers": random.choice(search_space["hidden_layers"]),
            "dropout_rate": random.choice(search_space["dropout_rate"]),
            "learning_rate": random.choice(search_space["learning_rate"]),
            "weight_decay": random.choice(search_space["weight_decay"]),
        }
        for _ in range(population_size)
    ]

    best_individual = None
    best_fitness = float("inf")

    for generation in tqdm(range(num_generations), desc="Generations"):
        # Evaluate the fitness of each individual in parallel
        fitness_scores = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(
                    evaluate_individual,
                    individual,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    batch_size,
                    patience,
                )
                for individual in population
            ]

            # Use tqdm to track individual progress
            for future, individual in tqdm(
                zip(tasks, population), total=len(tasks), desc="Evaluating population"
            ):
                try:
                    val_mse = future.result()
                    fitness_scores.append(val_mse)
                except Exception as e:
                    print(f"Error evaluating individual {individual}: {e}")
                    fitness_scores.append(float("inf"))  # Assign a high fitness score

        # Update the best individual
        for i, val_mse in enumerate(fitness_scores):
            if val_mse < best_fitness:
                best_fitness = val_mse
                best_individual = population[i]

        # Select parents (tournament selection)
        parents = select_parents(
            population, fitness_scores, num_parents=population_size // 2
        )

        # Create the next generation through crossover and mutation
        population = crossover_and_mutate(parents, search_space, population_size)

    print("\nBest hyperparameters found:", best_individual)
    return best_individual


def select_parents(population, fitness_scores, num_parents):
    """Select parents using roulette wheel selection."""
    # Invert fitness scores (lower is better)
    max_fitness = max(fitness_scores)
    adjusted_scores = [
        max_fitness - score + 1e-8 for score in fitness_scores
    ]  # Avoid zero probability

    # Normalize to probabilities
    total_score = sum(adjusted_scores)
    probabilities = [score / total_score for score in adjusted_scores]

    # Select parents probabilistically
    selected_parents = random.choices(population, weights=probabilities, k=num_parents)

    # Print the best individual of the generation
    best_index = fitness_scores.index(min(fitness_scores))
    print("\nBest hyperparameters found in generation:", population[best_index])

    return selected_parents


def mutate_individual(individual, search_space, mutation_probability=0.2):
    """Mutate an individual by making incremental changes to its parameters."""
    child = individual.copy()

    for key in child.keys():
        if random.random() < mutation_probability:  # Apply mutation probabilistically
            if key == "hidden_layers":
                # Increment/decrement neurons or modify layers
                layers = list(child[key])
                if layers and random.random() < 0.5:
                    # Adjust neurons in an existing layer
                    layer_idx = random.randint(0, len(layers) - 1)
                    layers[layer_idx] = max(
                        8, layers[layer_idx] + random.choice([-8, 8])
                    )
                else:
                    # Add/remove a layer
                    if random.random() < 0.5 and len(layers) < 3:  # Add a layer
                        layers.append(random.choice(range(8, 64, 8)))
                    elif len(layers) > 1:  # Remove a layer
                        layers.pop(random.randint(0, len(layers) - 1))
                child[key] = tuple(layers)

            elif key == "dropout_rate":
                # Adjust dropout rate slightly
                child[key] = max(
                    0.0, min(0.5, child[key] + random.choice([-0.05, 0.05]))
                )

            elif key == "learning_rate":
                # Scale learning rate slightly
                child[key] *= random.choice([0.8, 1.2])
                child[key] = max(1e-6, min(0.1, child[key]))

            elif key == "weight_decay":
                # Scale weight decay slightly
                child[key] *= random.choice([0.75, 1.5])
                child[key] = max(1e-6, min(0.1, child[key]))

    return child


def crossover_and_mutate(parents, search_space, population_size):
    """Perform crossover and mutation to generate the next generation."""
    next_population = parents[:]

    while len(next_population) < population_size:
        # Crossover
        parent1, parent2 = random.sample(parents, 2)
        child = {
            key: random.choice([parent1[key], parent2[key]]) for key in parent1.keys()
        }

        # Mutation
        child = mutate_individual(child, search_space, 0.2)

        next_population.append(child)

    return next_population


# ------------------------------ Main Workflow ------------------------------

if __name__ == "__main__":
    # Preprocess data
    x_train, x_test, y_train, y_test, preprocessor = preprocess_data()

    # Run genetic algorithm
    best_hyperparameters = genetic_algorithm(
        x_train,
        x_test,
        y_train,
        y_test,
        num_generations=10,
        population_size=8,
        patience=5,
        max_workers=4,
    )
    print("\nBest Hyperparameters:", best_hyperparameters)
