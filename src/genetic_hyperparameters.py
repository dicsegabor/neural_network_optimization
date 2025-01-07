import random
import torch
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from src.load_data import DataHandler
from src.mlp import MLPRegression

# Define the hyperparameter search space and mutation rules
search_space = {
    "dropout_rate": {
        "values": [0.1, 0.2, 0.3, 0.4, 0.5],
        "mutation": lambda val: max(0.0, min(0.5, val + random.choice([-0.05, 0.05]))),
    },
    "hidden_layers": {
        "values": [
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
        "mutation": lambda layers: mutate_hidden_layers(layers),
    },
    "learning_rate": {
        "values": [0.001, 0.0005, 0.0001, 5e-05, 1e-05, 5e-06, 1e-06],
        "mutation": lambda val: max(1e-6, min(0.1, val * random.choice([0.8, 1.2]))),
    },
    "weight_decay": {
        "values": [0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07],
        "mutation": lambda val: max(1e-6, min(0.1, val * random.choice([0.75, 1.5]))),
    },
}


# Custom mutation function for "hidden_layers"
def mutate_hidden_layers(layers):
    layers = list(layers)
    if layers and random.random() < 0.5:
        # Adjust neurons in an existing layer
        layer_idx = random.randint(0, len(layers) - 1)
        layers[layer_idx] = max(8, layers[layer_idx] + random.choice([-8, 8]))
    else:
        # Add/remove a layer
        if random.random() < 0.5 and len(layers) < 3:  # Add a layer
            layers.append(random.choice(range(8, 64, 8)))
        elif len(layers) > 1:  # Remove a layer
            layers.pop(random.randint(0, len(layers) - 1))
    return tuple(layers)


# Mutate an individual
def mutate_individual(individual, mutation_probability=0.2):
    """Mutate an individual by making incremental changes to its parameters."""
    child = individual.copy()

    for key, properties in search_space.items():
        if random.random() < mutation_probability:  # Apply mutation probabilistically
            mutation_fn = properties["mutation"]
            child[key] = mutation_fn(child[key])

    return child


def genetic_algorithm(
    num_generations=16,
    population_size=20,
    patience=5,
    batch_size=32,
    max_workers=4,
):
    """Optimize hyperparameters using a genetic algorithm with multithreading."""
    # Initialize the population
    population = [
        {
            "hidden_layers": random.choice(search_space["hidden_layers"]["values"]),
            "dropout_rate": random.choice(search_space["dropout_rate"]["values"]),
            "learning_rate": random.choice(search_space["learning_rate"]["values"]),
            "weight_decay": random.choice(search_space["weight_decay"]["values"]),
        }
        for _ in range(population_size)
    ]

    best_individual = None
    best_fitness = float("inf")

    for _ in tqdm(range(num_generations), desc="Generations"):
        # Evaluate the fitness of each individual in parallel
        fitness_scores = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(
                    evaluate_individual,
                    individual,
                    batch_size,
                    patience,
                    0.2,
                    50,
                    1e-3,
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
        population = crossover_and_mutate(parents, population_size)

    print("\nBest hyperparameters found:", best_individual)
    print(f"MSE score: {best_fitness}")
    return best_individual


def evaluate_individual(
    individual,
    batch_size,
    patience,
    test_ratio=0.2,
    num_epochs=50,
    improvement_threshold=1e-3,
):
    x_train, _, _, _ = DataHandler.split_traning_data(test_ratio)

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
        batch_size, test_ratio, validation_set=True
    )

    # Train the model
    model.train_gradient_with_early_stop(
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        patience=patience,
        show_plot=False,
        improvement_threshold=improvement_threshold,
    )

    # Evaluate fitness (validation loss)
    test_predictions, y_test_exp = model.evaluate(test_data)
    val_mse = mean_squared_error(y_test_exp, test_predictions)
    return val_mse


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


def crossover_and_mutate(parents, population_size):
    """Perform crossover and mutation to generate the next generation."""
    next_population = parents[:]

    while len(next_population) < population_size:
        # Crossover
        parent1, parent2 = random.sample(parents, 2)
        child = {
            key: random.choice([parent1[key], parent2[key]]) for key in parent1.keys()
        }

        # Mutation
        child = mutate_individual(child, 0.5)

        next_population.append(child)

    return next_population
