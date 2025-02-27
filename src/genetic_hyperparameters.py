import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from src.mlp import evaluate_individual
from src.cli import ProgressBarManager


def generate_layer_arrangement(max_layers=5, max_neurons=32, min_neurons=8, step=4):
    """
    Generate a random layer arrangement where:
    - The number of layers is random, up to `max_layers`.
    - Neurons in each layer decrease as you progress through the layers.

    Args:
        max_layers (int): Maximum number of layers.
        max_neurons (int): Maximum neurons in the first layer.
        min_neurons (int): Minimum neurons in any layer.
        step (int): Step size for the number of neurons.

    Returns:
        list[int]: A list representing the number of neurons in each layer.
    """
    # Choose a random number of layers (1 to max_layers)
    num_layers = random.randint(1, max_layers)

    # Generate a decreasing sequence of neurons
    available_neurons = list(range(min_neurons, max_neurons + 1, step))
    layers = []
    for _ in range(num_layers):
        if not available_neurons:
            break
        neurons = random.choice(available_neurons)
        layers.append(neurons)
        # Ensure subsequent layers have fewer neurons
        available_neurons = [n for n in available_neurons if n < neurons]

    return layers


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


# Define the hyperparameter search space and mutation rules
search_space = {
    "dropout_rate": {
        "values": [0.1, 0.2, 0.3, 0.4, 0.5],
        "mutation": lambda val: max(0.0, min(0.5, val + random.choice([-0.05, 0.05]))),
    },
    "hidden_layers": {
        "value": generate_layer_arrangement,
        "mutation": mutate_hidden_layers,
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


# Mutate an individual
def mutate_individual(individual, mutation_probability=0.2):
    """Mutate an individual by making incremental changes to its parameters."""
    child = individual.copy()

    for key, properties in search_space.items():
        if random.random() < mutation_probability:  # Apply mutation probabilistically
            mutation_fn = properties["mutation"]
            child[key] = mutation_fn(child[key])

    return child


def select_parents(population, fitness_scores, num_parents):
    # Exclude individuals with infinite or NaN fitness scores
    valid_indices = [
        i
        for i, score in enumerate(fitness_scores)
        if math.isfinite(score) and not math.isnan(score)
    ]

    if not valid_indices:
        raise ValueError("No valid individuals with finite fitness scores.")

    # Filter the population and fitness scores
    valid_population = [population[i] for i in valid_indices]
    valid_fitness_scores = [fitness_scores[i] for i in valid_indices]

    # Invert fitness scores to turn it into a maximization problem
    epsilon = 1e-8  # Small constant to prevent division by zero
    inverted_fitness = [1 / (score + epsilon) for score in valid_fitness_scores]

    # Normalize the inverted fitness scores to sum to 1
    total_fitness = sum(inverted_fitness)
    probabilities = [score / total_fitness for score in inverted_fitness]

    # Ensure the total of probabilities is finite
    if not math.isfinite(total_fitness):
        raise ValueError("Total of inverted fitness scores must be finite.")

    # Select parents using the calculated probabilities
    selected_parents = random.choices(
        valid_population, weights=probabilities, k=num_parents
    )

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


def genetic_algorithm(
    num_generations=16,
    population_size=20,
    patience=10,
    batch_size=32,
    max_workers=4,
    progress_manager: ProgressBarManager = None,
):
    """Optimize hyperparameters using a genetic algorithm."""
    # Initialize the population
    population = [
        {
            "hidden_layers": search_space["hidden_layers"]["value"](),
            "dropout_rate": random.choice(search_space["dropout_rate"]["values"]),
            "learning_rate": random.choice(search_space["learning_rate"]["values"]),
            "weight_decay": random.choice(search_space["weight_decay"]["values"]),
        }
        for _ in range(population_size)
    ]

    best_individual = None
    best_fitness = float("inf")
    early_stop_counter = 0

    if progress_manager:
        progress_manager.start_generation_bar()

    for generation in range(num_generations):
        fitness_scores = []

        if progress_manager:
            progress_manager.start_population_bar(generation)

        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for individual in population:
                tasks.append(
                    executor.submit(
                        evaluate_individual,
                        individual,
                        batch_size,
                        patience,
                        0.2,
                        0.2,
                        50,
                        1e-3,
                        False,
                    )
                )

            for future in as_completed(tasks):
                try:
                    # Get the result for this task
                    _, _, mse, _, _ = future.result()
                    fitness_scores.append(mse)
                except Exception as e:
                    print(f"Error evaluating an individual: {e}")
                    fitness_scores.append(float("inf"))  # Assign a high fitness score

                if progress_manager:
                    progress_manager.update_population_bar()

        if progress_manager:
            progress_manager.close_population_bar()

        # Update the best individual
        for i, mse in enumerate(fitness_scores):
            if mse < best_fitness:
                best_fitness = mse
                best_individual = population[i]

        # Select parents (tournament selection)
        parents = select_parents(
            population, fitness_scores, num_parents=population_size // 2
        )

        # Create the next generation through crossover and mutation
        population = crossover_and_mutate(parents, population_size)

        if progress_manager:
            progress_manager.update_generation_bar()

    if progress_manager:
        progress_manager.close_generation_bar()

    print("\nBest hyperparameters found:", best_individual)
    print(f"MSE score: {best_fitness}")
    return best_individual
