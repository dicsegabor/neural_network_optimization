import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from tqdm import tqdm

from src.mlp import evaluate_individual


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
                    0.2,
                    50,
                    1e-3,
                    False,
                )
                for individual in population
            ]

        for future in tqdm(
            as_completed(tasks), total=len(tasks), desc="Evaluating population"
        ):
            try:
                # Get the result for this task
                _, _, mse, _, _ = future.result()
                fitness_scores.append(mse)
            except Exception as e:
                print(f"Error evaluating an individual: {e}")
                fitness_scores.append(float("inf"))  # Assign a high fitness score

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

    print("\nBest hyperparameters found:", best_individual)
    print(f"MSE score: {best_fitness}")
    return best_individual
