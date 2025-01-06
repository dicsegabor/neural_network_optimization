import random
import torch

from sklearn.metrics import mean_squared_error

from main import preprocess_data, evaluate_model, prepare_dataloaders, train_model
from src.models.mlp import MLPRegression


from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


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
    num_generations=10,
    population_size=10,
    patience=5,
    batch_size=32,
):
    """Optimize hyperparameters using a genetic algorithm with multithreading."""
    # Define the hyperparameter search space
    search_space = {
        "hidden_layers": [(8,), (16, 8), (32, 16, 8)],
        "dropout_rate": [0.1, 0.2, 0.3],
        "learning_rate": [0.001, 0.0001, 0.00001],
        "weight_decay": [1e-4, 1e-5, 1e-6],
    }

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
        with ThreadPoolExecutor() as executor:
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
    """Select parents using tournament selection."""
    sorted_population = [
        individual for _, individual in sorted(zip(fitness_scores, population))
    ]
    return sorted_population[:num_parents]


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
        if random.random() < 0.2:  # Mutation probability
            key_to_mutate = random.choice(list(search_space.keys()))
            child[key_to_mutate] = random.choice(search_space[key_to_mutate])

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
        population_size=10,
        patience=5,
    )
    print("\nBest Hyperparameters:", best_hyperparameters)
