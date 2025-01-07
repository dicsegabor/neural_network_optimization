from tqdm import tqdm
from threading import Lock


class ProgressBarManager:
    def __init__(self, num_generations, population_size):
        self.num_generations = num_generations
        self.population_size = population_size
        self.generation_bar = None
        self.population_bar = None
        self.individual_lock = Lock()

    def start_generation_bar(self):
        """Start the progress bar for generations."""
        self.generation_bar = tqdm(total=self.num_generations, desc="Generations")

    def start_population_bar(self, generation):
        """Start the progress bar for the population."""
        self.population_bar = tqdm(
            total=self.population_size,
            desc=f"Population (Gen {generation + 1})",
            leave=False,
        )

    def update_population_bar(self):
        """Safely update the population-level progress bar."""
        with self.individual_lock:
            if self.population_bar:
                self.population_bar.update(1)

    def update_generation_bar(self):
        """Update the generation-level progress bar."""
        if self.generation_bar:
            self.generation_bar.update(1)

    def close_population_bar(self):
        """Close the population-level progress bar."""
        if self.population_bar:
            self.population_bar.close()
            self.population_bar = None

    def close_generation_bar(self):
        """Close the generation-level progress bar."""
        if self.generation_bar:
            self.generation_bar.close()
            self.generation_bar = None
