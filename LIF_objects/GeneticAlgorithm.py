import numpy as np
import multiprocessing
import random
import time
from tqdm import tqdm


class GeneticAlgorithm:
    """Implements the Genetic Algorithm logic."""
    def __init__(self, population_size, chromosome_length, fitness_func, fitness_func_args, # Pass fixed args
                 mutation_rate=0.05, mutation_strength=0.01, crossover_rate=0.7,
                 elitism_count=2, tournament_size=5,
                 weight_min=-0.02, weight_max=0.02):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_func = fitness_func # Function handle
        self.fitness_func_args = fitness_func_args # Tuple of fixed arguments for fitness func
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.population = self._initialize_population()
        self.fitness_scores = np.full(population_size, -np.inf)

    def _initialize_population(self):
        # Initialize weights with small random values
        population = []
        for _ in range(self.population_size):
             # Consider initializing closer to zero, or using a distribution
             # chromosome = np.random.uniform(self.weight_min, self.weight_max, self.chromosome_length)
             chromosome = np.random.normal(0, (self.weight_max - self.weight_min)/4 , self.chromosome_length)
             chromosome = np.clip(chromosome, self.weight_min, self.weight_max)
             population.append(chromosome)
        return population

    def evaluate_population(self, n_cores, show_progress=True):
         tasks = [(self.population[i],) + self.fitness_func_args for i in range(self.population_size)]
         start_eval_time = time.time()
         pool = None
         try:
             actual_processes = min(n_cores, self.population_size)
             # Use 'spawn' context for potentially better cross-platform compatibility
             # mp_context = multiprocessing.get_context('spawn')
             # pool = mp_context.Pool(processes=actual_processes)
             pool = multiprocessing.Pool(processes=actual_processes)

             results_iterator = pool.starmap(self.fitness_func, tasks)
             if show_progress:
                 results_iterator = tqdm(results_iterator, total=self.population_size, desc="Evaluating Pop", ncols=80, leave=False)

             new_fitness_scores = list(results_iterator)

             if any(score is None for score in new_fitness_scores):
                  print("\nWarning: Received None fitness score(s). Check fitness function.")
                  new_fitness_scores = [score if score is not None else -np.inf for score in new_fitness_scores]

             self.fitness_scores = np.array(new_fitness_scores)

         except Exception as e:
             print(f"\nFATAL Error during parallel fitness evaluation: {e}")
             # Consider how to handle this - stop? set all fitness to -inf?
             self.fitness_scores = np.full(self.population_size, -np.inf)
             # traceback.print_exc() # Print traceback for debugging
             # raise e # Optional: re-raise to stop execution
         finally:
             if pool: pool.close(); pool.join()


    def _tournament_selection(self):
        best_idx = -1; best_fitness = -np.inf
        k = min(self.population_size, self.tournament_size)
        if k <= 0: return self.population[np.random.choice(len(self.population))] if self.population else None
        competitor_indices = np.random.choice(self.population_size, k, replace=False)
        for idx in competitor_indices:
            if idx < len(self.fitness_scores) and self.fitness_scores[idx] > best_fitness:
                best_fitness = self.fitness_scores[idx]; best_idx = idx
        # Handle cases where all selected competitors have -inf fitness
        if best_idx == -1:
             best_idx = np.random.choice(competitor_indices) if competitor_indices.size > 0 else np.random.choice(len(self.population))
        # Ensure index is valid before returning
        if best_idx >= len(self.population):
             best_idx = np.random.choice(len(self.population)) # Fallback to random choice

        return self.population[best_idx]

    def _crossover(self, parent1, parent2):
        # Uniform crossover might explore the space better for weights
        child1, child2 = parent1.copy(), parent2.copy()
        if random.random() < self.crossover_rate:
            for i in range(self.chromosome_length):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
            return child1, child2
        else:
            return child1, child2 # Return copies even if no crossover

    def _mutate(self, chromosome):
        mutated_chromosome = chromosome.copy()
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                 # Additive Gaussian mutation
                 mutation_val = np.random.normal(0, self.mutation_strength)
                 mutated_chromosome[i] += mutation_val
                 # Ensure weights stay within bounds
                 mutated_chromosome[i] = np.clip(mutated_chromosome[i], self.weight_min, self.weight_max)
        return mutated_chromosome

    def run_generation(self):
         new_population = []
         actual_elitism_count = min(self.elitism_count, self.population_size)

         # Elitism: Copy best individuals directly
         if actual_elitism_count > 0 and len(self.fitness_scores) == self.population_size and np.any(np.isfinite(self.fitness_scores)):
              try:
                   # Use nanargsort if NaNs are possible, otherwise argsort
                   valid_scores = np.where(np.isneginf(self.fitness_scores), np.nan, self.fitness_scores)
                   if np.any(np.isfinite(valid_scores)):
                        elite_indices = np.argsort(self.fitness_scores)[-actual_elitism_count:] # Indices of highest fitness
                        for idx in elite_indices:
                            if idx < len(self.population):
                                 new_population.append(self.population[idx].copy())
                   else: print("Warning: No finite scores for elitism.")
              except IndexError: print("Warning: Error getting elite indices. Skipping elitism.")

         # Generate remaining individuals through selection, crossover, mutation
         while len(new_population) < self.population_size:
             parent1 = self._tournament_selection(); parent2 = self._tournament_selection()
             if parent1 is None or parent2 is None:
                  print("Warning: Parent selection failed. Breaking generation loop."); break
             child1, child2 = self._crossover(parent1, parent2)
             child1 = self._mutate(child1); child2 = self._mutate(child2)
             if len(new_population) < self.population_size: new_population.append(child1)
             if len(new_population) < self.population_size: new_population.append(child2)

         # Handle population size discrepancies (e.g., if odd number needed)
         if len(new_population) > self.population_size:
              new_population = new_population[:self.population_size]
         elif len(new_population) < self.population_size:
              print(f"Warning: New population size ({len(new_population)}) < target ({self.population_size}). This might indicate issues.")
              # Optionally fill remaining spots? (e.g., with random individuals or copies)
              while len(new_population) < self.population_size:
                   new_population.append(self._initialize_population()[0]) # Add a random new individual

         self.population = new_population
         self.fitness_scores = np.full(self.population_size, -np.inf) # Reset fitness for next gen
