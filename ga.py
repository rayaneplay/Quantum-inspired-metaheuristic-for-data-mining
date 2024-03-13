import numpy as np
from utils import *

class GA:

    def __init__(self, X, num_clusters, pop_size=100, num_generations=100, mutation_rate = 0.8, num_parents=10, crossover_rate=0.8):
        self.num_clusters = num_clusters
        self.X=X
        self.num_generations=num_generations
        self.mutation_rate = mutation_rate
        self.num_parents= num_parents
        self.crossover_rate = crossover_rate
        self.pop_size = pop_size
        

    def calculate_fitness(self, X, centroids, num_clusters, distance_method):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        total_distance = 0
        for cluster in range(num_clusters):
            cluster_points = X[labels == cluster]
            if len(cluster_points) > 0:  
                cluster_distance = distance_method(centroids[cluster], cluster_points)
                total_distance += cluster_distance
        return 1 / total_distance

    def select_mating_pool(self, pop, fitness, num_parents):
        parents = np.empty((num_parents, pop.shape[1], pop.shape[2]))
        for parent_num in range(num_parents):
            max_fitness_idx = np.random.choice(range(len(fitness)), p=fitness/np.sum(fitness))
            parents[parent_num, :, :] = pop[max_fitness_idx, :, :]
        return parents

    def crossover(self, parents, offspring_size):
        offspring = np.empty(offspring_size)
        for k in range(offspring_size[0]):
            if np.random.rand() < self.crossover_rate:
                parent1_idx = k % parents.shape[0]
                parent2_idx = (k + 1) % parents.shape[0]
                crossover_point = np.random.randint(1, parents.shape[1])
                offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
                offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
            else:
                offspring[k] = parents[k % parents.shape[0]]
        return offspring

    def mutation(self, offspring_crossover):
        for idx in range(offspring_crossover.shape[0]):
            if np.random.uniform(0, 1) < self.mutation_rate:
                mutation_point = np.random.randint(0, offspring_crossover.shape[1])
                offspring_crossover[idx, mutation_point] += np.random.uniform(-0.5, 0.5)
        return offspring_crossover
    
    def run(self):
        num_features = self.X.shape[1]
        initial_centroids = np.random.rand(self.num_clusters, num_features)
        pop = np.array([initial_centroids for _ in range(self.pop_size)])  
        best_fitness = []
        for generation in range(self.num_generations):
            fitness = np.array([self.calculate_fitness(self.X, centroids, self.num_clusters, distance_manhattan) for centroids in pop])
            best_fitness.append(np.max(fitness))
            parents = self.select_mating_pool(pop, fitness, self.num_parents)
            offspring_crossover = self.crossover(parents, offspring_size=(pop.shape[0] - parents.shape[0], pop.shape[1], pop.shape[2]))
            offspring_mutation = self.mutation(offspring_crossover)
            pop[:parents.shape[0], :, :] = parents
            pop[parents.shape[0]:, :, :] = offspring_mutation

        best_match_idx = np.argmax(fitness)
        best_centroids = pop[best_match_idx]
        print("Best centroids:\n", best_centroids)
        print("Best fitness:", best_fitness[-1])