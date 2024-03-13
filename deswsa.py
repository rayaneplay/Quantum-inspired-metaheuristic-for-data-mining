import numpy as np
from utils import *

class DESWSA:

    def __init__(self, X, num_elephants, p, t_max, w_min, w_max, num_clusters_max):
        self.num_elephants = num_elephants
        self.p = p
        self.X = X
        self.t_max = t_max
        self.w_min = w_min
        self.w_max = w_max
        self.num_clusters_max = num_clusters_max
        self.best_global_fitness = -np.inf
        self.best_global_solution = None
        self.best_local_fitness = np.full(num_elephants, -np.inf)
        self.best_local_solution = [None] * num_elephants

    def initialize_positions(self, num_clusters):
        positions = np.zeros((self.num_elephants, 2, self.num_clusters_max))
        for i in range(self.num_elephants):
            ids = np.random.choice(self.X.shape[0], size=self.num_clusters_max, replace=False)
            positions[i, 0, :self.num_clusters_max] = ids
            ones_ids = np.random.choice(self.num_clusters_max, size=num_clusters, replace=False)
            positions[i, 1, ones_ids] = 1
        return positions
    
    def initialize_velocities(self):
        velocities = np.zeros((1,self.num_elephants))
        for i in range(self.num_elephants):
           velocities[0, i] = np.random.randint(1, self.num_clusters_max) 
        return velocities
    
    def update_position(self, position, velocity, num_clusters, num_elephant):
        indices = np.arange(num_clusters)
        for _ in range(round(velocity[0, num_elephant])):
            id1, id2 = np.random.choice(indices, size=2, replace=False)
            position[num_elephant, 1, id1], position[num_elephant, 1, id2] = position[num_elephant, 1, id2], position[num_elephant, 1, id1]
        return position
    
    def update_velocity(self, velocity, position, local_best_position, global_best_position, p,i):
        w = self.w_max - ((self.w_max - self.w_min) / self.t_max)
        rand=np.random.uniform(1, self.num_clusters_max)
        if rand > p:
                velocity[0][i] = velocity[0][i] * w +  rand * np.linalg.norm(global_best_position-position[i][1])%self.num_clusters_max
        else:
                velocity[0][i] = velocity[0][i] * w  + rand * np.linalg.norm(local_best_position[i][1]-position[i][1])%self.num_clusters_max
        return velocity
    
    def create_clusters(self, centroids, distance_method):
        distances = {}
        for data_point in self.X:
            min_distance = float('inf')  
            closest_centroid = None
            for centroid in centroids:
                dist = distance_method(centroid, data_point)
                if dist < min_distance:
                    min_distance = dist
                    closest_centroid = tuple(centroid)  
            if closest_centroid in distances:
                distances[closest_centroid].append(data_point)
            else:
                distances[closest_centroid] = [data_point]  
        return distances
    
    def calculate_fitness (self, clusters, distance_method):
        dist =0
        for centroid in clusters.keys():
            for datapoint in clusters[centroid]:
                dist += distance_method(centroid, datapoint)
        return 1/dist
    
    def get_centroids(self, position):
        ids = np.where(position[1] == 1)[0]
        ids_centroids = position[0, ids]
        centroids = self.X[ids_centroids.astype(int), :]
        return centroids
    
    def fitness (self, position, distance_method):
        centroids = self.get_centroids(position)
        clusters = self.create_clusters(centroids, distance_method) 
        fitness_score = self.calculate_fitness(clusters, distance_method)
        return fitness_score
    
    def run(self):
        best_solutions = {}
        for num_clusters in range(2, self.num_clusters_max+1):
            for i in range(self.num_elephants):
                positions = self.initialize_positions(num_clusters)
                velocities = self.initialize_velocities()
                fitness = self.fitness(positions[i],distance_euclidean)
                self.best_local_fitness[i] = fitness
                self.best_local_solution[i] = positions[i].copy()
                if fitness > self.best_global_fitness:
                   self.best_global_fitness = fitness
                   self.best_global_solution = positions[i].copy()
            for _ in range(self.t_max):
                for i in range(self.num_elephants):
                    velocities = self.update_velocity(velocities, positions, self.best_local_solution, self.best_global_solution, self.p,i)
                    positions = self.update_position(positions, velocities, num_clusters, i)
                    fitness = self.fitness(positions[i],distance_euclidean)
                    if fitness > self.best_local_fitness[i]:
                        self.best_local_fitness[i] = fitness
                        self.best_local_solution[i] = positions[i].copy()
                    if fitness > self.best_global_fitness:
                        self.best_global_fitness = fitness
                        self.best_global_solution = positions[i].copy()
            best_solutions[num_clusters] = (self.best_global_solution, self.best_global_fitness)
        return best_solutions