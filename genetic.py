import random
import numpy as np

# cria um indivíduo (lista de números aleatórios entre -1 e 1)
def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

# gera uma população inteira de indivíduos com tamanho size
def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

# aplica mutações aleatórias a cada gene de um indivíduo com probabilidade mutation rate
def mutate(individual, mutation_rate):
    return [
        gene + random.uniform(-0.5, 0.5) if random.random() < mutation_rate else gene
        for gene in individual
    ]

# combina dois indivíduos pais criando um novo filho tirando a média de cada gene
def crossover(parent1, parent2):
    return [(g1 + g2) / 2 for g1, g2 in zip(parent1, parent2)]

# algoritmo genético 
def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations,
                      elite_rate=0.2, mutation_rate=0.05, on_generation=None):
    population = generate_population(individual_size, population_size)
    best_individual = None
    best_fitness = float('-inf')

    elite_count = max(1, int(elite_rate * population_size))

    for gen in range(generations):
        fitnesses = [fitness_function(ind) for ind in population]

        for i, fit in enumerate(fitnesses):
            if fit > best_fitness:
                best_fitness = fit
                best_individual = population[i]

        if on_generation:
            on_generation(gen, best_fitness)  # Chama o callback com a geração atual e o melhor fitness

        if best_fitness >= target_fitness:
            print(f"Target fitness reached at generation {gen}")
            break

        elite_indices = np.argsort(fitnesses)[-elite_count:]
        new_population = [population[i] for i in elite_indices]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(new_population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_individual, best_fitness
