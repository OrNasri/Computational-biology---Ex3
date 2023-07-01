# Or Nasri 316582246  Niv Nahman 318012564
import math
import random
import sys
import numpy as np

OUTPUT_SIZE = 1
INPUT_SIZE = 16
HIDDEN_LAYER_SIZE = 8
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.2
TRAIN_PERCENTAGE = 0.8
CROSSOVER_RATE = 0.8
TOURNAMENT = int(POPULATION_SIZE * 0.2)

class Individual:
    def __init__(self, weights):
        self.weights = weights
        self.fitness = -1


def load_data(train_file = "nn1.txt", test_file = "nn1.txt"):
    train_set = []
    test_set = []
    data = np.loadtxt(train_file, dtype=str)
    # Create a list of tuples
    list_of_tuples = [(item[0], int(item[1])) for item in data]
    # Calculate the split index
    split_index = int(len(list_of_tuples) * 0.8)
    # Split the list_of_tuples into train and test sets
    train_set = list_of_tuples[:split_index]
    test_set = list_of_tuples[split_index:]
    if train_file != test_file :
        train_set = np.concatenate((train_set, test_set))
        data = np.loadtxt(train_file, dtype=str)
        test_set = [(item[0], int(item[1])) for item in data]
    return train_set, test_set


def write_output_to_file(file_name, best_ind):
    with open(file_name, 'w') as file:
        file.write(' '.join(str(weight) for weight in best_ind.weights))
    file.close()


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def decode(s):
    return np.array([float(c) for c in s])


def neural_network(nn_weights, string):
    total_hidden = 0
    for i in range(len(string)):
        total_hidden += int(string[i]) * nn_weights.weights[i]  # Input to hidden layer computation
    hidden_activation = sigmoid(total_hidden)

    total_output = 0

    for i in range(INPUT_SIZE * HIDDEN_LAYER_SIZE, INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE):
        total_output += hidden_activation * nn_weights.weights[i]  # Hidden layer to output computation

    output_weights = nn_weights.weights[INPUT_SIZE * HIDDEN_LAYER_SIZE:INPUT_SIZE *
                                        HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE]
    output_threshold = (max(output_weights) + min(output_weights)) / 2

    return 1 if total_output >= output_threshold else 0


def fitness_function(individual, train_set):
    fitness = 0
    for data in train_set:
        input_data, expected_output = decode(data[0]), data[1]
        predicted_output = neural_network(individual, input_data)
        if predicted_output == expected_output:
            fitness += 1
    individual.fitness = fitness / len(train_set)
    return individual.fitness


def random_weights():
    weights = [random.uniform(-1, 1) for _ in range(INPUT_SIZE*HIDDEN_LAYER_SIZE + 
                                                    HIDDEN_LAYER_SIZE*OUTPUT_SIZE)]
    return weights


def crossover(old_parent1, old_parent2):
    rand_num = random.random()
    if rand_num < CROSSOVER_RATE:
        ind = np.random.randint(0,len(old_parent2.weights))
        new_child1 = np.hstack([old_parent1.weights[:ind], old_parent2.weights[ind:]])
        new_child2 = np.hstack([old_parent2.weights[:ind], old_parent1.weights[ind:]])
        child_1_individual = Individual(new_child1)
        child_2_individual = Individual(new_child2)
        return child_1_individual, child_2_individual
    else:
        return old_parent1, old_parent2


def mutate(individual):
    rand_num = random.random()
    if rand_num < MUTATION_RATE:
        for i in range(len(individual.weights)):
            threshold = random.uniform(-0.1, 0.1)
            individual.weights[i] += threshold
    return individual


def generation_loop(best_score, best_individual, train_data, test_data, population):
    for generation in range(1, GENERATIONS + 1):
        for ind in population:
            fitness_function(ind, train_data)
        population.sort(key=lambda x: x.fitness, reverse=True)
        print(f'Generation {generation}: {population[0].fitness}')

        if population[0].fitness > best_score:
            best_score = population[0].fitness
            best_individual = population[0]

        new_population = [population[0], population[1]]
        while len(new_population) < POPULATION_SIZE:
            # Select top individuals for mating
            new_child1, new_child2 = crossover(population[np.random.randint(0, 10)], population[np.random.randint(0, 10)])
            new_child1 = mutate(new_child1)
            new_child2 = mutate(new_child2)
            new_population.append(new_child1)
            new_population.append(new_child2)
        population = new_population

        fitness_grades = [fitness_function(ind, train_data) for ind in population]
        sorted_indexes = sorted(range(POPULATION_SIZE), key=lambda k: fitness_grades, reverse=True)[:TOURNAMENT]
        best_population = [population[i] for i in sorted_indexes]
        population = best_population + random.choices(population, k=POPULATION_SIZE - TOURNAMENT)
    return best_individual, test_data

def genetic_algorithm(train_data, test_data):
    population = [Individual(random_weights()) for _ in range(POPULATION_SIZE)]
    best_score = 0
    best_individual = None
    best_individual, test_data = generation_loop(best_score, best_individual, train_data, test_data, population)

    best_test_score = fitness_function(best_individual, test_data)
    print(f'test accuracy {best_test_score}')

    return best_individual, best_test_score


def main():
    if len(sys.argv) == 2:
        train_file = sys.argv[1]
        train_set, test_set = load_data(train_file)
    if len(sys.argv) == 3:
        train_file, test_file = sys.argv[1], sys.argv[2]
        train_set, test_set = load_data(train_file, test_file)
    else:
        train_set, test_set = load_data("nn1.txt")

    best_individual, best_score = genetic_algorithm(train_set, test_set)
    write_output_to_file("wnet1.txt", best_individual)
    print(f'test accuracy: {best_score}')


if __name__ == "__main__":
    main()