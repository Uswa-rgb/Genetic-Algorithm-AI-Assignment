import numpy as np
import cv2
import random
import copy
import time

img = cv2.imread('imageB.bmp', 0)
dimensions = (0, 0)
dimensions = img.shape

image = np.array(img)
_1DImage = []
_1DImage = image.flatten()
TARGET_Lenght = len(_1DImage)

POPULATION_SIZE = 100
GENES = []
for i in range(0, 255):
    GENES.append(i)

TARGET = copy.deepcopy(_1DImage)


class Individual(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        difference = np.subtract(self.chromosome, TARGET)
        absolute = np.abs(difference)
        return np.sum(absolute)

    @classmethod
    def create_Individual(cls):
        individual_lenght = len(TARGET)
        gene = []
        for i in range(individual_lenght):
            gene.append(random.choice(GENES))
        return gene

    # @classmethod
    # def mutated_genes(cls):
    #     global GENES
    #     gene = []
    #     gene = random.choice(GENES)

    def mate(self, p2):
        child_chromosome = []
        pivot = random.randrange(0, TARGET_Lenght)

        prob = np.random.random()
        if prob < 0.50:
            child_chromosome.extend(self.chromosome[:pivot])
            child_chromosome.extend(p2.chromosome[pivot:])
        else:
            child_chromosome.extend(p2.chromosome[:pivot])
            child_chromosome.extend(self.chromosome[pivot:])

        # for i in range(len(child_chromosome)):
        #     prob = random.random()
        #     if prob <= 0.10:
        #         mutate = random.choice(GENES)
        #         child_chromosome[i] = mutate

        pivot = np.random.randint(0, TARGET_Lenght)
        newval = random.choice(GENES)
        child_chromosome[pivot] = newval
        return child_chromosome


def main():
    START_TIME = time.time()
    GENERATION = 1
    initial_cost = 0
    # global POPULATION
    POPULATION = []
    found = False
    for i in range(0, POPULATION_SIZE):
        chromosome = Individual.create_Individual()
        POPULATION.append(Individual(chromosome))

    while not found:
        POPULATION.sort(key=lambda x: x.fitness)
        if GENERATION == 1:
            initial_cost = POPULATION[POPULATION_SIZE - 1].fitness

        if POPULATION[0].fitness <= 0:
            found = True
            break

        else:
            newGeneration = []

            # case-1    selection: 10%      crossove: 90%       mutation: 90%
            # s = int((10 * POPULATION_SIZE) / 100)
            # newGeneration.extend(POPULATION[:s])
            # n = int((90 * POPULATION_SIZE) / 100)
            # for _ in range(n):
            #     parent1 = random.choice(POPULATION[s:])
            #     parent2 = random.choice(POPULATION[s:])
            #     child_CHROMOSOME = parent1.mate(parent2)
            #     child = Individual(child_CHROMOSOME)
            #     newGeneration.append(child)

            # # case 2    selection: 0%      crossove: 100%       mutation: 100%
            # for _ in range(POPULATION_SIZE):
            #     parent1 = random.choice(POPULATION[:POPULATION_SIZE])
            #     parent2 = random.choice(POPULATION[:POPULATION_SIZE])
            #     child_CHROMOSOME = parent1.mate(parent2)
            #     child = Individual(child_CHROMOSOME)
            #     newGeneration.append(child)

            # # case-3    selection: 10%      crossove: 90%       mutation: 90%
            s = int((10 * POPULATION_SIZE) / 100)
            newGeneration.extend(POPULATION[:s])
            for _ in range(s):
                parent1 = random.choice(POPULATION[:s])
                parent2 = random.choice(POPULATION[:s])
                child_CHROMOSOME = parent1.mate(parent2)
                child = Individual(child_CHROMOSOME)
                newGeneration.append(child)

            POPULATION = newGeneration
            CURRENT_TIME = time.time()
            accuracy = (
                (initial_cost - POPULATION[0].fitness) / initial_cost) * 100
            print(
                "Generation: {} \tFitness: {} \tAccuray: {:.2f}% \tElapsed Time: {:.2f}"
                .format(GENERATION, POPULATION[0].fitness, accuracy,
                        (CURRENT_TIME - START_TIME)))

            GENERATION += 1
            if GENERATION % 100 == 0:
                img = [[]]
                img = np.reshape(POPULATION[0].chromosome, (-1, dimensions[1]))
                cv2.imwrite('result.bmp', img)

            if GENERATION == 1000 or GENERATION == 10000 or GENERATION == 100000 or GENERATION == 1000000:
                img = [[]]
                img = np.reshape(POPULATION[0].chromosome, (-1, dimensions[1]))
                cv2.imwrite('gen_{}.bmp'.format(GENERATION), img)

            if (POPULATION[0].fitness <= 6000):
                break

    print("Generation: {} \tFitness: {}".format(GENERATION,
                                                POPULATION[0].fitness))


main()
