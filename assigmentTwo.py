import random


class GeneticAlgorithm:
    def __init__(self,n_iter, cross_rate, mutate_rate,max_stagnant_gens=20):
        self.initalpopulation = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
        self.n_iter = n_iter
        self.population_size =  len(self.initalpopulation)
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.count = 0
        self.max_stagnant_gens = max_stagnant_gens

    def run(self):
        population = [self.generate_chromosome() for _ in range(self.population_size)]
        best, best_eval = population[0], self.fitness(population[0])
        stagnant_gens = 0

        for gen in range(self.n_iter):
            scores = [self.fitness(c) for c in population]
            for i in range(self.population_size):
                if scores[i] > best_eval:
                    best, best_eval = population[i], scores[i]
                    print(">%d, new best f(%s) = %.3f" % (i, population[i], scores[i]))
                    print()
                    stagnant_gens = 0
                else:
                    stagnant_gens += 1



            if stagnant_gens >= self.max_stagnant_gens:
                print(f"Terminating early due to {self.max_stagnant_gens} generations without improvement.")
                break
            selected = self.selection(population, scores, tournament_size=3)
            children = list()

            # Apply crossover and mutation based on the crossover and mutation rates
            while len(children) < self.population_size:
                # Select random indexes for crossover
                idx1, idx2 = random.randint(0, len(selected) - 1), random.randint(0, len(selected) - 1)
                p1, p2 = selected[idx1], selected[idx2]
            
                # Apply crossover with a probability of cross_rate
                if random.random() < self.cross_rate:
                    for c in self.crossover(p1, p2):
                        # Apply mutation with a probability of mutate_rate
                        if random.random() < self.mutate_rate:
                            self.mutate(c)
                        children.append(c)
                else:
                    # If no crossover, check if we want to mutate
                    # then mutate and add or directly add parents to the next generation
                    if random.random() < self.mutate_rate:
                        self.mutate(p1)
                    if random.random() < self.mutate_rate:
                        self.mutate(p2)
                    children.extend([p1, p2])
                
                # Ensure children list does not exceed population
                if len(children) > self.population_size:
                    children = children[:self.population_size]

            population = children
        return best, best_eval
    
    def fitness(self,chromosome):
        fitness = 0
        for task_index, person in enumerate(chromosome):
            person_index = self.initalpopulation.index(person)
            fitness += self.scores()[person_index][task_index]
        return fitness

    """A function to generate a new chromosome."""
    def generate_chromosome(self):
        chromosome = random.sample(self.initalpopulation, 10)
        return chromosome

    def selection(self, population, scores, tournament_size):
        selected = []
        tournament_size = max(1, min(tournament_size, len(population)))
        for  i in range(len(population)):
            k =  random.randint(0, len(population) - 1)
            candidates = random.sample(range(len(population)), tournament_size)
            best_candidate_index = min(candidates, key=  lambda index: scores[index])
            selected.append(population[best_candidate_index])

        return selected



    
    def crossover(self, p1, p2):
       cross_idx = random.randint(1, len(p1)-1) 
       offspring1 = p1[:cross_idx] + p2[cross_idx:]
       offspring2 = p2[:cross_idx] + p1[cross_idx:]
       return offspring1, offspring2


    def mutate(self, chromosome):
       idx1, idx2 = random.sample(range(len(chromosome)), 2)
       chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    def scores(self):
        try:
            with open('File.txt') as file:
                lines = file.readlines()
            score = []
            for line in lines[0:]:
                score_row = list(map(int, line.strip().split(",")))
                score.append(score_row)
            return score
        except FileNotFoundError:
            print("Error: The file 'File.txt' was not found.")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
        


def main():
    ga = GeneticAlgorithm(100,0.9, 0.1)
    best, best_eval = ga.run()
    print("Best individual:", best)
    print("Best fitness:", best_eval)

if __name__ == "__main__":
    main()
