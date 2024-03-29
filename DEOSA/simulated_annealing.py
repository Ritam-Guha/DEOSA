import numpy as np

from DEOSA.utilities._utilities import find_neighbor


def simulated_annealing(population,
                        fitness,
                        compute_fitness,
                        data,
                        seed=0):
    """
    :param population: current population of solutions
    :param fitness: corresponding fitnesses
    :param compute_fitness: function to compute fitness
    :param data: data in a particular format
    :param seed: seed used for random number generation
    :return: updates the population and corresponding fitness
    """

    # simulated annealing
    np.random.seed(seed)
    [particle_count, dimension] = population.shape
    T0 = dimension

    for particle_no in range(particle_count):
        T = 2 * dimension
        current_particle = population[particle_no][np.newaxis, :].copy()
        current_fitness = compute_fitness(current_particle, data, seed=seed)
        best_particle = current_particle.copy()
        best_fitness = current_fitness.copy()

        while T > T0:
            new_particle = find_neighbor(current_particle)
            new_fitness = np.float64(compute_fitness(new_particle, data, seed=seed))

            if new_fitness > best_fitness:
                current_particle = new_particle.copy()
                current_fitness = new_fitness.copy()
                best_particle = current_particle.copy()
                best_fitness = current_fitness.copy()

            else:
                prob = np.exp((best_fitness - current_fitness) / T)
                if (np.random.random() <= prob):
                    current_particle = new_particle.copy()
                current_fitness = new_fitness

            T = int(T * 0.7)

        population[particle_no, :] = best_particle.copy()
        fitness[particle_no] = best_fitness.copy()

