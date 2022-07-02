import copy

import DEOSA.config as config
from DEOSA.data.dataloader import data_loader
from DEOSA.utilities._utilities import avg_concentration, sign_func, get_transfer_function, compute_accuracy
from DEOSA.simulated_annealing import simulated_annealing
from DEOSA.fitness_functions import get_fitness_function

import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def DEOSA(  data,
            type_data,
            population_size=50,
            transfer_shape="u",
            a1=2,
            a2=1,
            GP=0.5,
            pool_size=4,
            max_iter=50,
            allow_SA=True,
            seed=0):
    """
    :param data: data should be passed in a particular format
    :param type_data: knapsack/uci
    :param population_size: number of population members
    :param transfer_shape: s/v/u
    :param a1: EO parameter
    :param a2: EO parameter
    :param GP: EO parameter
    :param pool_size: EO parameter
    :param max_iter: number of generations
    :param allow_SA: allowing simulated annealing
    :param seed: random seed used for random number generation
    :return:
    """

    # initialization
    np.random.seed(seed)
    if type_data == "uci":
        dimension = data["train_x"].shape[1]
    else:
        dimension = data["count"]
    population = np.random.randint(0, 2, size=(population_size, dimension))
    eq_pool = np.zeros((pool_size + 1, dimension))
    eq_fit = np.array([0.0]*pool_size)
    conv_plot = []
    transfer_func = get_transfer_function(transfer_shape)
    compute_fitness = get_fitness_function(type_data)
    best_solution = {}

    # iterations start
    for iter in range(max_iter):
        fitness = compute_fitness(population, data)
        sorted_idx = np.argsort(-fitness).squeeze()
        population = population[sorted_idx, :]
        fitness = fitness[0, sorted_idx]

        # replacements in the pool
        for i in range(pool_size):
            for j in range(pool_size):
                if fitness[i] > eq_fit[j]:
                    eq_fit[j] = copy.deepcopy(fitness[i])
                    eq_pool[j, :] = copy.deepcopy(population[i, :])
                    break

        print(f"Best fitness till iteration {iter}: {eq_fit[0]}")
        conv_plot.append(copy.deepcopy(eq_fit[0]))
        best_particle = copy.deepcopy(eq_pool[0, :])
        best_fitness = copy.deepcopy(eq_fit[0])

        Cave = avg_concentration(eq_pool, pool_size, dimension)
        eq_pool[pool_size] = Cave.copy()

        t = (1 - (iter / max_iter)) ** (a2 * iter / max_iter)

        for i in range(population_size):
            # randomly choose one candidate from the equilibrium pool
            inx = np.random.randint(0, pool_size)
            Ceq = np.array(eq_pool[inx])

            lambda_vec = np.zeros(np.shape(Ceq))
            r_vec = np.zeros(np.shape(Ceq))
            for j in range(dimension):
                lambda_vec[j], r_vec[j] = np.random.random(2)

            F_vec = np.zeros(np.shape(Ceq))
            for j in range(dimension):
                x = -1 * lambda_vec[j] * t
                x = np.exp(x) - 1
                x = a1 * sign_func(r_vec[j] - 0.5) * x

            r1, r2 = np.random.random(2)

            if r2 < GP:
                GCP = 0
            else:
                GCP = 0.5 * r1

            G0 = np.zeros(np.shape(Ceq))
            G = np.zeros(np.shape(Ceq))

            for j in range(dimension):
                G0[j] = GCP * (Ceq[j] - lambda_vec[j] * population[i][j])
                G[j] = G0[j] * F_vec[j]

            # use transfer function to map continuous->binary
            for j in range(dimension):
                temp = Ceq[j] + (population[i][j] - Ceq[j]) * F_vec[j] + G[j] * (1 - F_vec[j]) / lambda_vec[j]
                temp = transfer_func(temp)
                if temp > np.random.random():
                    population[i][j] = 1 - population[i][j]
                else:
                    population[i][j] = population[i][j]


        if (allow_SA):
            # performing simulated annealing
            simulated_annealing(population=population,
                                fitness=fitness,
                                compute_fitness=compute_fitness,
                                data=data,
                                seed=seed)

    if type_data == "uci":
        best_accuracy = compute_accuracy(train_X=data["train_x"],
                                     train_Y=data["train_y"],
                                     test_X=data["test_x"],
                                     test_Y=data["test_y"],
                                     particle=best_particle)

        best_solution["features"] = best_particle
        best_solution["accuracy"] = best_accuracy
        best_solution["fitness"] = best_fitness

        print(f"Best Fitness: {best_fitness}, Best Accuracy: {best_accuracy}, Number of features: {np.sum(best_particle)}")

    else:
        best_solution["features"] = best_particle
        best_solution["fitness"] = best_fitness

        print(f"Best weight: {best_fitness}, Number of items: {np.sum(best_particle)}")

    best_solution["conv_plot"] = conv_plot
    # pickle.dump(best_solution, open(f"{config.BASE_PATH}/storage/best_solution_{data['name']}.pickle", "wb"))
    return best_solution

def main():
    # test uci data
    type_data = "uci"
    dataset = "Ionosphere"
    omega = 0.9

    # reading dataset
    print('\n===============================================================================')
    print("Dataset:", dataset)
    df = data_loader(type_data=type_data,
                     data_name=dataset)
    (a, b) = np.shape(df)
    data = df.values[:, 0:b - 1]
    label = df.values[:, b - 1]
    dimension = np.shape(data)[1]  # particle dimension
    print("dimension:", dimension)

    # loading_dataset
    train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.2, random_state=0)
    np.random.seed(0)
    data_dict = {
        "name": dataset,
        "train_x": train_X,
        "train_y": train_Y,
        "test_x": test_X,
        "test_y": test_Y,
        "omega": omega
    }

    DEOSA(data_dict,
          type_data=type_data,
          allow_SA=True)

    # test the knapsack fitness
    type_data = "knapsack"
    dataset = "ks_24c"
    data_dict = data_loader(type_data=type_data,
                            data_name=dataset)
    data_dict["name"] = dataset

    DEOSA(data_dict,
          type_data=type_data,
          allow_SA=True)


if __name__ == "__main__":
    main()