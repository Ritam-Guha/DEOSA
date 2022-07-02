from DEOSA.utilities._utilities import compute_accuracy
from DEOSA.data.dataloader import data_loader

import copy
import numpy as np
from sklearn.model_selection import train_test_split

def fitness_fs(particles,
              data):
    """
    :param particles: population under consideration
    :param data: data to deal with
    :return: fitness for feature selection
    """

    # fitness computation for feature selection
    if particles.ndim == 1:
        particles = copy.deepcopy(particles)[np.newaxis, :]

    [num_particles, dimension] = particles.shape
    values = np.zeros((1, num_particles))

    for i, particle in enumerate(particles):
        current_particle = particle[np.newaxis, :]
        set_cnt = int(np.sum(current_particle))

        if set_cnt == 0:
            # if there is no feature, give the min fitness
            val = np.float64(0.0)

        else:
            set_cnt = set_cnt / dimension
            acc = compute_accuracy(data["train_x"], data["train_y"], data["test_x"], data["test_y"], particle)
            val = data["omega"] * acc + (1 - data["omega"]) * (1 - set_cnt)

        values[0, i] = np.float64(val)

    return values


def fitness_knapsack(particles,
                    data):

    """
    :param particles: population under consideration
    :param data: data to deal with
    :return: fitness for the knapsack problem
    """

    # fitness computation for knapsack
    if particles.ndim == 1:
        particles = copy.deepcopy(particles)[np.newaxis, :]

    [num_particles, dimension] = particles.shape
    values = np.zeros((1, num_particles))

    for i, particle in enumerate(particles):
        current_particle = particle[np.newaxis, :]
        set_cnt = int(np.sum(current_particle))

        if set_cnt == 0:
            # if there is no weight, give the min fitness
            val = np.float64(0.0)

        else:
            pos = np.flatnonzero(particle)
            val = (np.sum(data["weights"][pos]) <= data["max_weight"]) * np.sum(data["values"][pos])

        values[0, i] = np.float64(val)

    return values


def get_fitness_function(type_data):
    assert(type_data in ["knapsack", "uci"])
    if type_data == "uci":
        return fitness_fs
    else:
        return fitness_knapsack


def main():
    # test fitness for uci
    type_data = "uci"
    dataset = "BreastCancer"
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
    particle = np.random.randint(low=0, high=2, size=dimension)
    data = {
        "train_x": train_X,
        "train_y" : train_Y,
        "test_x": test_X,
        "test_y": test_Y,
        "omega": omega
    }

    fitness = fitness_fs(particles=particle,
                       data=data)

    print(fitness)

    # test the knapsack fitness
    type_data = "knapsack"
    dataset = "ks_8a"
    data_dict = data_loader(type_data=type_data,
                            data_name=dataset)

    np.random.seed(6)
    particle = np.random.randint(low=0, high=2, size=(10, data_dict["count"]))
    fitness = fitness_knapsack(particles=particle,
                             data=data_dict)
    print(fitness)


if __name__ == "__main__":
    main()



