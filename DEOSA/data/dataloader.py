import DEOSA.config as config

import pandas as pd
import numpy as np


def data_loader(type_data,
                data_name):
    """
    :param type_problem: problem type (uci/knapsack)
    :param data_name: Name of the dataset
    :return: a dataframe containing the data for uci,
             data dictionary for knapsack
    """
    assert(type_data in ["uci", "knapsack"])

    if type_data == "uci":
        df = pd.read_csv(f"{config.BASE_PATH}/data/{type_data}/{data_name}.csv", header=None)
        return df

    else:
        f = open(f"{config.BASE_PATH}/data/{type_data}/{data_name}.txt", 'r')
        count = int(f.readline())

        weights = np.array(list(map(int, f.readline().split())))
        worth = np.array(list(map(int, f.readline().split())))
        max_weight = int(f.readline())

        data = {
            "count": count,
            "weights": weights,
            "values": worth,
            "max_weight": max_weight
        }

        return data


def main():
    # test for uci dataloader
    type_data = "uci"
    data_name = "BreastCancer"
    df = data_loader(type_data=type_data,
                     data_name=data_name)
    print(df)

    # test for knapsack dataloader
    type_data = "knapsack"
    data_name = "ks_8a"
    data_dict = data_loader(type_problem=type_data,
                     data_name=data_name)
    print(data_dict)


if __name__ == "__main__":
    main()