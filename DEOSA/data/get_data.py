from DEOSA.data.dataloader import data_loader

import numpy as np
from sklearn.model_selection import train_test_split

def create_data_dict(type_data,
                     dataset_name,
                     seed=0):

    assert(type_data in ["uci", "knapsack"])
    if type_data == "uci":
        omega = 0.9

        # reading dataset
        df = data_loader(type_data=type_data,
                         data_name=dataset_name)
        (a, b) = np.shape(df)
        data = df.values[:, 0:b - 1]
        label = df.values[:, b - 1]
        dimension = np.shape(data)[1]  # particle dimension

        # loading_dataset
        train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.2, random_state=0)
        np.random.seed(0)
        data_dict = {
            "name": dataset_name,
            "train_x": train_X,
            "train_y": train_Y,
            "test_x": test_X,
            "test_y": test_Y,
            "omega": omega
        }

        return data_dict

    else:
        data_dict = data_loader(type_data=type_data,
                                data_name=dataset_name)
        data_dict["name"] = dataset_name

        return data_dict