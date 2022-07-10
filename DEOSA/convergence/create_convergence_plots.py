import DEOSA.config as config
from DEOSA.data.get_data import create_data_dict
from DEOSA.utilities.path_utils import create_dir
from DEOSA.algorithm import DEOSA

import pickle
import numpy as np
import matplotlib.pyplot as plt

datasets = {
    "uci": ["BreastCancer", "BreastEW", "CongressEW", "Exactly", "Exactly2", "HeartEW", "Ionosphere", "KrVsKpEW",
            "Lymphography", "M-of-n", "PenglungEW", "Sonar", "SpectEW", "Tic-tac-toe", "Vote", "WaveformEW", "Wine",
            "Zoo"],
    "knapsack": ["ks_8a", "ks_8b", "ks_8c", "ks_8d", "ks_8e", "ks_12a", "ks_12b", "ks_12c", "ks_12d", "ks_12e", "ks_16a", "ks_16b", "ks_16c", "ks_16d", "ks_16e", "ks_20a", "ks_20b", "ks_20c", "ks_20d", "ks_20e", "ks_24a", "ks_24b", "ks_24c", "ks_24d", "ks_24e"]
}

def run_experiments(datasets,
                    num_runs,
                    max_iter):

    for type_data in datasets.keys():
        for dataset_name in datasets[type_data]:
            data_dict = create_data_dict(type_data=type_data,
                                         dataset_name=dataset_name,
                                         seed=0)
            for type_algorithm in ["DEO", "DEOSA"]:
                for run_no in range(num_runs):
                    folder_name = f"storage/{type_data}/{dataset_name}/{type_algorithm}"
                    create_dir(folder_name)
                    if type_algorithm == "DEO":
                        best_solution = DEOSA(data=data_dict,
                                              type_data=type_data,
                                              seed=run_no,
                                              max_iter=max_iter,
                                              allow_SA=False)
                    else:
                        best_solution = DEOSA(data=data_dict,
                                              type_data=type_data,
                                              seed=run_no,
                                              max_iter=max_iter,
                                              allow_SA=True)

                    pickle.dump(best_solution, open(f"{config.BASE_PATH}/{folder_name}/final_solution_"
                                                    f"{dataset_name}_run_{run_no}.pickle", "wb"))


def plot_convergence():
    num_runs = 2
    max_iter = 30

    run_experiments(datasets=datasets,
                    num_runs=num_runs,
                    max_iter=max_iter)

    for type_data in datasets.keys():
        for dataset_name in datasets[type_data]:
            solutions = {"DEO": [],
                         "DEOSA": []}
            for type_algorithm in ["DEO", "DEOSA"]:
                for run_no in range(num_runs):
                    folder_name = f"storage/{type_data}/{dataset_name}/{type_algorithm}"
                    file_path = f"{config.BASE_PATH}/{folder_name}/final_solution_{dataset_name}_run_{run_no}.pickle"
                    solutions[type_algorithm].append(pickle.load(open(file_path, "rb"))["conv_plot"])

            fig, axes = plt.subplots()
            for type_algorithm in ["DEO", "DEOSA"]:
                solutions[type_algorithm] = np.array(solutions[type_algorithm])
                axes.plot(np.arange(max_iter), solutions[type_algorithm].mean(axis=0), label=type_algorithm)

            axes.legend(loc="lower right")
            axes.set_title(dataset_name)
            axes.set_xlabel("Iterations")
            axes.set_ylabel("Fitness")
            fig.savefig(f"plots/convergence_{dataset_name}.jpg")
            plt.show()




def main():
    plot_convergence()

if __name__ == "__main__":
    main()