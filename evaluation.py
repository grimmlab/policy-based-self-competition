import os
import pickle

import numpy as np
import ray
import time

from base_config import BaseConfig
from shared_storage import SharedStorage
from tqdm import tqdm


class Evaluation:
    def __init__(self, config: BaseConfig, shared_storage: SharedStorage):
        self.config = config
        self.shared_storage = shared_storage

    def start_evaluation(self):
        ray.get(self.shared_storage.set_evaluation_mode.remote(True))

    def stop_evaluation(self):
        ray.get(self.shared_storage.set_evaluation_mode.remote(False))

    def evaluate(self, n_episodes: int, set_path: str, save_results: bool = False):
        print("Performing Evaluation...")
        objectives = []

        # Get instances by loading them from the validation file.
        if ".pickle" in set_path:
            with open(set_path, "rb") as handle:
                validation_instances = pickle.load(handle)["instances"]
        elif ".npy" in set_path:
            validation_instances = np.load(set_path)
        else:
            raise Exception("Unknown file type")
        if n_episodes == -1:
            # use all
            n_episodes = validation_instances.shape[0]

        instance_list = [(i, validation_instances[i], "test") for i in range(n_episodes)]

        ray.get(self.shared_storage.set_to_evaluate.remote(instance_list))

        eval_results = [None] * n_episodes

        with tqdm(total=n_episodes) as progress_bar:
            while None in eval_results:
                time.sleep(0.5)
                fetched_results = ray.get(self.shared_storage.fetch_evaluation_results.remote())
                for (i, result) in fetched_results:
                    eval_results[i] = result
                progress_bar.update(len(fetched_results))

        for i, result in enumerate(eval_results):
            if self.config.singleplayer_options is None:
                objective = result["objectives"][result["winner"]]
            elif self.config.singleplayer_options["method"] in ["greedy_scalar", "single_timestep"]:
                objective = min(result["objective"], result["baseline_objective"])
            elif self.config.singleplayer_options["method"] == "single":
                objective = min(result["greedy_rollout"], result["objective"])
            objectives.append(objective)

        objectives = np.array(objectives)

        # Save the objectives for computing margins
        if save_results:
            np.save(os.path.join(self.config.results_path, "eval.npy"), objectives)

        # Compute some stats
        stats = {
            "type": "Validation",
            "avg_objective": objectives.mean(),
        }

        return stats
