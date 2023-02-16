import datetime
import os
from typing import Optional, Dict


class BaseConfig:
    """
    Base config class which should be subclassed for custom configuration
    """
    def __init__(self):
        self.problem_specifics = dict()  # Problem specific options such as problem size, network configs etc.

        # Gumbel AlphaZero specific parameters
        self.gumbel_sample_n_actions: int = 20  # (Max) number of actions to sample at the root for the sequential halving procedure
        self.gumbel_c_visit: float = 50.  # constant c_visit in sigma-function.
        self.gumbel_c_scale: float = 1.  # constant c_scale in sigma-function.
        self.gumbel_simple_loss: bool = True  # If True, KL divergence is minimized w.r.t. one-hot-encoded move, and not
        # w.r.t. distribution based on completed Q-values
        self.gumbel_test_greedy_rollout = False  # If True, then in evaluation mode the policy of the learning actor is rolled out greedily (no MCTS)
        self.gumbel_is_gt = False  # If True, we use the variant GAZ PTP GT, where the greedy actor is also simulated greedily by the learning actor.
        self.num_simulations: int = 25  # Number of search simulations in GAZ's tree search
        # ----

        self.singleplayer_options: Optional [Dict] = {   # Options for singleplayer variants. See individual folders for examples.
            "type": "greedy_scalar"
        }

        self.seed: int = 42  # Random seed for torch, numpy, initial states.

        # --- Inferencer and experience generation options --- #
        self.num_experience_workers: int = 3  # Number of actors (processes) which generate experience.
        self.num_inference_workers: int = 1  # Number of workers which perform network inferences. Each inferencer is pinned to a CPU.
        self.inference_on_experience_workers: bool = False  # If True, states are not sent to central actors but performed
                                                           # directly on the experience worker
        self.check_for_new_model_every_n_seconds: int = 30  # Check the storage for a new model every n seconds

        self.pin_workers_to_core: bool = True  # If True, workers are pinned to specific CPU cores, starting to count from 0.
        self.CUDA_VISIBLE_DEVICES: str = "0"  # Must be set, as ray can have problems detecting multiple GPUs
        self.cuda_device: str = "cuda:0"  # Cuda device on which to *train* the network. Set to `None` if not available.

        # For each inference worker, specify on which device it can run. Set to `None`, if shouldn't use a GPU.
        # Expects a list of devices, where i-th entry corresponds to the target device of i-th inference worker.
        # If `inference_on_experience_workers` is True, and length of `cuda_devices_for_inference_workers` equals
        # number of experience workers, the LocalInferencers on the experience workers are assigned the relative devices.
        # Else inference is performed on CPU by default.
        self.cuda_devices_for_inference_workers: [str] = ["cuda:0"] * self.num_inference_workers

        # Number of most recent games to store in the replay buffer
        self.replay_buffer_size: int = 2000

        self.best_starts_randomly: bool = True  # For debugging purposes only. If this is True, the greedy actor moves
        # uniform randomly until it has been dominated once. Used to test how well
        # the agent can outperform random play as an indicator if it is learning at all.

        # with this probability, the learning actor plays against its own current policy to stabilize training
        # and escape bad initializations. See paper for details.
        self.initial_self_play_parameter: float = 0.2
        self.reduce_self_play_parameter_after_n_games: int = 50
        self.self_play_parameter: float = 0.2

        # --- Training / Optimizer specifics --- #

        # Tries to keep the ratio of training steps to number of episodes within the given range.
        # Set it to None to disable it
        self.ratio_range: [float] = [0.0, 1.5]
        self.start_train_after_episodes = 200  # wait for n episodes before starting training
        # Total number of batches to train the network on
        self.training_games: int = 1000

        self.batch_size: int = 8  # Batch size for training.
        self.lr_init: float = 0.0001  # Initial learning rate
        self.weight_decay: float = 1e-4  # L2 weights regularization
        self.gradient_clipping: float = 1  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
        self.lr_decay_rate: float = 1  # Set it to 1 to use a constant learning rate.  Note: Currently unused.
        self.lr_decay_steps: float = 350e3  # means that after `decay_steps` training steps, the learning rate has decayed by `decay_rate`. Note: Currently unused.

        self.value_loss_weight: float = 1.0  # Linear scale of value loss
        self.checkpoint_interval: int = 10  # Number of training steps before using the model for generating experience.

        # --- Arena --- #
        self.arena_checkpoint_interval: int = 50  # Number of training steps until learning actor is pitted against greedy actor.
        self.arena_set_path: str = "./test/JSSP/jsp_6_6_arena.npy"  # Path to the arena set.
        self.num_arena_games: int = 20  # Number of arena games to play.
        self.arena_criteria_win_ratio: float = 0  # Ratio of arena games the learning actor has to win in order to beat the greedy actor.
                                                  # If set to 0, then the mean objectives are compared.
        # If set to 0, the total outcome gap is used instead as criterion, as in paper.
        self.results_path: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results",
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights

        self.save_model: bool = False  # Save the checkpoint in results_path as model.checkpoint
        self.load_checkpoint_from_path: Optional[str] = None  # If given, model weights and optimizer state is loaded from this path.
        self.only_load_model_weights: bool = False  # If True, only the model weights are loaded from `load_checkpoint_from_path`
        # Optimizer state, number of played games etc., is freshly created.

        # --- Logging --- #
        self.log_avg_stats_every_n_episodes: int = 10  # Compute average episode statistics over last n episodes and log them
        self.log_avg_loss_every_n_steps: int = 10  # Compute average loss over last n training steps and log them
        self.log_policies_for_moves: [int] = [1, 18, 30]  # Logs stats about probability distribution for n-th moves
        self.do_log_to_file: bool = True

        # --- Evaluation --- #
        self.num_evaluation_games: int = 10  # For each validation run, how many instances should be solved of the validation set (taken from the start)
        self.validation_set_path: str = "./test/JSSP/jsp_6_6_validation.npy"
        self.test_set_path: str = "./test/JSSP/jsp_6_6_validation.npy"
        self.evaluate_every_n_steps: int = 1000  # Make evaluation run every n training steps
