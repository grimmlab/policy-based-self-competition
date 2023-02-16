import os
import datetime

from base_config import BaseConfig


class Config(BaseConfig):
    """
    TSP N=20
    """

    def __init__(self):
        super().__init__()

        self.problem_specifics = {
            "N_cities": 20,  # number of cities
            "num_attention_heads": 8,
            "num_transformer_blocks": 5,
            "latent_dimension": 128,
            "training_apply_geometry_augmentation": True
            # If True, data augmentation is applied to sampled training states
        }

        # Gumbel AlphaZero specific parameters
        self.gumbel_sample_n_actions = 16  # (Max) number of actions to sample at the root for the sequential halving procedure
        self.gumbel_c_visit = 50.  # constant c_visit in sigma-function.
        self.gumbel_c_scale = 1.  # constant c_scale in sigma-function.
        self.gumbel_simple_loss = False  # If True, KL divergence is minimized w.r.t. one-hot-encoded move, and not
        # w.r.t. distribution based on completed Q-values
        self.gumbel_test_greedy_rollout = False  # If True, then in evaluation mode the policy of the learning actor is rolled out greedily (no MCTS)
        self.num_simulations = 200  # Number of search simulations in GAZ's tree search
        # ----

        self.singleplayer_options = {
            "method": "single",
            "bootstrap_final_objective": False,  # If set to `False`, we bootstrap the value from `bootstrap_n_steps` for training instead.
            "bootstrap_n_steps": 20
        }

        self.seed = 42  # Random seed for torch, numpy, initial states.

        # --- Inferencer and experience generation options --- #
        self.num_experience_workers = 90  # Number of actors (processes) which generate experience.
        self.num_inference_workers = 1  # Number of workers which perform network inferences. Each inferencer is pinned to a CPU.
        self.inference_on_experience_workers: bool = True  # If True, states are not sent to central actors but performed
        # directly on the experience worker
        self.check_for_new_model_every_n_seconds = 30  # Check the storage for a new model every n seconds

        self.pin_workers_to_core = True  # If True, workers are pinned to specific CPU cores, starting to count from 0.
        self.CUDA_VISIBLE_DEVICES = "0,1,2,3"  # Must be set, as ray can have problems detecting multiple GPUs
        self.cuda_device = "cuda:0"  # Cuda device on which to *train* the network. Set to `None` if not available.
        # For each inference worker, specify on which device it can run. Set to `None`, if shouldn't use a GPU.
        # Expects a list of devices, where i-th entry corresponds to the target device of i-th inference worker.
        self.cuda_devices_for_inference_workers = ["cuda:1"] * 3 + ["cuda:2"] * 3 + ["cuda:3"] * 4

        # Number of most recent games to store in the replay buffer
        self.replay_buffer_size = 2000

        self.best_starts_randomly = False  # For debugging purposes only. If this is True, the greedy actor moves
        # uniform randomly until it has been dominated once. Used to test how well
        # the agent can outperform random play as an indicator if it is learning at all.

        # with this probability, the learning actor plays against its own current policy to stabilize training
        # and escape bad initializations. See paper for details.
        self.initial_self_play_parameter: float = 0.2
        self.reduce_self_play_parameter_after_n_games: int = 0
        self.self_play_parameter: float = 0.2

        # --- Training / Optimizer specifics --- #

        # Tries to keep the ratio of training steps to number of episodes within the given range.
        # Set it to None to disable it
        n = self.problem_specifics["N_cities"]
        self.ratio_range = [0.08 * n, 0.1 * n]
        # Total number of batches to train the network on
        self.training_games: int = 100000

        self.batch_size = 256
        self.lr_init = 0.0001  # Initial learning rate
        self.weight_decay = 1e-4  # L2 weights regularization
        self.gradient_clipping = 1  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.

        self.lr_init = 0.0001  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3  # means that after `decay_steps` training steps, the learning rate has decayed by `decay_rate`

        self.value_loss_weight = 1.0  # Linear scale of value loss
        self.checkpoint_interval = 100  # Number of training steps before using the model for generating experience.

        # --- Arena --- #
        self.arena_checkpoint_interval = 400 * 0.1 * n  # Number of training steps until learning actor is pitted against greedy actor.
        self.arena_set_path = "./test/TSP/tsp_20_arena.npy"  # Path to the arena set.
        self.num_arena_games = 200  # Number of arena games to play.
        self.arena_criteria_win_ratio = 0  # Ratio of arena games the learning actor has to win in order to beat
        # the greedy actor.
        # If set to 0, the total outcome gap is used instead as criterion, as in paper.
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results",
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights

        self.save_model = False  # Save the checkpoint in results_path as model.checkpoint
        self.load_checkpoint_from_path = None  # If given, model weights and optimizer state is loaded from this path.
        self.only_load_model_weights = False  # If True, only the model weights are loaded from `load_checkpoint_from_path`
        # Optimizer state, number of played games etc., is freshly created.

        # --- Logging --- #
        self.log_avg_stats_every_n_episodes = 100  # Compute average game statistics over last n games and log them
        self.log_avg_loss_every_n_steps = 100  # Compute average loss over last n training steps and log them
        self.log_policies_for_moves = [1, 10, 15]  # Logs probability distributions for numbered moves
        self.do_log_to_file = True

        # --- Evaluation --- #
        self.num_evaluation_games = 100  # For each validation run, how many instances should be solved in the
        # of the validation set (taken from the start)
        self.validation_set_path = "./test/TSP/tsp_20_validation.pickle"
        self.test_set_path = "./test/TSP/tsp_20_seed1234.npy"
        self.evaluate_every_n_steps = 2500 * n * 0.1  # Make evaluation run every n training steps
