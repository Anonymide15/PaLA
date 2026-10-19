"""
This is the main entry point for the simulation. You can run this script

Author: Anonymous
Date: 02-04-2025
Version: 2.0
"""
import constants
from federated_network.network import FederatedNetwork

import os
import random
import numpy as np
import torch

# Prevents the error on CUDA device-side assertion failure, which are likely triggered by invalid tensor operations
# (e.g., NaN, Inf, or out-of-bounds values) during loss computation in the training loop.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic enough for experiments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything(42)

    async_drift_specs = dict(
        num_drift_groups=2,  # Number of groups of clients that are affected by the drift asynchronously
        drift_groups=None,  # Groups of clients that are affected by the drift asynchronously
        drift_split_round=0.8,  # Times at which the drift is split into multiple asynchronous drifts,
        # Whether to read the asynchronous drift patterns from the ./drift_concepts/scenarios directory
        is_read_scenarios=True,
        # Scenario number for the asynchronous drift patterns defined in the ./drift_concepts/scenarios directory
        scenario_num=1
    )

    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.8,  # Fraction of clients that are drift affected(literature also uses a list of fractions)
        # Proportions of the size of the drift affected client groups at each drift_step_rounds.
        # outer list - timesteps
        # inner list: group size proportions
        # drift_group_proportions=[[0.5, 0.5],  # Scenario C: At the first drift step - drift_step_rounds[0]
        #                          [0.3, 0.7],  # drift_step_rounds[1]
        #                          [0.8, 0.2]],
        # drift_group_proportions=[[0.5, 0.5],  # Scenario A
        #                          [0.2, 0.8]],
        drift_group_proportions=[[0.1, 0.9],  # Scenario B
                                 [0.8, 0.2]],
        # drift_group_proportions=[[1],  # At the first drift step - drift_step_rounds[0]
        #                          [1],  # drift_step_rounds[1]
        #                          [1]],
        # drift_step_rounds[2], the last drift_step_round indicates the end of the drift period.
        is_synchronous=False,  # If the drift is synchronous or asynchronous
        is_random=True,  # Whether to randomly select the clients to be affected by the drift
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        # --------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.LABEL_SWAP_ONCE,  # Drift creation method
        # drift_step_rounds=[0.4, 0.6, 0.7, 0.9],
        # # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
        # #--------------------------------------------------------------------------------
        drift_mode=constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS,  # Drift creation method
        # drift_step_rounds=[0.4, 0.65, 0.7, 1],  # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
        drift_step_rounds=[0.4, 0.65, 1],
        # --------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.ROTATION_GRADUAL,  # Drift creation method
        # drift_step_rounds=[0.4, 0.65, 0.7, 0.9],   # In Rotation gradual case, this indicates only the start and end of drift period.
        # # # --------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.ROTATION_GRADUAL_INCREMENTAL,  # Drift creation method
        # drift_step_rounds=[0.2, 0.6, 1],    # In Rotation gradual case, this indicates only the start and end of drift period.
        # # --------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.ROTATION_STEP_INCREMENTAL,  # Drift creation method
        # drift_step_rounds=[0.2, 0.6, 1], # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
        # --------------------------------------------------------------------------------
        # Therefore, it must have at least two entries (start and end of drift).
        max_rotation=45,  # Maximum rotation angle for the drift created by rotations
        class_pairs_to_swap=[[(1, 2), (3, 4)], [(5, 7)]],  # label indices (not the class names)
        # class_pairs_to_swap=[[(1, 2),(3, 4)]],   # label indices (not the class names)
        # -----------------------------------------
        # MNIST, CIFAR-10: for asynchronous drift in clustering algorithms
        drift_pattern_id_map={
            1: [(1, 2), (3, 4)],
            2: [(5, 7)]
        },  # 0 - no drift
        # --------------------
        # drift_patterns_over_time=[[1, 2],
        #                          [1, 2],
        #                          [2, 1]], # Scenario C
        # drift_patterns_over_time=[[1, 1],
        #                           [1, 2]],   # Scenario A
        drift_patterns_over_time=[[1, 2],
                                  [1, 2]],  # Scenario B
        # --------------------
        # Classes to be swapped in the label-swapping drift method
        label_swap_percentage_steps=[1, 1],  # Percentages to swap per step (label-swapping)
        current_drift_step=-1  # Current drift step (used internally during simulation. -1 represents no drift yet)
    )

    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_plot_client_data_distributions=False,  # Whether to plot the client data distributions
        client_ids_to_plot_data_distributions=[0, 1],  # Client IDs whose internal data distributions to be plotted.
        # Whether servers have test data for evaluation or the server accuracy/loss is got by averaging the client test accuracy/losses
        servers_have_test_data=False
    )

    # ######################################################
    # ##### Paper ICSOC 2026- performance experiments ######
    # ######################################################

    # #################### MNIST ##########################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/MNIST/saved_plots_fedavg/',
    #     log_save_path='logs/swap/MNIST/saved_logs_fedavg/')

    # ######################## PaLA (Alias FedEx) ##################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDEX,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/MNIST/saved_plots_fedex/',
        log_save_path='logs/swap/MNIST/saved_logs_fedex/')

    # ##################### Oracle #############################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.ORACLE,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/MNIST/saved_plots_oracle/',
        log_save_path='logs/swap/MNIST/saved_logs_oracle/')

    # ###########################################################
    # ###################### F_MNIST ############################
    #############################################################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/F_MNIST/saved_plots_fedavg/',
        log_save_path='logs/swap/F_MNIST/saved_logs_fedavg/')

    # ################### PaLA (alias FedEx) ########################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDEX,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/F_MNIST/saved_plots_fedex/',
        log_save_path='logs/swap/F_MNIST/saved_logs_fedex/')

    # ###################### Oracle ###########################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.ORACLE,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/F_MNIST/saved_plots_oracle/',
        log_save_path='logs/swap/F_MNIST/saved_logs_oracle/')

    # ##############################################################
    # ######################### CIFAR-10 ###########################
    ################################################################

    # ###################### PaLA (alias FedEx) ####################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDEX,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=_fedex_alpha,  # EMA weight (alpha) parameter for the FedEx algorithm.
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=400,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.CIFAR_10,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/CIFAR-10/saved_plots_fedex/',
        log_save_path='logs/swap/CIFAR-10/saved_logs_fedex/')

    # ####################### Oracle #####################################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.ORACLE,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=400,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.CIFAR_10,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/CIFAR-10/saved_plots_oracle/',
        log_save_path='logs/swap/CIFAR-10/saved_logs_oracle/')

    # ################### FedAvg ################################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=400,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.CIFAR_10,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/CIFAR-10/saved_plots_fedavg/',
        log_save_path='logs/swap/CIFAR-10/saved_logs_fedavg/')

    #################################################################################
    ################### For Testing & Ablation study purposes #######################
    #################################################################################
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=0,  # Number of IID clients in the federated network
        num_noniid_client_instances=10,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        num_training_rounds=20,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # # FOR TEST RUNS: Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/test/saved_plots_fedex/',
    #     log_save_path='logs/swap/test/saved_logs_fedex/')

    # #########################################################
    # #### Optimal L (layer removal) Ablation experiments #####
    ###########################################################
    # # For these experiments, please refer to the modifications done inside the fedex.py amd utils.py files
    # # Only CIFAR-10 and MNIST dataset is used for these experiments
    # #############################################
    # ################ MNIST ######################
    ###############################################
    # # case 1: dropping the layer 'fc2' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_1_rm_fc_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_1_rm_fc_2/saved_logs_fedex/')
    # --------------------------------------------------------
    # # case 2: dropping the layer 'fc2' and 'fc1' (last 2 layers) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_2_rm_fc_1_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_2_rm_fc_1_2/saved_logs_fedex/')
    # --------------------------------------------------------
    # # case 3: dropping all layer except layer 'fc2' (last layer) in FedEx (aggregating only the last layer)
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_3_agg_fc_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_3_agg_fc_2/saved_logs_fedex/')
    # #--------------------------------------------------------
    # # case 4: add an additional fully connected layer 'fc3' after 'fc2' in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_4_add_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_4_add_fc_3/saved_logs_fedex/')
    # #--------------------------------------------------------
    # # case 5: dropping the layer 'fc3' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_5_rm_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_5_rm_fc_3/saved_logs_fedex/')
    # #--------------------------------------------------------
    # # case 6: keep only the layer 'fc3' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_6_agg_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_6_agg_fc_3/saved_logs_fedex/')
    # #--------------------------------------------------------
    #     # # case 7: keep only the layers 'fc2' and 'fc3' (last 2 layers) in FedEx
    #     # fed_net.run_simulation(
    #     #     file_save_path='plots/swap/layer_removal/MNIST/case_7_agg_fc_2_3/saved_plots_fedex/',
    #     #     log_save_path='logs/swap/layer_removal/MNIST/case_7_agg_fc_2_3/saved_logs_fedex/')
    # --------------------------------------------------------
    # # case 8: Drop layers 'fc2' (middle layer) with fc3 available in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_8_agg_fc_2_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_8_agg_fc_2_3/saved_logs_fedex/')

    ######################################################
    ##################### CIFAR-10 #######################
    ######################################################
    # # case 1: dropping the layer 'fc2' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_1_rm_fc_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_1_rm_fc_2/saved_logs_fedex/')

    # --------------------------------------------------------
    # # case 2: dropping the layer 'fc2' and 'fc1' (last 2 layers) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_2_rm_fc_1_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_2_rm_fc_1_2/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 3: dropping all layer except layer 'fc2' (last layer) in FedEx (aggregating only the last layer)
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_3_agg_fc_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_3_agg_fc_2/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 4: add an additional fully connected layer 'fc3' after 'fc2' in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_4_add_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_4_add_fc_3/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 5: dropping the layer 'fc3' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_4_add_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_4_add_fc_3/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 6: keep only the layer 'fc3' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_6_agg_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_6_agg_fc_3/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 7: keep only the layers 'fc2' and 'fc3' (last 2 layers) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_7_agg_fc_2_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_7_agg_fc_2_3/saved_logs_fedex/')

    # --------------------------------------------------------
    # # case 8: Drop layers 'fc2' (middle layer) with fc3 available in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_8_agg_fc_2_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_8_agg_fc_2_3/saved_logs_fedex/')


if __name__ == "__main__":
    main()
