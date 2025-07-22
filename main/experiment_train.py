# %%
from collections import OrderedDict
from typing import List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr as fl
import numpy as np
import torch
import glob
import os
from flwr.common import ndarrays_to_parameters
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, FitRes, Parameters, Scalar
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

# %%
NUM_CLIENTS = 2
BATCH_SIZE = 256
NUM_ROUNDS = 200

from test_flwr import get_all_vocab, split_data
vocab = get_all_vocab(["datasets/20NG"])
datasets = split_data(dir = "datasets/20NG", num_split=NUM_CLIENTS, vocab = vocab, batch_size= BATCH_SIZE)

# %%
from model.ETM import ETM
from trainer.basic_trainer import BasicTrainer



# %%
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# %%
from data.basic_dataset import RawDataset
class FlowerClient(NumPyClient):
  def __init__(self, net, dataset : RawDataset, id):
    self.net = net
    self.dataset = dataset
    self.trainer = BasicTrainer(net, dataset, epochs = 1, log_interval=10, device = DEVICE, save_model = True, save_interval=NUM_ROUNDS)
    self.id = id
    self.save_dir = "model_parameters/"
    self.round_id = 0
    self.total_round = NUM_ROUNDS

  # return the current local model parameters
  def get_parameters(self, config):
    return get_parameters(self.net)

  # receive global parameter, train, return updated model to server
  def fit(self, parameters, config):
    set_parameters(self.net, parameters)
    self.trainer.train(model_name = f"ETM_Client{self.id}")

    return get_parameters(self.net), len(self.dataset.train_texts), {}

  # receive global parameter, evaluate model from local's data, return the evaluation result
  def evaluate(self, parameters, config):
    set_parameters(self.net, parameters)
    loss, acc = -1, -1
    return float(loss), 1, {"accuracy":float(acc)}


test = FlowerClient(ETM(len(vocab)), datasets[0], 0)

# %%
def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = ETM(len(vocab)).to(DEVICE)

    # Load data
    partition_id = context.node_config["partition-id"]
    dataset = datasets[partition_id]

    # Create a single Flower client representing a single organization
    return FlowerClient(net, dataset, partition_id).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)

# %%
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays to disk
            if server_round % 10 == 0:
                print(f"Saving round {server_round} aggregated_ndarrays...")
                np.savez(f"model_parameters/model_round_{server_round}.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


# Create strategy and pass into ServerApp
def server_fn(context):
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)


server = ServerApp(server_fn=server_fn)

# %%
def get_latest_server_model(net):
    list_of_files = [fname for fname in glob.glob("model_parameters/model_round_*")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    
    # Load NumPy arrays from .npz file
    with np.load(latest_round_file) as data:
        arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
    
    # Convert to PyTorch state_dict
    state_dict = {k: torch.from_numpy(v) for k, v in zip(net.state_dict().keys(), arrays)}
    net.load_state_dict(state_dict)
    
    # Convert to Flower Parameters
    state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
    parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)
    return parameters

# %%
# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

# %%
# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
    verbose_logging=False
)

# %%
from utils._utils import get_top_words
net = ETM(len(vocab))
test = get_latest_server_model(net)
beta = net.get_beta().detach().cpu().numpy()
topwords = get_top_words(beta, vocab, 15, verbose=True)

# %%
trainer = BasicTrainer(net, datasets[0])

# %%
net.to(DEVICE)

# %%
########################### test new documents ####################################
from data.preprocess import Preprocess

preprocess = Preprocess()

new_docs = [
    "This is a new document about space, including words like space, satellite, launch, orbit.",
    "This is a new document about Microsoft Windows, including words like windows, files, dos."
]

parsed_new_docs, new_bow = preprocess.parse(new_docs, vocab)
print(new_bow.shape)

print(new_bow.toarray())
input = torch.as_tensor(new_bow.toarray(), device="cuda").float()
print(input)
new_theta = trainer.test(input)

print(new_theta.argmax(1))
for x in new_theta.argmax(1):
    print(topwords[x])


