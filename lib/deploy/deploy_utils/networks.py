from enum import Enum


class MockNetworkType(str, Enum):
    SIM = 'sim'
    SIM_SYNC_MUJOCO = 'sim_sync_mujoco'


def is_mock_network(network: str) -> bool:
    return network in [MockNetworkType.SIM, MockNetworkType.SIM_SYNC_MUJOCO]


def is_sync_sim_network(network: str) -> bool:
    return network == MockNetworkType.SIM_SYNC_MUJOCO
