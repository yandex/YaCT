import typing as tp

from dataclasses import dataclass

@dataclass
class NodeState:
    payload: tp.Optional[tp.Any] = None
    cause: tp.Optional[str] = None

class UnsetState(NodeState):
    ...

class ActiveState(NodeState):
    ...

class BusyState(NodeState):
    ...

class DisabledState(NodeState):
    ...

class MoveToEdgeState(NodeState):
    payload: int


class Node:

    type_count: int = 0

    def __init__(self):
        self._edges = []
        self._state = DisabledState()
        self.type_count = self.__class__.type_count
        self.__class__.type_count += 1

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.type_count = 0

    @property
    def edges(self) -> tp.List["Edge"]:
        return self._edges

    def add_edge(self, edge: "Edge"):
        self.edges.append(edge)

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}-{self.type_count}"

    @property
    def state(self) -> NodeState:
        return self._state

    @state.setter
    def state(self, new_state: NodeState):
        self._state = new_state

    def activate(self):
        raise NotImplementedError()

    def deactivate(self):
        raise NotImplementedError()

    def get_info(self) -> tp.Dict[str, tp.Any]:
        return {}


class StartableNode(Node):
    ...


@dataclass
class Edge:
    source: Node
    target: Node


def edge_trigger_keys(edge_id: int, to_home: bool) -> tp.List[str]:
    """Keys/buttons that trigger an edge: the keyboard number always; ``x`` / VR-A
    also reach the home (damp) node; VR-B also triggers the forward edge (id 1)."""
    keys = [str(edge_id)]
    if to_home:
        keys += ["x", "VR-A"]
    if edge_id == 1:
        keys.append("VR-B")
    return keys


def find_node_of_type(node: Node, nodetype: Node) -> int:
    for i, e in enumerate(node.edges):
        if isinstance(e.target, nodetype):
            return i
    raise ValueError(f"{nodetype} node not found! Connect this node with {nodetype}")


if __name__ == '__main__':
    nodes = [Node(), Node(), StartableNode(), StartableNode()]
    for n in nodes:
        print(n.name)
