import inspect
import time
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional
from lib.deploy.debug.logs import DeployConsole, _display_fsm_node
from lib.deploy.fsm.common import StartableNode, Edge, MoveToEdgeState, edge_trigger_keys


@dataclass
class FsmNodeTransition:
    timestamp: float
    source_node_name: str
    target_node_name: str
    cause: Optional[str]

class FiniteStateMachine:
    """
    TODO: Make use of Busy node status
    Make available auto transitions to some nodes!
    """
    def __init__(
        self,
        start_node: StartableNode,
        iter_time: float = 0.01,
        print_every: float = 0.5,
        *,
        console: Optional[DeployConsole] = None,
    ):
        assert isinstance(start_node, StartableNode)
        self._current_node = start_node
        self._home_node = start_node  # damp / safety node (targets of x / VR-A)
        self._move_to_edge = None
        self._iter_time = iter_time
        self._print_every_iters = max(int(print_every / iter_time), 1)
        self.console = (
            console
            if console is not None
            else DeployConsole(self.__class__.__name__)
        )

        # Threading helpers
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._transition_history: List[FsmNodeTransition] = []

    def stop(self, join: bool = True, timeout: float | None = None):
        # Signal the run loop to stop
        self._stop_event.set()

        with self._lifecycle_lock:
            self._current_node.deactivate()
            t = self._thread

        # The worker may be waiting for the lifecycle lock; join outside it.
        if join and t is not None and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout)
        # Retain a live worker handle after a timed or non-joining stop.
        with self._lifecycle_lock:
            if self._thread is t and t is not None and not t.is_alive():
                self._thread = None

    def run_async(self, daemon: bool = True, callbacks = []) -> threading.Thread:
        with self._lifecycle_lock:
            # If already running, return the existing thread
            if self._thread is not None and self._thread.is_alive():
                return self._thread

            # Clear stop flag and start thread
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self.run,
                kwargs={"callbacks": callbacks},
                name=f"FSMThread-{self._current_node.name}",
                daemon=daemon,
            )
            self._thread.start()
            return self._thread

    def move_by_id(self, edge_id: int, cause: str = None):
        with self._lifecycle_lock:
            if self._stop_event.is_set():
                return
            if edge_id >= len(self._current_node.edges):
                self.console.warning(f"There is no edge with such number: {edge_id}")
                return
            edge = self._current_node.edges[edge_id]
            self._move_along_edge(edge, cause=cause)

    def move_by_name(self, target_node_name: str, cause: str = None):
        with self._lifecycle_lock:
            if self._stop_event.is_set():
                return
            found_edges = [e for e in self._current_node.edges if e.target.name == target_node_name]
            if len(found_edges) == 1:
                self._move_along_edge(found_edges[0], cause=cause)
            if len(found_edges) == 0:
                self.console.warning(
                    f"Wrong name, {target_node_name} not found. "
                    f"Names available: {[e for e in self._current_node.edges if e.target.name]}"
                )
            if len(found_edges) > 1:
                self.console.warning("There is a bug. Found >1 node with the same name")

    def _move_along_edge(self, edge: Edge, cause: str = None):
        # Caller holds _lifecycle_lock from edge selection through activation.
        target_node = edge.target
        self.console.transition(
            self._current_node.name,
            target_node.name,
            cause=cause,
        )
        self._transition_history.append(
            FsmNodeTransition(
                timestamp=time.monotonic(),
                source_node_name=self._current_node.name,
                target_node_name=target_node.name,
                cause=cause,
            )
        )
        self._current_node.deactivate()
        if self._stop_event.is_set():
            return
        self._current_node = target_node
        target_node.activate()
        self._print_transition_hints()

    def _print_transition_hints(self):
        """After a node change, show which key/button leads where from here."""
        parts = []
        for edge_id, edge in enumerate(self._current_node.edges):
            if edge.target is self._current_node:
                continue
            keys = edge_trigger_keys(edge_id, edge.target is self._home_node)
            parts.append(f"{' / '.join(keys)} → {_display_fsm_node(edge.target.name)}")
        if parts:
            self.console.info("Next: " + "    ".join(parts))

    def print_transitions(self):
        """Print all available transitions from the current node."""
        self.console.info(f"\n{'='*60}")
        self.console.info(f"FSM TRANSITIONS - Current Node: {self._current_node.name}")
        self.console.info(f"{'='*60}")

        if not self._current_node.edges:
            self.console.info("No transitions available from this node.")
            return

        self.console.info(f"{'ID':<4} {'Target Node':<25} {'Target Type':<20}")
        self.console.info(f"{'-'*4} {'-'*25} {'-'*20}")

        for i, edge in enumerate(self._current_node.edges):
            target_name = edge.target.name
            target_type = edge.target.__class__.__name__
            self.console.info(f"{i:<4} {target_name:<25} {target_type:<20}")

        self.console.info(f"{'='*60}\n")

    def run(self, *, callbacks: List[Callable] = []):
        cnt = 0
        try:
            with self._lifecycle_lock:
                if self._stop_event.is_set():
                    return
                self._current_node.activate()
            while not self._stop_event.is_set():
                time.sleep(self._iter_time)

                with self._lifecycle_lock:
                    if self._stop_event.is_set():
                        break
                    if cnt % self._print_every_iters == 0:
                        self.print_transitions()
                    state = self._current_node.state
                    if isinstance(state, MoveToEdgeState):
                        self.move_by_id(state.payload, cause=state.cause)

                for callback in callbacks:
                    callback()
                cnt += 1
        except Exception:
            self.console.warning("Shutting down FSM.")
            # Avoid deadlock: do not join when stopping from within the same thread
            self.stop(join=False)
            self.console.info("FSM shut down!")
            raise

    @property
    def transition_history(self) -> List[FsmNodeTransition]:
        return self._transition_history
