from collections import deque
import time
import contextlib
import statistics
import json
import os
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Enums and Config
# ---------------------------------------------------------

class SortBy(str, Enum):
    TOTAL = "total"
    MEAN = "mean"
    MAX = "max"
    COUNT = "count"
    NAME = "name"

class ReportColumn(str, Enum):
    NAME = "Name"
    MEAN = "Mean"
    MEDIAN = "Median"
    P95 = "P95"
    MAX = "Max"
    PCT_ROOT = "% Root"
    PCT_PARENT = "% Parent" # New Column
    COUNT = "Cnt"
    TOTAL = "Total"

# ---------------------------------------------------------
# Data Model
# ---------------------------------------------------------

class TimerNode(BaseModel):
    name: str
    durations: list = Field(default_factory=lambda: list())
    children: Dict[str, 'TimerNode'] = Field(default_factory=dict)

    class Config:
        extra = "ignore"

    def add_measurement(self, duration: float):
        self.durations.append(duration)

    @property
    def count(self) -> int:
        return len(self.durations)

    @property
    def total_time(self) -> float:
        return sum(self.durations)

    @property
    def mean(self) -> float:
        return statistics.mean(self.durations) if self.durations else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.durations) if self.durations else 0.0

    @property
    def max_time(self) -> float:
        return max(self.durations) if self.durations else 0.0

    def percentile(self, n: int) -> float:
        if not self.durations: return 0.0
        try:
            return statistics.quantiles(self.durations, n=100, method='exclusive')[n-1]
        except statistics.StatisticsError:
            return self.max_time

# ---------------------------------------------------------
# The Profiler Class
# ---------------------------------------------------------

class Profiler:
    def __init__(self, enabled: bool = True, jsonl_log_path: Optional[str] = None):
        self._roots: Dict[str, TimerNode] = {}
        self._stack: List[TimerNode] = []
        self.enabled = enabled
        self._log_file = None

        if self.enabled and jsonl_log_path:
            self._log_file = open(jsonl_log_path, "a", buffering=1, encoding="utf-8")

    def __del__(self):
        if self._log_file:
            self._log_file.close()

    @contextlib.contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield None
            return

        current_node: TimerNode
        if self._stack:
            parent = self._stack[-1]
            if name not in parent.children:
                parent.children[name] = TimerNode(name=name)
            current_node = parent.children[name]
        else:
            if name not in self._roots:
                self._roots[name] = TimerNode(name=name)
            current_node = self._roots[name]

        self._stack.append(current_node)
        start_ts = time.perf_counter()

        try:
            yield current_node
        finally:
            end_ts = time.perf_counter()
            duration = end_ts - start_ts

            if self._log_file:
                path = [n.name for n in self._stack]

            self._stack.pop()
            current_node.add_measurement(duration)

            if self._log_file:
                record = {"path": path, "duration": duration}
                self._log_file.write(json.dumps(record) + "\n")

    def load_from_jsonl(self, filepath: str):
        self._roots = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        path: List[str] = record.get("path", [])
                        duration: float = record.get("duration", 0.0)

                        if not path: continue

                        root_name = path[0]
                        if root_name not in self._roots:
                            self._roots[root_name] = TimerNode(name=root_name)

                        current = self._roots[root_name]
                        for step in path[1:]:
                            if step not in current.children:
                                current.children[step] = TimerNode(name=step)
                            current = current.children[step]

                        current.add_measurement(duration)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"File {filepath} not found.")

    # -----------------------------------------------------
    # FLEXIBLE REPORT GENERATION
    # -----------------------------------------------------
    def report(self,
               sort_by: SortBy = SortBy.TOTAL,
               min_count: int = 0,
               columns: List[ReportColumn] = None) -> str:

        # Default columns if not provided
        if not columns:
            columns = [
                ReportColumn.NAME,
                ReportColumn.MEAN,
                ReportColumn.MAX,
                ReportColumn.PCT_ROOT,
                ReportColumn.PCT_PARENT, # Added by default
                ReportColumn.COUNT,
                ReportColumn.TOTAL
            ]

        def get_sort_key(node: TimerNode):
            if sort_by == SortBy.TOTAL: return node.total_time
            if sort_by == SortBy.MEAN:  return node.mean
            if sort_by == SortBy.MAX:   return node.max_time
            if sort_by == SortBy.COUNT: return node.count
            if sort_by == SortBy.NAME:  return node.name
            return 0

        is_reverse = sort_by != SortBy.NAME

        # Column Configuration (Widths and Headers)
        col_cfg = {
            ReportColumn.NAME:       {"w": 60, "align": "<"},
            ReportColumn.MEAN:       {"w": 9,  "align": "<"},
            ReportColumn.MEDIAN:     {"w": 9,  "align": "<"},
            ReportColumn.P95:        {"w": 9,  "align": "<"},
            ReportColumn.MAX:        {"w": 9,  "align": "<"},
            ReportColumn.PCT_ROOT:   {"w": 8,  "align": "<"},
            ReportColumn.PCT_PARENT: {"w": 8,  "align": "<"},
            ReportColumn.COUNT:      {"w": 5,  "align": "<"},
            ReportColumn.TOTAL:      {"w": 9,  "align": "<"},
        }

        # Build Header
        header_parts = []
        for col in columns:
            cfg = col_cfg[col]
            header_parts.append(f"{col.value:{cfg['align']}{cfg['w']}}")

        header_str = " | ".join(header_parts)
        lines = [header_str, "-" * len(header_str)]

        sorted_roots = sorted(self._roots.values(), key=get_sort_key, reverse=is_reverse)

        for i, root in enumerate(sorted_roots):
            lines.extend(
                self._format_node(
                    node=root,
                    prefix="",
                    is_last=(i == len(sorted_roots) - 1),
                    is_root=True,
                    sort_key_func=get_sort_key,
                    reverse=is_reverse,
                    min_count=min_count,
                    root_total_time=root.total_time,
                    parent_total_time=root.total_time, # Root belongs to itself/global, treat as 100%
                    columns=columns,
                    col_cfg=col_cfg
                )
            )

        if len(lines) == 2: return "No data."
        return "\n".join(lines)

    def _format_node(self, node: TimerNode, prefix: str, is_last: bool, is_root: bool,
                     sort_key_func, reverse: bool, min_count: int,
                     root_total_time: float, parent_total_time: float,
                     columns: List[ReportColumn], col_cfg: Dict) -> List[str]:

        if node.count < min_count: return []

        # -- Helpers --
        def fmt_t(val: float) -> str:
            if val < 0.001: return f"{val*1e6:.0f}µs"
            if val < 1.0:   return f"{val*1e3:.1f}ms"
            return f"{val:.3f}s"

        # -- Calculate Tree Name --
        name_width = col_cfg[ReportColumn.NAME]['w']
        if is_root:
            display_name = f"• {node.name}"
            child_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            display_name = f"{prefix}{connector}{node.name}"
            child_prefix = prefix + ("    " if is_last else "│   ")

        # Truncate
        if len(display_name) > name_width:
             display_name = display_name[:name_width-3] + "..."

        # -- Calculate Stats --
        pct_root = (node.total_time / root_total_time * 100) if root_total_time > 0 else 0
        pct_parent = (node.total_time / parent_total_time * 100) if parent_total_time > 0 else 0

        # -- Build Row String --
        row_parts = []
        for col in columns:
            cfg = col_cfg[col]
            w = cfg['w']
            align = cfg['align']

            val_str = ""
            if col == ReportColumn.NAME:       val_str = display_name
            elif col == ReportColumn.MEAN:     val_str = fmt_t(node.mean)
            elif col == ReportColumn.MEDIAN:   val_str = fmt_t(node.median)
            elif col == ReportColumn.P95:      val_str = fmt_t(node.percentile(95))
            elif col == ReportColumn.MAX:      val_str = fmt_t(node.max_time)
            elif col == ReportColumn.PCT_ROOT: val_str = f"{pct_root:.1f}%"
            elif col == ReportColumn.PCT_PARENT: val_str = f"{pct_parent:.1f}%"
            elif col == ReportColumn.COUNT:    val_str = str(node.count)
            elif col == ReportColumn.TOTAL:    val_str = fmt_t(node.total_time)

            row_parts.append(f"{val_str:{align}{w}}")

        output = [" | ".join(row_parts)]

        # -- Recurse --
        sorted_children = sorted(node.children.values(), key=sort_key_func, reverse=reverse)
        for i, child in enumerate(sorted_children):
            output.extend(
                self._format_node(
                    node=child,
                    prefix=child_prefix,
                    is_last=(i == len(sorted_children) - 1),
                    is_root=False,
                    sort_key_func=sort_key_func,
                    reverse=reverse,
                    min_count=min_count,
                    root_total_time=root_total_time,
                    parent_total_time=node.total_time, # Current node is new parent
                    columns=columns,
                    col_cfg=col_cfg
                )
            )

        return output

# ---------------------------------------------------------
# Usage Example
# ---------------------------------------------------------

if __name__ == "__main__":
    LOG_FILE = "runtime_log.jsonl"
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

    prof = Profiler(jsonl_log_path=LOG_FILE)

    # 1. Generate Data
    with prof.time("Main Request"):
        time.sleep(0.05)
        for _ in range(3):
            with prof.time("Database Call"):
                time.sleep(0.02)
                with prof.time("Connect"):
                    time.sleep(0.005)
                with prof.time("Query"):
                    time.sleep(0.015)

        with prof.time("Processing Results"):
            time.sleep(0.03)

    # 2. Reconstruct
    rec_prof = Profiler()
    rec_prof.load_from_jsonl(LOG_FILE)

    print("\n--- Standard Report (Default Columns) ---")
    print(rec_prof.report())

    print("\n--- Custom Report: Focus on Relative Time ---")
    # Define your specific column list here
    my_columns = [
        ReportColumn.NAME,
        ReportColumn.TOTAL,
        ReportColumn.PCT_PARENT, # <--- Here it is
        ReportColumn.PCT_ROOT,
        ReportColumn.COUNT
    ]

    print(rec_prof.report(sort_by=SortBy.TOTAL, columns=my_columns))
