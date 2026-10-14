"""Process scheduling helpers for realtime policy threads."""

import logging
import os


def try_enable_rr_scheduler(loop_name: str) -> None:
    """Best-effort switch of the current policy thread to SCHED_RR."""
    priority_raw = os.environ.get("YACT_POLICY_SCHED_RR_PRIORITY", "99")
    try:
        priority = int(priority_raw)
    except ValueError:
        logging.warning(
            "Invalid YACT_POLICY_SCHED_RR_PRIORITY=%r for %s; skipping SCHED_RR",
            priority_raw,
            loop_name,
        )
        return

    if priority <= 0:
        return

    try:
        min_priority = os.sched_get_priority_min(os.SCHED_RR)
        max_priority = os.sched_get_priority_max(os.SCHED_RR)
        bounded_priority = max(min_priority, min(priority, max_priority))
        os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(bounded_priority))
        logging.info(
            "Enabled SCHED_RR for %s with priority=%s",
            loop_name,
            bounded_priority,
        )
    except PermissionError:
        logging.debug(
            "Failed to enable SCHED_RR for %s: permission denied",
            loop_name,
        )
    except AttributeError:
        logging.warning(
            "SCHED_RR is not supported on this platform for %s",
            loop_name,
        )
    except OSError as exc:
        logging.warning(
            "Failed to enable SCHED_RR for %s: %s",
            loop_name,
            exc,
        )
