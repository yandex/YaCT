#!/bin/sh
''''exec "${PYTHON_EXECUTABLE:-python3}" "$0" "$@"; exit $? # '''
"""
deploy.py - Deploy using TeleopController with caller-owned loop.

Keyboard controls:
    FSM:   x = DampNode, 0-9 = edge by ID
    Move:  w/s = forward/back, a/d = left/right, q/e = rotate, space = stop
    Sim:   r = reset sim

Usage (uses PYTHON_EXECUTABLE if set, else python3):

    # MuJoCo sim + WASD + auto FSM (headless)
    MUJOCO_GL=egl python3 -m lib.controller.scripts.deploy \
        --wasd --network sim_sync_mujoco --auto-next-node

    # MuJoCo sim + VR replay trajectory
    MUJOCO_GL=egl python3 -m lib.controller.scripts.deploy \
        --network sim_sync_mujoco \
        --vr-replay trajectories/walk_forward.json --auto-next-node

    # Real robot with Vuer WebXR VR (open the printed https URL in the headset browser)
    python3 -m lib.controller.scripts.deploy --vr --network enp3s0

    # Real robot with Dex3 hands + VR
    python3 -m lib.controller.scripts.deploy --vr --network enp3s0 --dex3

    Assets are stored in ordinary Git; run `./yact setup` to verify the checkout.
"""

import argparse
import json
import logging
import os
import time

from tqdm import tqdm

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

from lib.assets.paths import resolve_asset
from lib.vendor.env import LOCAL_ROOT, print_path_lines
from lib.deploy.vr.vr_tracker_wrappers import TCPVelTrackerFromPose
from lib.deploy.vr.vr_replay_wrapper import VRReplayWrapper
from lib.deploy.deploy_utils.networks import is_mock_network, is_sync_sim_network
from lib.deploy.vr.commanded_pose_tracker import CommandedVRPoseTracker

from lib.controller import TeleopController, TeleopAction, GripperAction
from lib.controller.utils.edge_detector import EdgeDetector
from lib.controller.utils.input import ControllerReader, InputReader


from lib.deploy.configs.deploy_defaults import DEFAULT_CHECKPOINT
from lib.deploy.control.mujoco_env import G1MujocoEnv
from lib.deploy.control.live_view import LiveView
from lib.deploy.debug.logs import DeployConsole

CHECKPOINT = DEFAULT_CHECKPOINT


class _VrKeypointLogger:
    """Debug: append per-step VR head/wrist keypoint poses to a JSONL file.

    One JSON object per control step with each keypoint's position and
    orientation (as both a wxyz quaternion and xyz Euler degrees), for
    inspecting whether the tracker's frame conversion is correct.
    """

    _KEYPOINTS = (
        ("head", "get_head_position", "get_head_rotation"),
        ("left_wrist", "get_left_wrist_position", "get_left_wrist_rotation"),
        ("right_wrist", "get_right_wrist_position", "get_right_wrist_rotation"),
    )

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "w", encoding="utf-8")
        self._n = 0

    def log(self, pose_state, sim_t=None) -> None:
        if pose_state is None:
            return
        rec = {"step": self._n, "sim_t": sim_t}
        for name, pos_attr, rot_attr in self._KEYPOINTS:
            pos = getattr(pose_state, pos_attr)()
            rot = getattr(pose_state, rot_attr)()
            if pos is None or rot is None:
                rec[name] = None
                continue
            rec[name] = {
                "pos": [round(float(v), 5) for v in pos],
                "quat_wxyz": [round(float(v), 6) for v in rot.as_quat(scalar_first=True)],
                "euler_deg": [round(float(v), 2) for v in rot.as_euler("xyz", degrees=True)],
            }
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()
        self._n += 1

    def close(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Deploy with TeleopController')
    parser.add_argument('-c', '--checkpoint', default=CHECKPOINT, type=str,
                        help='Path to checkpoint under assets/ or absolute path')
    parser.add_argument('-n', '--network', default='sim_sync_mujoco', type=str,
                        help='Network (sim_sync_mujoco or real robot interface name)')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Inference device')
    parser.add_argument('--vr-replay', default=None, type=str,
                        help='VR replay JSON file for testing')
    parser.add_argument('--dex3', default=False, action='store_true',
                        help='Enable Dex3-1 hands (real robot DDS only; not MuJoCo sim)')
    parser.add_argument('--auto-next-node', default=False, action='store_true',
                        help='Automatically traverse FSM nodes (for sim testing)')
    parser.add_argument('--save-videos', default=None, action=argparse.BooleanOptionalAction,
                        help='Save videos (sim only). Default: True if no DISPLAY.')
    parser.add_argument('--restart-localization', default=False, action='store_true',
                        help='Force restart Fast-LIO on the Jetson (real robot only)')
    parser.add_argument('--vel-v2', default=False, action='store_true',
                        help='Use direct velocity from displacement (skip VrVelPoseTracker re-estimation)')
    parser.add_argument('--wasd', default=False, action='store_true',
                        help='Replace VR tracking with WASD keyboard-commanded pose tracker')
    parser.add_argument('--vr', default=False, action='store_true',
                        help='Live VR via Vuer WebXR (open the printed URL in the headset browser)')
    parser.add_argument('--vr-port', type=int, default=8012,
                        help='HTTPS/WSS port for the Vuer WebXR server (default: 8012)')
    parser.add_argument('--controller', default=False, action='store_true',
                        help='Use game controller for velocity and FSM (experimental)')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--profile', default=False, action='store_true')
    parser.add_argument('--window', default=False, action='store_true',
                        help='Open the interactive MuJoCo viewer window (needs a display; '
                        'the CLI forwards X11 into the container)')
    parser.add_argument(
        '--view',
        default=False,
        action='store_true',
        help='Live MJPEG preview in browser (MuJoCo sim only)',
    )
    parser.add_argument(
        '--view-port',
        type=int,
        default=8766,
        help='HTTP port for --view (default: 8766)',
    )
    parser.add_argument(
        '--no-wait',
        default=False,
        action='store_true',
        help='With --view, start immediately instead of waiting for browser Start / Enter',
    )
    parser.add_argument(
        '--fast',
        default=False,
        action='store_true',
        help='With --view, run the sim as fast as possible (uncap the realtime pacing)',
    )
    return parser.parse_args(argv)


def _noop(*args, **kwargs) -> None:
    pass


def _print_sim_banner(args: argparse.Namespace) -> None:
    if args.vr_replay:
        mode = f"VR replay ({args.vr_replay})"
    elif args.wasd:
        mode = "WASD keyboard"
    elif args.vr:
        mode = "Vuer WebXR VR"
    else:
        mode = "unknown"

    print("YaCT sim")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  mode:       {mode}")
    if args.auto_next_node:
        print("  sequence:   damp → stand → interpolate → teleop")
    if args.view and not args.no_wait:
        print("  start:      click Start in browser (Enter in terminal also works)")
    if args.wasd and not args.vr_replay:
        keys = "WASD · space stop · x damp · 0-9 FSM · r reset · Ctrl+C quit"
        if args.view:
            print(f"  controls:   {keys}  (browser; click page to focus)")
        else:
            print(f"  controls:   {keys}")
    elif args.vr_replay:
        if args.view:
            print("  controls:   x damp · 0-9 FSM · r reset (browser; no WASD in replay)")
        else:
            print("  controls:   x damp · 0-9 FSM · r reset · Ctrl+C quit (no WASD in replay)")
    elif args.vr:
        print("  controls:   VR A/B buttons · keyboard x/0-9 FSM · Ctrl+C quit")
    print()


def _validate_input_mode(args: argparse.Namespace) -> None:
    modes = sum([args.wasd, args.vr_replay is not None, args.vr])
    if modes > 1:
        raise SystemExit("Choose one input mode: --wasd, --vr-replay, or --vr")
    if modes == 0:
        raise SystemExit(
            "No VR input mode selected. Use --wasd, --vr-replay <json>, or --vr."
        )


def _wait_for_vr_connection(vr_input_source, *, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_status = 0.0
    while True:
        state = vr_input_source.get_state()
        if state.is_full():
            return
        now = time.monotonic()
        if now >= deadline:
            missing = []
            if state.head_pose is None:
                missing.append("head")
            if state.left_wrist_pose is None:
                missing.append("left wrist")
            if state.right_wrist_pose is None:
                missing.append("right wrist")
            raise SystemExit(
                f"Timed out waiting for VR headset ({timeout_s:.0f}s). "
                f"Missing: {', '.join(missing) or 'unknown'}. "
                "Open the printed https URL on the headset and tap 'Passthrough'."
            )
        if now - last_status >= 5.0:
            print("  still waiting for VR poses…", flush=True)
            last_status = now
        time.sleep(0.1)


def _create_vr_tracker(args: argparse.Namespace, control_dt: float):
    """Return (tcp_vel_tracker, vr_input_source, replay, replay_path, commanded_tracker)."""
    replay = None
    replay_path = None
    commanded_tracker = None
    vr_input_source = None

    if args.wasd:
        commanded_tracker = CommandedVRPoseTracker(control_dt=control_dt, smoothing_alpha=0.12)
        tcp_vel_tracker = TCPVelTrackerFromPose(commanded_tracker)
        vr_input_source = commanded_tracker
    elif args.vr_replay is not None:
        replay_path = (
            resolve_asset(args.vr_replay) if not os.path.isabs(args.vr_replay) else args.vr_replay
        )
        replay = VRReplayWrapper(replay_path)
        tcp_vel_tracker = TCPVelTrackerFromPose(replay)
        vr_input_source = replay
    else:
        from lib.deploy.vr.vuer_vr_pose_tracker import VuerVRPoseTracker

        vr_input_source = VuerVRPoseTracker(port=args.vr_port)
        tcp_vel_tracker = TCPVelTrackerFromPose(vr_input_source)
        print("Waiting for VR headset to connect (tap 'Passthrough' in the browser)...", flush=True)
        _wait_for_vr_connection(vr_input_source)
        print("VR connected.", flush=True)

    return tcp_vel_tracker, vr_input_source, replay, replay_path, commanded_tracker


def _create_controller(args, tcp_vel_tracker, *, using_mock, replay, replay_stem, verbose):
    controller = TeleopController(
        checkpoint=args.checkpoint,
        network=args.network,
        device=args.device,
        dex3=args.dex3,
        save_videos=args.save_videos,
        restart_localization=args.restart_localization,
        debug=args.debug,
        profile=args.profile,
        vel_v2=args.vel_v2,
        window=args.window,
    )
    controller.set_vr_pose_getter(lambda: tcp_vel_tracker.last_pose_state)

    if using_mock and isinstance(controller.env, G1MujocoEnv):
        if replay is not None:
            video_path = os.path.join(LOCAL_ROOT, "tmp", f"replay_{replay_stem}.mp4")
        else:
            video_path = os.path.join(LOCAL_ROOT, "tmp", f"sim_{int(time.time())}.mp4")
        controller.env.configure_single_video(video_path)
        if verbose:
            buffered = controller.env.img_buf.maxlen
            print_path_lines(
                "  video out:  ", video_path,
                note=f"last {buffered} frames, saved on exit",
            )
    return controller


def _create_input_reader(args, controller, using_mock):
    import sys

    max_abs = 1.0 if using_mock else 0.25
    if args.controller:
        remote_ctrl = controller.env.get_remote_controller()
        if using_mock:
            from lib.deploy.deploy_utils.shm_remote_controller import create_remote_controller_shm
            remote_ctrl = create_remote_controller_shm()['controller']
        return ControllerReader(
            damp_node_name=controller._damp_node.name,
            remote_controller=remote_ctrl,
            max_abs=max_abs,
        )

    sim_reset_fn = (lambda: controller.env.reset_sim()) if is_sync_sim_network(args.network) else None
    # Enable terminal keys (x damp · 0-9 FSM · WASD · r) whenever we own a TTY and
    # aren't driving input from the browser (--view). Works on the real robot too;
    # WASD there is capped by max_abs above. `--controller` already returned earlier.
    use_stdin = not args.view and sys.stdin.isatty()
    return InputReader(
        damp_node_name=controller._damp_node.name,
        max_abs=max_abs,
        sim_reset_fn=sim_reset_fn,
        use_stdin=use_stdin,
    )


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    using_mock = is_sync_sim_network(args.network) or args.network == 'sim'
    verbose = using_mock and not args.debug
    say = print if verbose else _noop

    if verbose:
        _print_sim_banner(args)

    if args.dex3 and is_mock_network(args.network):
        raise SystemExit("--dex3 requires a real robot network interface (not sim)")

    _validate_input_mode(args)

    control_dt = 0.04
    tcp_vel_tracker, vr_input_source, replay, replay_path, commanded_tracker = _create_vr_tracker(
        args, control_dt,
    )
    replay_stem = (
        os.path.splitext(os.path.basename(replay_path))[0] if replay_path is not None else None
    )

    # Prime the tracker
    tcp_vel_tracker.get_state()

    say("Loading model…", flush=True)
    controller = _create_controller(
        args, tcp_vel_tracker,
        using_mock=using_mock, replay=replay, replay_stem=replay_stem, verbose=verbose,
    )

    from lib.deploy.debug.logs import format_fsm_graph, _display_fsm_node
    print(format_fsm_graph(
        controller.fsm_graph(), _display_fsm_node(controller.current_node_name)) + "\n", flush=True)

    view = LiveView.start(
        controller,
        enabled=args.view, port=args.view_port,
        replay=replay, replay_stem=replay_stem,
        vr=args.vr, no_wait=args.no_wait, verbose=verbose, realtime=not args.fast,
    )

    kp_logger = None
    if args.debug:
        os.makedirs(os.path.join(LOCAL_ROOT, "tmp"), exist_ok=True)
        kp_path = os.path.join(LOCAL_ROOT, "tmp", f"vr_keypoints_{int(time.time())}.jsonl")
        kp_logger = _VrKeypointLogger(kp_path)
        print_path_lines("  vr keypoints: ", kp_path)

    inp = None
    replay_pbar = None
    replay_complete = False

    try:
        view.wait_ready()

        inp = _create_input_reader(args, controller, using_mock)
        inp.start()

        motion_enabled = replay is None and not args.vr
        view.attach_input(controller, inp, motion_enabled=motion_enabled)

        controller.env.command_source = inp.get_velocity

        if commanded_tracker is not None:
            commanded_tracker.set_commander(inp)
            # Integrate WASD over the env clock (sim time), so commanded motion is
            # independent of how fast/slow the sim runs vs wall clock.
            commanded_tracker.set_env(controller.env)

        say("Running… (Ctrl+C to stop)\n")

        if replay is not None:
            replay_pbar = tqdm(
                total=replay.len(),
                desc="VR replay",
                unit="frame",
                leave=True,
                disable=args.debug,
                mininterval=0.25,
                dynamic_ncols=True,
            )
            DeployConsole.set_terminal_writer(replay_pbar.write)

        # --- 4. Main loop ---
        while True:
            # Advance the replay only while teleoperating (DoTeleop active). Before that
            # the FSM is still damping/standing/interpolating and doesn't consume
            # the VR poses, so stepping would burn the front of the trajectory.
            if replay is not None and controller.is_teleoperating:
                replay.step()
                if replay_pbar is not None:
                    replay_pbar.n = min(replay.current_index, replay.len())
                    replay_pbar.refresh()

            vr_state = tcp_vel_tracker.get_state()

            if kp_logger is not None:
                kp_logger.log(tcp_vel_tracker.last_pose_state, controller.get_env_time())

            # Poll VR buttons (rising-edge)
            inp.poll_vr_buttons(vr_input_source)

            # FSM target: keyboard/VR > auto
            fsm_target_res = inp.consume_fsm_target()
            fsm_target = fsm_target_res.target
            if fsm_target is None and args.auto_next_node:
                fsm_target = controller.get_auto_next_target()

            # Dex3 gripper (trigger on right controller when available)
            right_gripper = None
            if args.dex3 and vr_input_source is not None:
                _, right_ctrl = vr_input_source.get_controller_states()
                if right_ctrl:
                    right_gripper = GripperAction(pos=int(right_ctrl.trigger_front * 100), mode='normal', teleop_button=True)

            controller.act(TeleopAction(
                vr_state=vr_state,
                fsm_target=fsm_target,
                right_gripper=right_gripper,
            ))


            controller.sleep(control_dt)

            if replay is not None and replay.finished:
                if replay_pbar is not None:
                    replay_pbar.n = replay.len()
                    replay_pbar.refresh()
                replay_complete = True
                view.update(controller, inp, status="finished")
                view.hold_finished(controller, inp)
                break

            view.update(controller, inp, status="running")

    except KeyboardInterrupt:
        if verbose:
            sim_time = controller.get_env_time()
            print(f"\nStopping sim ({sim_time:.1f}s sim time)…", flush=True)
    finally:
        DeployConsole.set_terminal_writer(None)
        if replay_pbar is not None:
            replay_pbar.close()
        if replay_complete and verbose:
            print("VR replay complete.", flush=True)
        if inp is not None:
            inp.stop()
        if kp_logger is not None:
            kp_logger.close()
        controller.stop()


if __name__ == '__main__':
    args = parse_args()
    main(args)
