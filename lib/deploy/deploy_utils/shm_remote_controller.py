import argparse
from multiprocessing import shared_memory
import struct
import threading
import time
import subprocess

# Default max chunk size (e.g., 64KB)
DEFAULT_MAX_CHUNK_SIZE = 256
MEM_NAME = 'remote_controller_shm'

class LatestChunkIPC:
    """Share latest chunk between processes"""

    def __init__(self, name: str, create: bool = False):
        self.name = name
        self.header_size = 8  # For storing actual data length

        if create:
            # Sender creates with max size
            self.total_size = self.header_size + DEFAULT_MAX_CHUNK_SIZE

            try:
                # Clean up if exists
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(
                create=True,
                size=self.total_size,
                name=name
            )
            # Initialize with zeros
            self.shm.buf[:] = b'\x00' * self.total_size
        else:
            # Receiver just attaches
            self.shm = shared_memory.SharedMemory(name=name)
            self.total_size = len(self.shm.buf)

    def write(self, data: bytes):
        """Write chunk (overwrites previous)"""
        max_data_size = self.total_size - self.header_size
        if len(data) > max_data_size:
            data = data[:max_data_size]

        # Write length header + data
        struct.pack_into('Q', self.shm.buf, 0, len(data))
        self.shm.buf[self.header_size:self.header_size + len(data)] = data

    def read(self) -> bytes:
        """Read current chunk"""
        # Read length header
        data_len = struct.unpack_from('Q', self.shm.buf, 0)[0]
        if data_len == 0:
            return b''

        # Read actual data
        return bytes(self.shm.buf[self.header_size:self.header_size + data_len])

    def close(self):
        """Close connection to shared memory"""
        self.shm.close()

    def cleanup(self):
        """Remove shared memory (call from creator process when done)"""
        self.shm.close()
        self.shm.unlink()


# Simple functional interface:

def create_sender(name: str):
    """Create a sender for variable-size chunks"""
    header_size = 8
    total_size = header_size + DEFAULT_MAX_CHUNK_SIZE

    try:
        # Clean up if exists
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

    shm = shared_memory.SharedMemory(create=True, size=total_size, name=name)
    shm.buf[:] = b'\x00' * total_size

    def send(data: bytes):
        if len(data) > DEFAULT_MAX_CHUNK_SIZE:
            data = data[:DEFAULT_MAX_CHUNK_SIZE]
        struct.pack_into('Q', shm.buf, 0, len(data))
        shm.buf[header_size:header_size + len(data)] = data

    send.close = shm.close
    send.cleanup = shm.unlink
    return send


def create_receiver(name: str):
    """Create a receiver - no size needed!"""
    shm = shared_memory.SharedMemory(name=name)
    header_size = 8

    def receive() -> bytes:
        data_len = struct.unpack_from('Q', shm.buf, 0)[0]
        if data_len == 0:
            return b''
        return bytes(shm.buf[header_size:header_size + data_len])

    receive.close = shm.close
    return receive


def create_remote_controller_sender_process(network: str = None):
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    ChannelFactoryInitialize(0, network)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    mem = LatestChunkIPC(MEM_NAME, create=True)
    def cb(msg: LowState_):
        controller_bytes = msg.wireless_remote
        time.sleep(0.01)
        mem.write(controller_bytes)
    sub.Init(cb, 1)


def create_remote_controller_shm():
    import os
    import sys

    script = os.path.join(os.path.dirname(__file__), "shm_remote_controller.py")
    shm_process = subprocess.Popen([
        sys.executable,
        script,
        "--send",
    ])
    print(f"Started SHM sender process with PID: {shm_process.pid}")
    time.sleep(1.)

    mem = LatestChunkIPC(MEM_NAME)
    from lib.deploy.control.remote_controller import RemoteController
    controller = RemoteController()

    # Thread control
    stop_event = threading.Event()

    def update_loop():
        """Continuously read from shared memory and update controller"""
        while not stop_event.is_set():
            try:
                data = mem.read()
                if data:  # Only update if there's data
                    controller.set(data)
            except Exception as e:
                print(f"Error updating controller: {e}")

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)  # 1ms, adjust based on your needs

    # Start update thread
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()

    # Return handles for cleanup
    return {
        'mem': mem,
        'controller': controller,
        'stop': lambda: stop_event.set(),
        'wait': lambda: update_thread.join(),
        'thread': update_thread,
        'shm_process': shm_process,
    }



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network',
                        default=None,
                        type=str, help='network interface.')

    parser.add_argument('--send',
                        action='store_true')

    parser.add_argument('--read',
                        action='store_true')

    return parser.parse_args()

# Usage example:
if __name__ == "__main__":
    args = parse_args()
    assert args.send or args.read
    assert not (args.send and args.read)

    if args.send:
        sender = create_remote_controller_sender_process(args.network)
        while True:
            time.sleep(0.1)

    if args.read:
        from lib.deploy.control.remote_controller import RemoteController, Joystick
        remote_controller_shm: RemoteController = create_remote_controller_shm()['controller']
        while True:
            print(remote_controller_shm.button)
            print(remote_controller_shm.get_joystick(Joystick.left))
            print(remote_controller_shm.get_joystick(Joystick.right))
            time.sleep(0.1)
