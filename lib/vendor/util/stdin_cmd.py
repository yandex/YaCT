import os
import sys
import tty
import time
import termios
import threading

from select import select
from signal import SIGINT
from queue import Queue, Empty
from typing import Union, Callable


def get_char(timeout=1):
    """
    Get single char from stdin with timeout
    Return None if no input and timeout
    """        
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    ch = None
    try:
        tty.setcbreak(sys.stdin.fileno())
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
    


class StdinCommands:
    """
    Read stdin and invokes commands based on input chars
    """

    def __init__(self):
        self._finish_signal = Queue()        
        self._commands = {}

    def add_command(self, key: Union[str, int], callback: Callable):
        """
        Args:
            key: input char, str for chars or int for key_id (ord(key))
            callback: command for invoke when key are pushed
        """
        self._commands[key] = callback

    def start(self):
        self._thread = threading.Thread(target=StdinCommands._read_worker, args=(self._commands, self._finish_signal))
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._finish_signal.put('stop')
        self._thread.join(timeout=1.0)

    @staticmethod
    def _read_worker(commands, finish_queue):
        while True:
            ch = get_char()
            if ch is not None:
                if ch in commands.keys():
                    r = commands[ch]()
                    if r == -1:
                        break

                elif ch.lower() in commands.keys():
                    r = commands[ch.lower()]()
                    if r == -1:
                        break

                elif ord(ch) in commands.keys():
                    r = commands[ord(ch)]()
                    if r == -1:
                        break

            try:
                finish_queue.get_nowait()
                break
            except Empty:
                pass
            
_LOCK = threading.Lock()
_STDINCOMMANDS = None

def get_stdincommands_singleton():
    global _STDINCOMMANDS, _LOCK
    with _LOCK:
        if _STDINCOMMANDS is None:
            _STDINCOMMANDS = StdinCommands()
        return _STDINCOMMANDS


if __name__ == '__main__':

    cmnds = StdinCommands()
    cmnds.add_command('r', lambda: print('reset'))
    cmnds.start()
    
    for i in range(10):
        time.sleep(1)
        print(i)

    cmnds.stop()
