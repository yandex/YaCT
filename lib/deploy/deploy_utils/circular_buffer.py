import torch

class CircularBuffer:
    def __init__(self, max_len: int, memory_size: int = 2048):
        self.max_len = max_len
        self.memory_size = memory_size
        self._buffer = None
        self.idx = 0

    def append(self, t: torch.Tensor, non_blocking: bool = False):
        if self._buffer is None:
            self._buffer = torch.empty((self.memory_size, *t.shape), device=t.device, dtype=t.dtype)
            self._buffer[:self.max_len] = t
            self.idx = self.max_len - 1
        else:
            self.idx = (self.idx + 1) % self.memory_size
            if self.idx == 0:
                if self.max_len > 1:
                    self._buffer[:self.max_len - 1].copy_(self._buffer[-(self.max_len - 1):].clone())
                self.idx = self.max_len - 1
                self._buffer[self.idx].copy_(t, non_blocking=non_blocking)
            else:
                self._buffer[self.idx].copy_(t, non_blocking=non_blocking)
    @property
    def buffer(self):
        if self._buffer is None:
            raise ValueError("Buffer was not inited")

        if self.idx < self.max_len - 1:
            raise ValueError("There is a bug in buffer")
        else:
            # Buffer is full, get the last max_len elements
            start_idx = self.idx - (self.max_len - 1)
            end_idx = (self.idx + 1)

            assert start_idx >= 0 and end_idx <= self.memory_size
            return self._buffer[start_idx:end_idx]

    def reset(self):
        self._buffer = None
        self.idx = 0
