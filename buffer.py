import numpy as np

class Buffer:
    def __init__(self, dtype=np.float32, initial_size=65536):
        self.buffer = np.zeros(initial_size, dtype=dtype)
        self.start = 0
        self.end = 0

    def extend(self, data):
        if self.end - self.start + data.shape[0] > self.buffer.shape[0]:
            self.pop_slice(0, self.end)
        self.buffer[self.end-self.start:self.end-self.start+data.shape[0]] = data
        self.end += data.shape[0]

    def length(self):
        return self.end - self.start

    def pop_slice(self, start, end):
        xs = self.buffer[start-self.start:end-self.start]
        zeros = np.zeros(self.buffer.shape[0])
        zeros[0:self.buffer.shape[0]-end+self.start] = self.buffer[end-self.start:]
        self.buffer = zeros
        self.start = end
        return xs

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start is not None else self.start
            stop = key.stop if key.stop is not None else self.end
            return self.buffer[start - self.start:stop - self.start]
        return self.buffer[key - self.start]

    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self), indent=4, width=1)

#b = Buffer()
#
#b.extend(np.array([1., 3., .4]))
#b.extend(np.array([1., 3., .4]))
#b.extend(np.array([1., 3., .4]))
#
#print(b.buffer[0:10])
#print(b)
#
#print("pop", b.pop_slice(0,4))
#
#print(b.buffer[0:10])
#print(b)
