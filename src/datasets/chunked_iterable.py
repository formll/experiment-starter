class ChunkedIterable:
    def __init__(self, base_iterable, chunk_size):
        self.chunk_size = chunk_size
        self.base_iterable = base_iterable
        self.iterator = iter(range(0))  # initializing as empty iterator to defer the possibly lengthy iter(self.base) call

    def __iter__(self):
        c = 0
        while c < self.chunk_size:
            try:
                yield next(self.iterator)
                c += 1
            except StopIteration:
                self.iterator = iter(self.base_iterable)

    def __len__(self):
        return self.chunk_size