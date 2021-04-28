class NaiveCompressorConfig:
    def __init__(self):
        self._destination_block_size: int = 4
        self._source_block_size: int = 8
