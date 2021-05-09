class CompressorConfig:
    def __init__(self):
        self.min_destination_block_size: int = 2
        self.source_block_start_size: int = 32
        self.source_block_levels: int = 4
        self.source_block_step: float = 0.2
        self.loss_tolerance: float = 0.001
