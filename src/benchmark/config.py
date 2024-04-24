from dataclasses import dataclass

@dataclass
class NeuralNetConfig:
    epochs: int = 4
    batch_size: int = 4096
    hidden_size: int = 64
    dropout_prob: float = 0.5
    shuffle: bool = True
    num_workers: int = 2
    test_pct: float = 0.2
    val_pct: float = 0.2
    input_size: int = 20
    output_size: int = 61