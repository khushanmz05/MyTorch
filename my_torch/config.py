"""Configuration parsing for neural-network architecture files."""

from dataclasses import dataclass, field


@dataclass
class NetworkConfig:
    input_size: int = 781
    hidden_sizes: list = field(default_factory=lambda: [256, 128])
    output_size: int = 5
    hidden_activation: str = "relu"
    output_activation: str = "softmax"
    dropout: float = 0.0
    seed: int = 42

    @classmethod
    def from_file(cls, path):
        cfg = cls()
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                key, value = (part.strip() for part in line.split(":", 1))
                if key == "input_size":
                    cfg.input_size = int(value)
                elif key == "hidden_sizes":
                    cfg.hidden_sizes = [int(x) for x in value.replace(" ", "").split(",") if x]
                elif key == "output_size":
                    cfg.output_size = int(value)
                elif key == "hidden_activation":
                    cfg.hidden_activation = value
                elif key == "output_activation":
                    cfg.output_activation = value
                elif key == "dropout":
                    cfg.dropout = float(value)
                elif key == "seed":
                    cfg.seed = int(value)
                else:
                    raise ValueError(f"Unknown configuration key: {key}")
        return cfg

    def layer_sizes(self):
        return [self.input_size, *self.hidden_sizes, self.output_size]
