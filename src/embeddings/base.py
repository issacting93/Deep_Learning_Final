from abc import ABC, abstractmethod
from pathlib import Path


class EmbeddingGenerator(ABC):
    """Base class for all embedding generators."""

    @abstractmethod
    def generate(
        self,
        track_ids: list,
        output_dir: Path,
        batch_size: int = 32,
        resume: bool = True,
    ) -> tuple:
        """Generate embeddings for given track IDs.

        Returns:
            (embeddings: np.ndarray, valid_ids: list)
        """
        ...

    @abstractmethod
    def load_embeddings(self, output_dir: Path) -> tuple:
        """Load previously saved embeddings.

        Returns:
            (embeddings: np.ndarray, track_ids: list)
        """
        ...
