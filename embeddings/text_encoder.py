"""Text encoder for API-based embeddings (placeholder for future use)."""

from __future__ import annotations

import numpy as np


class TextEncoder:
    """Stub for text-embedding-3-small integration.
    
    This is a placeholder for future text-based embeddings.
    Currently returns zero vectors to maintain API compatibility.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        """Initialize text encoder.
        
        Args:
            dimensions: Embedding dimension (1536 for text-embedding-3-small)
        """
        self.dimensions = dimensions

    def encode(self, text: str) -> np.ndarray:
        """Encode text into embedding vector.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector (currently returns zero vector as placeholder)
        """
        # Placeholder: return zero vector
        return np.zeros(self.dimensions, dtype=np.float32)
