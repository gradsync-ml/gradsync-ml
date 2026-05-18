"""
Tensor compression and decompression utilities for GRADSYNC.
Implements various compression algorithms to reduce network transmission overhead.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, Dict, Any
import logging
import zlib
import pickle
import struct

from enum import Enum

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Available compression types for tensor serialization."""
    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    BINARY = "binary"
    SPARSE = "sparse"


class TensorCompressor:
    """
    Tensor compression utility for reducing network transmission overhead.
    Supports multiple compression algorithms with automatic fallback.
    """

    def __init__(self, default_compression: CompressionType = CompressionType.NONE):
        """Initialize tensor compressor with default compression type."""
        self.default_compression = default_compression
        self.compression_stats = {
            'compressed_bytes': 0,
            'original_bytes': 0,
            'compression_ratio': 0.0
        }

    def compress(
        self,
        tensor_bytes: bytes,
        compression_type: Optional[CompressionType] = None,
        compression_level: int = 6
    ) -> bytes:
        """
        Compress tensor bytes using specified compression algorithm.

        Args:
            tensor_bytes: Raw tensor bytes to compress
            compression_type: Compression algorithm to use
            compression_level: Compression level (0-9, higher = more compression)

        Returns:
            Compressed tensor bytes
        """
        compression_type = compression_type or self.default_compression
        self.compression_stats['original_bytes'] += len(tensor_bytes)

        try:
            if compression_type == CompressionType.NONE:
                return tensor_bytes

            elif compression_type == CompressionType.FP16:
                return self._compress_fp16(tensor_bytes)

            elif compression_type == CompressionType.INT8:
                return self._compress_int8(tensor_bytes, compression_level)

            elif compression_type == CompressionType.BINARY:
                return self._compress_binary(tensor_bytes)

            elif compression_type == CompressionType.SPARSE:
                return self._compress_sparse(tensor_bytes, compression_level)

            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")

        except Exception as e:
            logger.warning(f"Compression failed with {compression_type}, falling back to none: {e}")
            return tensor_bytes

        return tensor_bytes

    def decompress(
        self,
        compressed_bytes: bytes,
        compression_type: str,
        original_shape: Optional[Tuple[int, ...]] = None
    ) -> bytes:
        """
        Decompress tensor bytes using specified compression algorithm.

        Args:
            compressed_bytes: Compressed tensor bytes
            compression_type: Compression algorithm used (string)
            original_shape: Original tensor shape (needed for some algorithms)

        Returns:
            Decompressed tensor bytes
        """
        try:
            compression_type = CompressionType(compression_type)

            if compression_type == CompressionType.NONE:
                return compressed_bytes

            elif compression_type == CompressionType.FP16:
                return self._decompress_fp16(compressed_bytes)

            elif compression_type == CompressionType.INT8:
                return self._decompress_int8(compressed_bytes, original_shape)

            elif compression_type == CompressionType.BINARY:
                return self._decompress_binary(compressed_bytes)

            elif compression_type == CompressionType.SPARSE:
                return self._decompress_sparse(compressed_bytes, original_shape)

            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise

        return compressed_bytes

    def _compress_fp16(self, tensor_bytes: bytes) -> bytes:
        """Compress using fp16 quantization."""
        # Convert bytes to numpy array of float32
        tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32)

        # Quantize to float16
        tensor_fp16 = tensor_np.astype(np.float16)

        # Include header with compression info
        header = struct.pack('!II', len(tensor_np), len(tensor_fp16) * 2)

        return header + tensor_fp16.tobytes()

    def _decompress_fp16(self, compressed_bytes: bytes) -> bytes:
        """Decompress fp16 compressed data."""
        # Extract header
        header_size = struct.calcsize('!II')
        original_size, compressed_size = struct.unpack('!II', compressed_bytes[:header_size])

        # Convert compressed data back to float32
        tensor_fp16 = np.frombuffer(compressed_bytes[header_size:], dtype=np.float16)
        tensor_fp32 = tensor_fp16.astype(np.float32)

        return tensor_fp32.tobytes()

    def _compress_int8(self, tensor_bytes: bytes, level: int = 6) -> bytes:
        """Compress using int8 quantization and optional zlib."""
        tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32)

        # Compute dynamic range
        tensor_min = tensor_np.min()
        tensor_max = tensor_np.max()
        tensor_range = tensor_max - tensor_min

        # Quantize to int8 range
        if tensor_range > 0:
            tensor_int8 = ((tensor_np - tensor_min) / tensor_range * 255 - 128).astype(np.int8)
        else:
            tensor_int8 = np.zeros(len(tensor_np), dtype=np.int8)

        # Optional zlib compression for int8 data
        if level > 0:
            tensor_int8 = zlib.compress(tensor_int8.tobytes(), level)

        # Package with metadata
        header = struct.pack('!IIfff', len(tensor_np), len(tensor_int8),
                             tensor_min, tensor_max, tensor_range)
        return header + tensor_int8

    def _decompress_int8(self, compressed_bytes: bytes, original_shape: Optional[Tuple] = None) -> bytes:
        """Decompress int8 compressed data."""
        header_size = struct.calcsize('!IIfff')
        original_length, compressed_length, tensor_min, tensor_max, tensor_range = \
            struct.unpack('!IIfff', compressed_bytes[:header_size])

        tensor_int8_bytes = compressed_bytes[header_size:]

        # If original length != compressed length, it was zlib compressed
        if original_length != compressed_length:
            tensor_int8_bytes = zlib.decompress(tensor_int8_bytes)

        tensor_int8 = np.frombuffer(tensor_int8_bytes, dtype=np.int8)

        # Dequantize back to float32
        tensor_fp32 = (tensor_int8.astype(np.float32) + 128) / 255 * tensor_range + tensor_min

        return tensor_fp32.tobytes()

    def _compress_binary(self, tensor_bytes: bytes) -> bytes:
        """Compress using binary quantization to +/-1."""
        tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32)

        # Binary quantization
        binary_values = (tensor_np > 0).astype(np.int8)

        # Store as bit-packed array for better compression
        packed_bits = np.packbits(binary_values)

        # Include metadata for reconstruction
        tensor_abs_mean = np.abs(tensor_np).mean()
        header = struct.pack('!If', len(tensor_np), tensor_abs_mean)

        return header + packed_bits.tobytes()

    def _decompress_binary(self, compressed_bytes: bytes) -> bytes:
        """Decompress binary quantized data."""
        header_size = struct.calcsize('!If')
        original_length, tensor_abs_mean = struct.unpack('!If', compressed_bytes[:header_size])

        packed_bits = compressed_bytes[header_size:]
        binary_values = np.unpackbits(np.frombuffer(packed_bits, dtype=np.uint8))[:original_length]

        # Restore to +/-tensor_abs_mean
        tensor_fp32 = np.where(binary_values, tensor_abs_mean, -tensor_abs_mean).astype(np.float32)

        return tensor_fp32.tobytes()

    def _compress_sparse(self, tensor_bytes: bytes, threshold: float = 0.01) -> bytes:
        """Compress using sparse representation of small values."""
        tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32)

        # Find significant values (above threshold)
        abs_tensor = np.abs(tensor_np)
        max_val = abs_tensor.max() if abs_tensor.max() > 0 else 1.0
        threshold_value = threshold * max_val

        significant_mask = abs_tensor > threshold_value
        significant_indices = np.where(significant_mask)[0]
        significant_values = tensor_np[significant_mask]

        # Store sparse representation
        if len(significant_values) < len(tensor_np) * 0.8:
            header = struct.pack('!III', len(tensor_np), len(significant_values), threshold_value)
            sparse_data = header + \
                significant_indices.astype(np.uint32).tobytes() + \
                significant_values.tobytes()

            # Apply additional zlib compression to sparse data
            return zlib.compress(sparse_data, 6)
        else:
            # Not sparse enough, fallback to simple compression
            return self._compress_int8(tensor_bytes)

    def _decompress_sparse(self, compressed_bytes: bytes, original_shape: Optional[Tuple] = None) -> bytes:
        """Decompress sparse compressed data."""
        sparse_data = zlib.decompress(compressed_bytes)

        header_size = struct.calcsize('!III')
        original_length, num_significant, threshold_value = \
            struct.unpack('!III', sparse_data[:header_size])

        indices_size = num_significant * 4 # uint32
        indices = np.frombuffer(sparse_data[header_size:header_size+indices_size], dtype=np.uint32)
        values = np.frombuffer(sparse_data[header_size+indices_size:], dtype=np.float32)

        # Reconstruct original array
        tensor_fp32 = np.zeros(original_length, dtype=np.float32)
        tensor_fp32[indices] = values

        return tensor_fp32.tobytes()

    def compute_compression_ratio(self) -> float:
        """Compute overall compression ratio across all operations."""
        if self.compression_stats['original_bytes'] > 0:
            ratio = self.compression_stats['compressed_bytes'] / self.compression_stats['original_bytes']
            self.compression_stats['compression_ratio'] = ratio * 100.0
        return self.compression_stats['compression_ratio']

    def reset_stats(self):
        """Reset compression statistics."""
        self.compression_stats = {
            'compressed_bytes': 0,
            'original_bytes': 0,
            'compression_ratio': 0.0
        }


def get_optimal_compression(tensor: torch.Tensor) -> CompressionType:
    """
    Determine optimal compression algorithm based on tensor characteristics.

    Args:
        tensor: Input tensor to analyze

    Returns:
        Recommended compression type
    """
    tensor_np = tensor.detach().cpu().numpy()

    # Check sparsity
    zero_ratio = np.sum(np.abs(tensor_np) < 1e-6) / tensor_np.size
    if zero_ratio > 0.3: # More than 30% sparse
        return CompressionType.SPARSE

    # Check dynamic range
    tensor_min, tensor_max = tensor_np.min(), tensor_np.max()
    dynamic_range = (tensor_max - tensor_min) / (tensor_np.std() + 1e-6)

    if dynamic_range > 100: # Very high dynamic range
        return CompressionType.INT8

    # Check value distribution
    unique_vals = len(np.unique(tensor_np))
    if unique_vals < tensor_np.size * 0.1: # Highly quantized values
        return CompressionType.BINARY

    # Default fallback based on precision needs
    return CompressionType.FP16


def validate_tensor_compression(
    tensor: torch.Tensor,
    compression_type: CompressionType,
    threshold: float = 0.01
) -> bool:
    """
    Validate whether compression is beneficial for given tensor.

    Args:
        tensor: Input tensor
        compression_type: Compression algorithm to test
        threshold: Acceptable loss threshold

    Returns:
        True if compression is beneficial
    """
    import copy

    try:
        # Convert tensor to bytes
        tensor_bytes = tensor.detach().cpu().float().numpy().tobytes()

        # Compress and decompress
        compressor = TensorCompressor(compression_type)
        compressed = compressor.compress(tensor_bytes)
        decompressed = compressor.decompress(compressed, compression_type.value)

        # Convert back to tensor
        restored_tensor = torch.frombuffer(
            np.frombuffer(decompressed, dtype=np.float32),
            dtype=torch.float32
        ).reshape(tensor.shape)

        # Compute reconstruction error
        reconstruction_error = torch.abs(tensor.cpu() - restored_tensor).mean().item()
        relative_error = reconstruction_error / (tensor.abs().mean().item() + 1e-6)

        # Check if error is acceptable
        is_acceptable = relative_error < threshold

        # Check compression ratio
        compression_ratio = len(compressed) / len(tensor_bytes)
        is_beneficial = compression_ratio < 0.8 # Less than 80% of original size

        return is_acceptable and is_beneficial

    except Exception as e:
        logger.error(f"Compression validation failed: {e}")
        return False


# Export main class and utilities
__all__ = [
    'TensorCompressor',
    'CompressionType',
    'get_optimal_compression',
    'validate_tensor_compression'
]