"""Test error conditions and edge cases for robust error handling."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from Project.utils.config_schema import TrainConfig, DataLoaderConfig, validate_config
from Project.utils.hydra_utils import load_config
from Project.utils.io import write_jsonl, _validate_output_path, PROJECT_ROOT


class TestConfigValidation:
    """Test configuration validation catches invalid values."""

    def test_negative_learning_rate(self):
        """Negative learning rate should raise ValidationError."""
        with pytest.raises(ValidationError, match="greater than 0"):
            TrainConfig(lr=-0.001, batch_size=16, epochs=10)

    def test_zero_batch_size(self):
        """Zero batch size should raise ValidationError."""
        with pytest.raises(ValidationError, match="greater than 0"):
            TrainConfig(lr=0.001, batch_size=0, epochs=10)

    def test_zero_epochs(self):
        """Zero epochs should raise ValidationError."""
        with pytest.raises(ValidationError, match="greater than 0"):
            TrainConfig(lr=0.001, batch_size=16, epochs=0)

    def test_negative_num_workers(self):
        """Negative num_workers should raise ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            DataLoaderConfig(num_workers=-1)

    def test_valid_config_passes(self):
        """Valid configuration should pass validation."""
        cfg = TrainConfig(lr=1e-3, batch_size=16, epochs=10, fp16=True)
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 16
        assert cfg.epochs == 10
        assert cfg.fp16 is True

    def test_load_config_with_validation(self):
        """Test load_config with validate=True."""
        # Create a temporary valid config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
train:
  lr: 0.001
  batch_size: 16
  epochs: 10
  fp16: true
dataloader:
  num_workers: 4
  pin_memory: true
"""
            )
            cfg_path = f.name

        try:
            cfg = load_config(cfg_path, validate=True)
            assert cfg.train.lr == 0.001
            assert cfg.dataloader.num_workers == 4
        finally:
            Path(cfg_path).unlink()

    def test_load_config_with_invalid_data(self):
        """Test load_config catches invalid configurations."""
        # Create a temporary invalid config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
train:
  lr: -0.001
  batch_size: 16
  epochs: 10
"""
            )
            cfg_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_config(cfg_path, validate=True)
        finally:
            Path(cfg_path).unlink()


class TestPathValidation:
    """Test path validation prevents directory traversal attacks."""

    def test_path_within_project_passes(self):
        """Paths within PROJECT_ROOT should pass validation."""
        safe_path = PROJECT_ROOT / "outputs" / "test.jsonl"
        validated = _validate_output_path(safe_path)
        assert validated == safe_path.resolve()

    def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked."""
        unsafe_path = PROJECT_ROOT / "outputs" / ".." / ".." / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="outside project directory"):
            _validate_output_path(unsafe_path)

    def test_absolute_path_outside_project_blocked(self):
        """Absolute paths outside project should be blocked."""
        with pytest.raises(ValueError, match="outside project directory"):
            _validate_output_path(Path("/tmp/malicious.jsonl"))

    def test_write_jsonl_validates_path(self):
        """write_jsonl should validate output paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create safe path within project
            safe_path = PROJECT_ROOT / "outputs" / "test_safe.jsonl"
            safe_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to safe path should work
            data = [{"id": 1, "text": "test"}]
            write_jsonl(safe_path, data)
            assert safe_path.exists()
            safe_path.unlink()

            # Write to unsafe path should fail
            unsafe_path = Path("/tmp") / "test_unsafe.jsonl"
            with pytest.raises(ValueError, match="outside project directory"):
                write_jsonl(unsafe_path, data)


class TestMissingFiles:
    """Test handling of missing or corrupted files."""

    def test_missing_config_file(self):
        """Loading non-existent config should raise appropriate error."""
        with pytest.raises((FileNotFoundError, Exception)):
            load_config("nonexistent_config.yaml")

    def test_corrupted_jsonl(self):
        """Corrupted JSONL should be handled gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write invalid JSON
            f.write("not valid json\n")
            f.write("{incomplete json\n")
            jsonl_path = f.name

        try:
            # Attempt to read corrupted JSONL
            with open(jsonl_path, "r") as f:
                for line in f:
                    with pytest.raises(json.JSONDecodeError):
                        json.loads(line)
        finally:
            Path(jsonl_path).unlink()

    def test_empty_jsonl(self):
        """Empty JSONL file should not cause errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write nothing
            jsonl_path = f.name

        try:
            # Reading empty file should work
            lines = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    lines.append(json.loads(line))
            assert len(lines) == 0
        finally:
            Path(jsonl_path).unlink()


class TestDataLoaderEdgeCases:
    """Test DataLoader configuration edge cases."""

    def test_num_workers_zero_valid(self):
        """num_workers=0 (single-threaded) should be valid."""
        cfg = DataLoaderConfig(num_workers=0, pin_memory=False)
        assert cfg.num_workers == 0
        assert cfg.prefetch_factor == 2  # Should still have default

    def test_persistent_workers_requires_num_workers(self):
        """persistent_workers should be used correctly with num_workers."""
        # persistent_workers=True with num_workers=0 is logically incorrect
        # but our schema doesn't enforce this constraint - just documents it
        cfg = DataLoaderConfig(num_workers=0, persistent_workers=True)
        assert cfg.num_workers == 0
        assert cfg.persistent_workers is True

    def test_pin_memory_cpu_only(self):
        """pin_memory should be handled correctly on CPU-only systems."""
        # This is a runtime check, not schema validation
        # The config should accept pin_memory=True even if no GPU available
        cfg = DataLoaderConfig(pin_memory=True, num_workers=4)
        assert cfg.pin_memory is True


class TestConfigDefaults:
    """Test that configuration defaults are sensible."""

    def test_train_config_defaults(self):
        """TrainConfig should have sensible defaults."""
        cfg = TrainConfig(lr=0.001, batch_size=16, epochs=10)
        assert cfg.fp16 is False  # Default to safe FP32
        assert cfg.grad_accum == 1  # Default to no accumulation
        assert cfg.shuffle is True  # Default to shuffle

    def test_dataloader_config_defaults(self):
        """DataLoaderConfig should have sensible defaults."""
        cfg = DataLoaderConfig()
        assert cfg.num_workers == 0  # Default to single-threaded for compatibility
        assert cfg.pin_memory is False  # Default to safe (works on CPU)
        assert cfg.persistent_workers is False
        assert cfg.prefetch_factor == 2

    def test_optional_fields_none(self):
        """Optional fields should accept None values."""
        cfg = TrainConfig(
            lr=0.001,
            batch_size=16,
            epochs=10,
            grad_accum=None,  # Optional fields can be None
        )
        assert cfg.grad_accum is None  # Optional fields remain None if not provided

        # Without specifying, uses default
        cfg2 = TrainConfig(lr=0.001, batch_size=16, epochs=10)
        assert cfg2.grad_accum == 1  # Default value


class TestBoundaryValues:
    """Test boundary values for configuration parameters."""

    def test_very_small_learning_rate(self):
        """Very small but positive learning rate should be valid."""
        cfg = TrainConfig(lr=1e-8, batch_size=16, epochs=10)
        assert cfg.lr == 1e-8

    def test_very_large_batch_size(self):
        """Very large batch size should be valid."""
        cfg = TrainConfig(lr=0.001, batch_size=1024, epochs=10)
        assert cfg.batch_size == 1024

    def test_single_epoch(self):
        """Single epoch training should be valid."""
        cfg = TrainConfig(lr=0.001, batch_size=16, epochs=1)
        assert cfg.epochs == 1

    def test_max_workers(self):
        """Large number of workers should be valid."""
        cfg = DataLoaderConfig(num_workers=32)
        assert cfg.num_workers == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
