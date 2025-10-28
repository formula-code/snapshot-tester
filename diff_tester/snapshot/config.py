"""
Configuration management for snapshot testing.

This module handles loading and managing configuration settings
for the snapshot testing tool.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SnapshotConfig:
    """Configuration for snapshot testing."""
    
    # Directories
    benchmark_dir: str = "benchmarks/"
    snapshot_dir: str = ".snapshots/"
    project_dir: Optional[str] = None
    
    # Comparison settings
    tolerance: Dict[str, float] = None
    
    # Filtering
    exclude_benchmarks: list = None
    
    # Tracing settings
    trace_depth_limit: int = 100
    
    # Output settings
    verbose: bool = False
    quiet: bool = False
    
    def __post_init__(self):
        if self.tolerance is None:
            self.tolerance = {
                "rtol": 1e-5,
                "atol": 1e-8,
                "equal_nan": False
            }
        
        if self.exclude_benchmarks is None:
            self.exclude_benchmarks = ["timeraw_*"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnapshotConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'SnapshotConfig':
        """Load configuration from JSON file."""
        if not config_path.exists():
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            return cls()
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_benchmark_dir(self) -> Path:
        """Get benchmark directory as Path."""
        return Path(self.benchmark_dir)
    
    def get_snapshot_dir(self) -> Path:
        """Get snapshot directory as Path."""
        return Path(self.snapshot_dir)
    
    def get_project_dir(self) -> Optional[Path]:
        """Get project directory as Path."""
        if self.project_dir:
            return Path(self.project_dir)
        return None
    
    def should_exclude_benchmark(self, benchmark_name: str) -> bool:
        """Check if a benchmark should be excluded."""
        for pattern in self.exclude_benchmarks:
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                if benchmark_name.startswith(prefix):
                    return True
            elif pattern == benchmark_name:
                return True
        return False


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("snapshot_config.json")
        self.config = SnapshotConfig.from_file(self.config_path)
    
    def get_config(self) -> SnapshotConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def save_config(self) -> None:
        """Save configuration to file."""
        self.config.save_to_file(self.config_path)
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = SnapshotConfig.from_file(self.config_path)
    
    def create_default_config(self) -> None:
        """Create a default configuration file."""
        default_config = SnapshotConfig()
        default_config.save_to_file(self.config_path)
        print(f"Created default configuration at {self.config_path}")
