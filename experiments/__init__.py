from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass
class ExperimentResult:
    score: float
    metrics: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class Experiment(ABC):
    def __init__(self, *args, **kwargs):
        self.setup_complete = False
        self.cleanup_complete = False
    
    def setup(self, *args, **kwargs):
        self.setup_complete = True
    
    @abstractmethod
    def run(self, *args, **kwargs) -> ExperimentResult:
        raise NotImplementedError()
    
    def cleanup(self):
        self.cleanup_complete = True
    
    def execute(self, *args, **kwargs) -> ExperimentResult:
        self.setup(*args, **kwargs)
        result = self.run(*args, **kwargs)
        self.cleanup()
        return result
