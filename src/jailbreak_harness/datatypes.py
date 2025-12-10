# jailbreak_harness/datatypes.py

from dataclasses import dataclass, field
from typing import List, Optional
import logging

# Set up logging for validation warnings
logger = logging.getLogger(__name__)


# ---------------- Data Classes ----------------

@dataclass
class Variant:
    """Represents a single attempt/prompt variant within a TestCase."""
    id: str
    prompt: str
    system_note: Optional[str] = None
    temperature: Optional[float] = None
    
    # Optional fields for execution results (added for completeness, assuming they are used later)
    result: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


    def __post_init__(self):
        """Validate variant data after initialization."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Variant id must be a non-empty string, got: {self.id}")
        if not self.prompt or not isinstance(self.prompt, str):
            raise ValueError(f"Variant prompt must be a non-empty string")
        
        # Ensure temperature is either None or validated
        if self.temperature is not None:
            if not isinstance(self.temperature, (int, float)):
                raise ValueError(
                    f"Temperature must be numeric, got: {type(self.temperature)}"
                )
            if not 0.0 <= self.temperature <= 2.0:
                logger.warning(
                    f"Temperature {self.temperature} outside typical range [0.0, 2.0] for Variant ID: {self.id}"
                )


@dataclass
class TestCase:
    """A full test case containing multiple variants (e.g., different prompts)."""
    id: str
    name: str
    description: str
    variants: List[Variant]
    
    # Optional fields for test case metadata
    tags: List[str] = field(default_factory=list)
    expected_failure: bool = False

    def __post_init__(self):
        """Validate test case data after initialization."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"TestCase id must be a non-empty string, got: {self.id}")
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"TestCase name must be a non-empty string")
        if not self.variants or not isinstance(self.variants, list):
            raise ValueError(f"TestCase must have at least one variant")
        if not all(isinstance(v, Variant) for v in self.variants):
            # Log the specific issue instead of just raising a generic error
            invalid_variants = [type(v).__name__ for v in self.variants if not isinstance(v, Variant)]
            raise ValueError(
                f"All variants must be Variant instances. Found non-Variant types: {invalid_variants}"
            )
