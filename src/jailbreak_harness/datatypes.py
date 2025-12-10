import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass(frozen=True)
class Variant:
    """
    Represents a single input variation within a TestCase.
    """
    # --- NON-DEFAULT FIELDS (Mandatory fields must come first) ---
    prompt: str = field(metadata={'help': 'The main user prompt for the model.'})
    
    # --- DEFAULTED FIELDS (Must follow non-default fields) ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()), metadata={'help': 'Unique identifier for the variant.'})
    system_note: Optional[str] = field(default=None, metadata={'help': 'System-level instruction or context for the model.'})
    temperature: Optional[float] = field(default=None, metadata={'help': 'Sampling temperature for the model. None indicates a sweepable setting.'})
    
    # Fields to store results after execution (init=False)
    result: Dict[str, Any] = field(default_factory=dict, init=False, metadata={'help': 'Storage for the final test result dictionary.'})
    meta: Dict[str, Any] = field(default_factory=dict, init=False, metadata={'help': 'Storage for caller-specific metadata.'})

    def __post_init__(self):
        """Perform validation after initialization."""
        if not self.id:
            # Note: This is mostly for safety if default_factory fails, but good practice.
            raise ValueError("Variant id must be a non-empty string.")
        if self.temperature is not None and not isinstance(self.temperature, (int, float)):
             raise ValueError(f"Temperature must be numeric (int or float) or None. Found: {type(self.temperature)}")


@dataclass(frozen=True)
class TestCase:
    """
    Represents a full test case, containing one or more input variants.
    """
    # --- NON-DEFAULT FIELDS (Mandatory fields must come first) ---
    name: str = field(metadata={'help': 'Human-readable name for the test case.'})
    description: str = field(metadata={'help': 'Detailed description of the test case objective.'})
    variants: List[Variant] = field(metadata={'help': 'List of input variations for this test case.'})
    
    # --- DEFAULTED FIELDS ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()), metadata={'help': 'Unique identifier for the test case.'})
    tags: List[str] = field(default_factory=list, metadata={'help': 'Categorization tags for the test case (e.g., "PII", "Do-Anything").'})
    expected_failure: bool = field(default=False, metadata={'help': 'If True, this is a test case the model is expected to fail (i.e., a known jailbreak).'})

    def __post_init__(self):
        """Perform validation after initialization."""
        if not self.variants:
            raise ValueError("TestCase must contain at least one variant.")
