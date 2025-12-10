# src/jailbreak_harness/__init__.py
"""LLM Jailbreak Testing Harness - Security research tool for testing LLM safety."""

__version__ = "0.2.1"


from .callers import (
    BaseModelCaller,
    MockModelCaller,
    OpenAIModelCaller,
    HFModelCaller,
)


from .datatypes import (
    TestCase,
    Variant,
)


from .harness import (
    JailbreakHarness,
    load_test_suite_from_file,
)


__all__ = [
    "JailbreakHarness",
    "BaseModelCaller",
    "MockModelCaller",
    "OpenAIModelCaller",
    "HFModelCaller",
    "TestCase",
    "Variant",
    "load_test_suite_from_file",
]
