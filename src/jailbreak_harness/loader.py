# src/jailbreak_harness/loader.py

import json
import logging
from typing import List, Dict, Any
import yaml

from .datatypes import TestCase, Variant

logger = logging.getLogger(__name__)


def load_test_suite_from_file(file_path: str) -> List[TestCase]:
    """
    Loads a test suite from a JSON or YAML file.

    Args:
        file_path: Path to the input file.

    Returns:
        A list of TestCase objects.
    
    Raises:
        ValueError: If the file format is invalid or data validation fails.
    """
    # 1. Update supported file formats
    supported_formats = ('.json', '.yaml', '.yml')
    if not file_path.lower().endswith(supported_formats):
        raise ValueError(
            f"Unsupported file format: {file_path}. Must be one of: {', '.join(supported_formats)}."
        )

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                raw_data = json.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                raw_data = yaml.safe_load(f)
            elif file_path.endswith('.csv'):
                # This is kept as a placeholder/reminder from the earlier code
                raise NotImplementedError("CSV loading is not yet implemented for full TestCases.")
            
    except FileNotFoundError:
        # FIX 1: Propagate the original FileNotFoundError for the test to catch.
        # It's better to let the calling code handle FileNotFoundError directly.
        raise
    except (json.JSONDecodeError, yaml.YAMLError, NotImplementedError) as e:
        # Catch JSON/YAML parsing errors or CSV NotImplementedError
        raise ValueError(f"Invalid file content in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

    # 2. Handle top-level 'tests' key (common in YAML)
    if isinstance(raw_data, dict) and 'tests' in raw_data:
        raw_data = raw_data['tests']

    if not isinstance(raw_data, list):
        raise ValueError("File content must contain a top-level list of test cases.")

    # ---------------- Data Parsing Logic (FIX 2) ----------------
    
    test_suite: List[TestCase] = []
    
    for case_data in raw_data:
        # Extract and validate base TestCase fields
        case_id = case_data.get('id')
        name = case_data.get('name')
        description = case_data.get('description', '')
        raw_variants = case_data.get('variants', []) # Default to empty list
        
        # Check mandatory fields for the TestCase
        if not case_id or not name:
            logger.warning(f"Skipping malformed test case: {case_data.get('id', 'No ID')} (Missing ID or Name)")
            continue

        variants: List[Variant] = []
        for i, v_data in enumerate(raw_variants):
            try:
                # Variant requires 'prompt'
                v_id = v_data.get('id', f"{case_id}.v{i+1}")
                variant = Variant(
                    id=v_id,
                    prompt=v_data['prompt'], 
                    system_note=v_data.get('system_note'),
                    temperature=v_data.get('temperature')
                )
                variants.append(variant)
            except KeyError as e:
                logger.error(f"Invalid variant data in Case {case_id}, Variant {i+1}: Missing required field {e}")
            except ValueError as e:
                logger.error(f"Invalid variant data in Case {case_id}, Variant {i+1}: Validation failed: {e}")

        
        if variants:
            try:
                test_case = TestCase(
                    id=case_id,
                    name=name,
                    description=description,
                    variants=variants,
                    tags=case_data.get('tags', []),
                    expected_failure=case_data.get('expected_failure', False)
                )
                test_suite.append(test_case)
            except ValueError as e:
                logger.error(f"Validation failed for TestCase {case_id}: {e}")

    return test_suite
