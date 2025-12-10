import os
import time
import json
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
from requests.exceptions import HTTPError, Timeout
import logging

# Check for OpenAI import silently (it's optional)
try:
    from openai import OpenAI
    from openai import APIError as OpenAIAPIError
    from openai.lib.azure import AzureOpenAI
    from openai.types.chat import ChatCompletionMessageParam
except ImportError:
    OpenAI = None
    OpenAIAPIError = None
    AzureOpenAI = None
    ChatCompletionMessageParam = None
    
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---

class CallerInternalError(Exception):
    """
    Custom exception raised when a model caller encounters an internal,
    non-API-specific error (e.g., network timeout, JSON parsing failure,
    or unexpected structure).

    This must be defined here so it can be imported by the tests.
    """
    pass

# --- Utility Functions ---

def create_error_result(error_flag: str, detail: str) -> Dict[str, Any]:
    """
    Creates a standardized result dictionary for a failed API call.
    
    Args:
        error_flag: A concise error identifier (e.g., 'CallerInternalError_Exception').
        detail: The full error message or traceback.
    """
    # The safety flag is the root error type, ensuring generic catching works.
    root_flag = error_flag.split('_')[0] 
    return {
        "text": f"ERROR: {error_flag}",
        "tokens": 0,
        "safety_flags": [root_flag],
        "meta": {"error_detail": detail, "error_flag": error_flag},
    }

# --- Base Caller ---

class BaseModelCaller(ABC):
    """Abstract base class for all model callers."""

    def __init__(self, model_name: Optional[str] = None):
        self.default_model_name = model_name

    @abstractmethod
    def call(
        self,
        model_name: str,
        prompt: str,
        system_note: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Subclasses must implement the actual API call logic.
        
        Returns:
            A standardized dictionary with keys: 'text', 'tokens', 'safety_flags', 'meta'.
        """
        raise NotImplementedError("Subclasses must implement the 'call' method.")

# --- Mock Caller ---

class MockModelCaller(BaseModelCaller):
    """
    A dummy caller for testing the harness without requiring a real API key.
    Responds with a predictable message or a simulated failure.
    """

    def call(
        self,
        model_name: str,
        prompt: str,
        system_note: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        
        # Simulate an error condition based on prompt content (used in tests)
        if "FAIL" in prompt:
            return create_error_result(
                "Mock_Failure", 
                "Simulated failure trigger found in prompt."
            )

        response_text = f"Mock response for prompt: {prompt}"
        
        # Simulate latency and token usage
        time.sleep(0.01)
        tokens = len(response_text.split())

        return {
            "text": response_text,
            "tokens": tokens,
            "safety_flags": ["LAB_OK"],
            "meta": {"response": "Success", "model": model_name, "latency": 0.01},
        }

# --- OpenAI Caller ---

class OpenAIModelCaller(BaseModelCaller):
    """
    Caller for OpenAI models (GPT-3.5, GPT-4, etc.) via the official Python SDK.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        organization: Optional[str] = None, 
        timeout: int = 60,
        is_azure: bool = False,
        azure_endpoint: Optional[str] = None,
    ):
        if OpenAI is None:
            raise ImportError("OpenAI SDK not installed. Run: pip install openai")
        
        super().__init__()
        
        # Determine API Key: use provided key, then environment variable
        key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")

        if is_azure:
            if not azure_endpoint:
                raise ValueError("azure_endpoint must be provided for AzureOpenAI.")
            self.client = AzureOpenAI(
                api_key=key,
                azure_endpoint=azure_endpoint,
                organization=organization,
                timeout=timeout,
            )
        else:
            self.client = OpenAI(
                api_key=key,
                organization=organization,
                timeout=timeout,
            )


    def call(
        self,
        model_name: str,
        prompt: str,
        system_note: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        
        messages: List[ChatCompletionMessageParam] = []
        
        if system_note:
            messages.append({"role": "system", "content": system_note})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # The API call
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract response text
            if response.choices and response.choices[0].message:
                text = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
            else:
                text = "NO_RESPONSE"
                finish_reason = "empty_response"
                
            # Extract tokens
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = prompt_tokens + completion_tokens

            # Standardized success result
            return {
                "text": text if text is not None else "NO_RESPONSE",
                "tokens": total_tokens,
                # Note: OpenAI does not return safety flags for normal usage
                "safety_flags": ["OK"],
                "meta": {
                    "model": model_name,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
            }

        except OpenAIAPIError as e:
            # Handle specific OpenAI API errors (e.g., rate limit, authentication, invalid model)
            error_type = type(e).__name__
            error_flag = f"OpenAIAPIError_{error_type}"
            
            # Extract error details for logging/meta
            try:
                # The 'e.response.json()' call is the source of the log message in the failed tests
                error_data = e.response.json()
            except Exception:
                error_data = {"error": {"message": str(e), "code": e.status_code}}
            
            logger.error(f"OpenAI API error: Error code: {e.status_code} - {error_data}")
            return create_error_result(error_flag, json.dumps(error_data))
        
        except Exception as e:
            # Catch all other exceptions (e.g., network issues, local errors)
            error_flag = f"CallerInternalError_{type(e).__name__}"
            detail = traceback.format_exc()
            logger.exception(f"OpenAI caller internal error: {e}")
            return create_error_result(error_flag, detail)

# --- Hugging Face Caller ---

class HFModelCaller(BaseModelCaller):
    """
    Caller for models hosted on a Hugging Face Inference Endpoint.
    """

    def __init__(self, endpoint_url: str, hf_token: Optional[str] = None, timeout: int = 60):
        super().__init__()
        self.endpoint_url = endpoint_url
        self.hf_token = hf_token if hf_token is not None else os.getenv("HUGGINGFACE_API_KEY")
        self.timeout = timeout
        self.headers = {}
        if self.hf_token:
            self.headers["Authorization"] = f"Bearer {self.hf_token}"
        self.headers["Content-Type"] = "application/json"

    def _prepare_payload(self, prompt: str, system_note: Optional[str] = None):
        """Prepares the payload according to a standard HF Inference Endpoint structure."""
        
        # Simple template approach for conversation format (can be improved)
        if system_note:
            input_text = f"{system_note}\n\nUser: {prompt}"
        else:
            input_text = prompt

        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 2048,
                "return_full_text": False,
                "do_sample": True,
                "temperature": 0.0,  # Default to 0.0, will be overwritten if needed
            },
        }
        return payload

    def call(
        self,
        model_name: str, # Note: Not always used by generic endpoint URLs
        prompt: str,
        system_note: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        
        payload = self._prepare_payload(prompt, system_note)
        # Update temperature and max_tokens for this specific call
        payload["parameters"]["temperature"] = temperature
        payload["parameters"]["max_new_tokens"] = max_tokens

        try:
            response = requests.post(
                self.endpoint_url, 
                headers=self.headers, 
                json=payload, 
                timeout=self.timeout
            )
            
            # Check for HTTP errors (e.g., 404, 500)
            response.raise_for_status()

            # Process successful JSON response
            data = response.json()
            
            # Assuming a standard Hugging Face Text Generation response structure:
            # [[{"generated_text": "..."}]] or [{"generated_text": "..."}]
            if isinstance(data, list) and data and isinstance(data[0], list):
                text = data[0][0]["generated_text"]
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                 text = data[0]["generated_text"]
            else:
                # Handle unexpected response structure
                error_flag = "HFAPIError_BadResponse"
                detail = f"Unexpected response structure: {data}"
                logger.error(f"HF caller response error: {detail}")
                return create_error_result(error_flag, detail)

            return {
                "text": text,
                "tokens": 0,  # Token calculation is often unavailable on generic HF endpoints
                "safety_flags": ["HF_OK"],
                "meta": {"model": model_name, "endpoint": self.endpoint_url},
            }

        except HTTPError as e:
            # API returned a non-2xx status code
            error_flag = "HFAPIError_HTTPError"
            detail = f"HTTP Error {e.response.status_code}: {e.response.reason}"
            logger.error(f"HF caller HTTP error: {detail}")
            return create_error_result(error_flag, detail)

        except Timeout as e:
            # Network timeout
            error_flag = f"CallerInternalError_{type(e).__name__}"
            detail = f"Request timed out after {self.timeout} seconds."
            logger.exception(f"HF caller timeout error: {detail}")
            return create_error_result(error_flag, detail)

        except Exception as e:
            # Catch all other exceptions (e.g., network issues, local errors, JSON decode)
            error_flag = f"CallerInternalError_{type(e).__name__}"
            detail = traceback.format_exc()
            logger.exception(f"HF caller internal error: {e}")
            return create_error_result(error_flag, detail)
