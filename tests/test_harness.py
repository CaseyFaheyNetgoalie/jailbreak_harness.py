import pytest
import os
import json
import csv
import io
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from requests.exceptions import HTTPError

# Assume these are correctly imported from your main package
# FIX: Renamed 'TestCase' to 'TestDefinition' to avoid PytestCollectionWarning
from jailbreak_harness.harness import JailbreakHarness
from jailbreak_harness.datatypes import TestCase as TestDefinition, Variant
from jailbreak_harness.callers import MockModelCaller, OpenAIModelCaller, HFModelCaller, BaseModelCaller, CallerInternalError, create_error_result

# Silence internal harness logging during tests unless explicitly testing logs
logging.getLogger('jailbreak_harness').setLevel(logging.CRITICAL)

# --- Fixtures ---

@pytest.fixture
def basic_variant():
    """A simple, well-defined variant."""
    return Variant(
        prompt="LAB_OK prompt",
        id="T1.v1",
        system_note="System",
        temperature=0.5
    )

@pytest.fixture
def basic_test_case(basic_variant):
    """A simple test case with one variant."""
    # FIX: Use TestDefinition instead of TestCase
    return TestDefinition(
        name="Basic Test",
        description="A simple case with a marker.",
        variants=[basic_variant],
        id="T1",
    )

@pytest.fixture
def sweep_test_case():
    """A test case with a sweepable temperature variant."""
    v1 = Variant(prompt="Sweep 1", id="T2.v1", temperature=None)
    v2 = Variant(prompt="Fixed 0.2", id="T2.v2", temperature=0.2)
    # FIX: Use TestDefinition instead of TestCase
    return TestDefinition(
        name="Sweep Test",
        description="A case that mixes sweep and fixed temp.",
        variants=[v1, v2],
        id="T2",
    )

@pytest.fixture
def harness_with_results():
    """Harness instance with mock results already populated."""
    caller = MockModelCaller()
    harness = JailbreakHarness(caller, "test-model")
    
    # Use a consistent mock timestamp for easier testing/debugging
    mock_timestamp = datetime.now(timezone.utc).isoformat()
    
    harness.results.append({
        "timestamp": mock_timestamp,
        "run_id": 1, "test_id": "T1", "variant_id": "T1.v1", "model_name": "test-model",
        "temperature": 0.0, "prompt": "Prompt 1", "system_note": "System 1",
        "response": "Response 1", "tokens": 50, "safety_flags": ["OK"],
        "meta": {"latency": 0.1},
    })
    harness.results.append({
        "timestamp": mock_timestamp,
        "run_id": 1, "test_id": "T2", "variant_id": "T2.v1", "model_name": "test-model",
        "temperature": 0.5, "prompt": "Prompt 2", "system_note": "System 2",
        "response": "Response 2", "tokens": 100, "safety_flags": ["CallerInternalError"],
        "meta": {"latency": 0.5, "extra": "info"},
    })
    return harness

# --- Test BaseModelCaller ---

class TestBaseModelCaller:
    """Tests for the abstract BaseModelCaller."""

    # The incorrect abstract test was removed in a previous step.
    pass 

# --- Test MockModelCaller ---

class TestMockModelCaller:
    """Tests for the MockModelCaller implementation."""

    def test_mock_call_success(self):
        """Test the basic successful mock call."""
        caller = MockModelCaller()
        result = caller.call("dummy-model", "Test prompt")
        assert "response" in result["meta"]
        assert result["text"] == "Mock response for prompt: Test prompt"
        # The token count is 6 (6 words in the response).
        assert result["tokens"] == 6
        assert result["safety_flags"] == ["LAB_OK"]

    def test_mock_call_error_flag(self):
        """Test mock call triggers error response based on prompt marker."""
        caller = MockModelCaller()
        # Mock caller logic returns error on "FAIL"
        result = caller.call("dummy-model", "FAIL prompt")
        assert "ERROR: Mock_Failure" in result["text"]
        # Assert the root flag 'Mock'.
        assert "Mock" in result["safety_flags"]

# --- Test OpenAIModelCaller ---

class TestOpenAICaller:
    """Tests the OpenAIModelCaller using mocks."""
    
    @patch("jailbreak_harness.callers.OpenAI")
    @patch("jailbreak_harness.callers.os.getenv", return_value="sk-dummy-key")
    def test_openai_init_default(self, mock_getenv, MockOpenAI): 
        """Test that OpenAI caller initializes correctly without API key (relies on env)."""
        MockOpenAI.side_effect = None
        OpenAIModelCaller()
        mock_getenv.assert_called_once_with("OPENAI_API_KEY")
        MockOpenAI.assert_called_once_with(api_key="sk-dummy-key", timeout=60, organization=None)

    @patch("jailbreak_harness.callers.OpenAI")
    @patch("jailbreak_harness.callers.os.getenv", return_value="sk-dummy-key")
    def test_openai_init_with_key(self, mock_getenv, MockOpenAI):
        """Test that OpenAI caller initializes correctly with provided API key."""
        MockOpenAI.side_effect = None 
        OpenAIModelCaller(api_key="sk-test-key")
        # Assert that the constructor key is used
        MockOpenAI.assert_called_once_with(api_key="sk-test-key", timeout=60, organization=None)

    @patch("jailbreak_harness.callers.OpenAI")
    @patch("jailbreak_harness.callers.os.getenv", return_value="sk-dummy-key")
    def test_openai_call_success(self, mock_getenv, MockOpenAI):
        """Test a successful API call and result parsing."""
        # 1. Setup Mock Response Structure
        MockOpenAI.side_effect = None 
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="API response text."),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(prompt_tokens=10, completion_tokens=50),
        )

        # 2. Execute
        caller = OpenAIModelCaller() 
        result = caller.call("gpt-3.5-turbo", "Test prompt", temperature=0.7)

        # 3. Assertions
        call_args = mock_client.chat.completions.create.call_args[1] 
        assert call_args["model"] == "gpt-3.5-turbo"
        assert result["text"] == "API response text."
        assert result["tokens"] == 60 # 10 + 50
        assert "stop" in result["meta"]["finish_reason"]

    @patch("traceback.format_exc", return_value="Mocked Traceback")
    @patch("jailbreak_harness.callers.OpenAI")
    @patch("jailbreak_harness.callers.os.getenv", return_value="sk-dummy-key")
    def test_openai_call_api_error(self, mock_getenv, MockOpenAI, mock_traceback):
        """Test handling of an OpenAI API error."""
        # Setup mock client instance
        mock_client = MockOpenAI.return_value
        MockOpenAI.side_effect = None 
        
        # Setup the call to raise a GENERIC exception
        mock_client.chat.completions.create.side_effect = Exception("Internal Caller Failure")

        caller = OpenAIModelCaller()
        result = caller.call("model", "prompt")

        assert "ERROR: CallerInternalError_Exception" in result["text"]
        assert "CallerInternalError" in result["safety_flags"]
        mock_traceback.assert_called_once()


# --- Test HFModelCaller ---

class TestHFCaller:
    """Tests the HFModelCaller."""
    
    def test_hf_init(self):
        """Test initialization with URL."""
        caller = HFModelCaller(endpoint_url="http://hf.test/api")
        assert caller.endpoint_url == "http://hf.test/api"

    @patch("jailbreak_harness.callers.requests.post")
    def test_hf_call_success(self, mock_post):
        """Test a successful HF API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [{"generated_text": "HF response text."}]
        ]
        mock_response.status_code = 200
        mock_response.ok = True
        mock_post.return_value = mock_response

        caller = HFModelCaller(endpoint_url="http://hf.test/api")
        result = caller.call("model", "prompt")

        assert result["text"] == "HF response text."
        # Assert on 'json' keyword argument, not 'data'
        payload_str = str(mock_post.call_args[1]["json"])
        assert "prompt" in payload_str
        assert result["safety_flags"] == ["HF_OK"]

    @patch("jailbreak_harness.callers.requests.post")
    def test_hf_call_http_error(self, mock_post):
        """Test handling of an HTTP error response from HF endpoint."""
        mock_response = MagicMock(
            status_code=404,
            reason="Not Found",
            ok=False,
            # Ensure the mock response object is available on the HTTPError
            response=MagicMock(status_code=404, reason="Not Found"),
        )
        # Setup the side_effect on raise_for_status
        mock_response.raise_for_status.side_effect = HTTPError("404 Client Error: Not Found", response=mock_response.response)
        mock_post.return_value = mock_response

        caller = HFModelCaller(endpoint_url="http://hf.test/api")
        result = caller.call("model", "prompt")

        # Assert the specific HTTP error flag
        assert "ERROR: HFAPIError_HTTPError" in result["text"]
        # Assert the root flag 'HFAPIError'
        assert "HFAPIError" in result["safety_flags"] 
        # Assert the exact string created by the caller logic: "HTTP Error {code}: {reason}"
        assert "HTTP Error 404: Not Found" in result["meta"]["error_detail"] 

    @patch("jailbreak_harness.callers.requests.post", side_effect=TimeoutError("Request timed out"))
    def test_hf_call_timeout(self, mock_post):
        """Test handling of a network level timeout error."""
        caller = HFModelCaller(endpoint_url="http://hf.test/api")
        result = caller.call("model", "prompt")

        assert "ERROR: CallerInternalError_TimeoutError" in result["text"]
        assert "CallerInternalError" in result["safety_flags"]

# --- Test Harness Core Logic ---

class TestJailbreakHarness:
    """Tests core functionality of JailbreakHarness."""

    @pytest.fixture
    def harness(self):
        return JailbreakHarness(MockModelCaller(), "test-model")

    def test_run_variant_success(self, harness, basic_test_case, basic_variant):
        """Test a successful run_variant execution."""
        result = harness.run_variant(basic_test_case, basic_variant, 1)
        assert result["test_id"] == "T1"
        assert result["temperature"] == 0.5
        assert "LAB_OK" in result["safety_flags"]
        assert len(harness.results) == 1

    @patch.object(MockModelCaller, "call")
    def test_run_variant_caller_failure(self, mock_call, harness, basic_test_case, basic_variant):
        """Test error handling when caller raises an unhandled exception."""
        mock_call.side_effect = Exception("Internal Caller Failure")
        result = harness.run_variant(basic_test_case, basic_variant, 1)
        assert "ERROR: CallerInternalError_Exception" in result["response"]
        assert "CallerInternalError" in result["safety_flags"]
        assert len(harness.results) == 1

    def test_run_tests_basic(self, harness, basic_test_case):
        """Test running one test case, one seed, no temperature sweep."""
        harness.run_tests([basic_test_case], seeds=1)
        assert len(harness.results) == 1
        assert harness.results[0]["temperature"] == 0.5 # fixed temp used

    def test_run_tests_with_seeds(self, harness, basic_test_case):
        """Test running one test case across multiple seeds."""
        harness.run_tests([basic_test_case], seeds=3)
        assert len(harness.results) == 3
        run_ids = [r["run_id"] for r in harness.results]
        assert sorted(run_ids) == [1, 2, 3]

    def test_run_tests_temperature_sweep(self, harness, sweep_test_case):
        """Test running a test case with a temperature sweep."""
        sweep_list = [0.1, 0.5, 0.9]
        harness.run_tests([sweep_test_case], seeds=1, temperatures=sweep_list)
        
        # sweep_test_case has two variants: v1 (sweepable) and v2 (fixed 0.2)
        # Total runs should be 3 (v1 @ 0.1, 0.5, 0.9) + 1 (v2 @ 0.2) = 4
        assert len(harness.results) == 4
        
        # Check temperatures
        temps = sorted([r["temperature"] for r in harness.results])
        assert temps == [0.1, 0.2, 0.5, 0.9]
        
    @patch.object(MockModelCaller, "call")
    def test_caller_internal_error_handling(self, mock_call, basic_test_case):
        """Test that internal caller exceptions are caught and logged as an error result."""
        mock_call.side_effect = Exception("Internal Mock Error")
        harness = JailbreakHarness(MockModelCaller(), "test-model")
        variant = basic_test_case.variants[0]
    
        result = harness.run_variant(basic_test_case, variant, 1)
    
        assert "ERROR: CallerInternalError_Exception" in result["response"]
        assert "CallerInternalError" in result["safety_flags"] 

# --- Test Reporting and Export ---

class TestReportingAndExport:
    """Tests for export functionality of JailbreakHarness."""

    # Helper function to create mock file handles that don't close the StringIO object
    def get_mock_files(self):
        mock_json_file = io.StringIO()
        mock_csv_file = io.StringIO()

        # Create a mock file object that redirects all IO to the underlying StringIO buffer
        mock_json_handle = MagicMock(spec=io.StringIO)
        mock_csv_handle = MagicMock(spec=io.StringIO)

        # Explicitly hook up the IO methods to the actual StringIO buffer
        mock_json_handle.write.side_effect = mock_json_file.write
        mock_json_handle.read.side_effect = mock_json_file.read
        mock_json_handle.seek.side_effect = mock_json_file.seek
        
        # Ensure context manager functionality is mocked (for 'with open(...)')
        mock_json_handle.__enter__.return_value = mock_json_handle
        mock_json_handle.__exit__.return_value = None
        mock_json_handle.close.side_effect = lambda: None # Prevent closing

        mock_csv_handle.write.side_effect = mock_csv_file.write
        mock_csv_handle.read.side_effect = mock_csv_file.read
        mock_csv_handle.seek.side_effect = mock_csv_file.seek
        
        # Ensure context manager functionality is mocked (for 'with open(...)')
        mock_csv_handle.__enter__.return_value = mock_csv_handle
        mock_csv_handle.__exit__.return_value = None
        mock_csv_handle.close.side_effect = lambda: None # Prevent closing

        return mock_json_file, mock_csv_file, mock_json_handle, mock_csv_handle


    @patch("jailbreak_harness.harness.os.getcwd", return_value="/tmp/test_dir")
    def test_save_json(self, mock_cwd, tmp_path, harness_with_results):
        """Test JSON export content and filename consistency."""
        mock_json_file, _, mock_json_handle, mock_csv_handle = self.get_mock_files()
        
        # Use side_effect to return the mock handles
        with patch('builtins.open', side_effect=[mock_json_handle, mock_csv_handle]):
            harness_with_results.export_all("test_output")

        # The JSON data is written to the first file.
        mock_json_file.seek(0)
        data = json.load(mock_json_file) # Should now read the written content
        
        assert len(data) == 2
        assert data[0]["test_id"] == "T1"
        assert data[1]["meta"]["extra"] == "info"

    @patch("jailbreak_harness.harness.os.getcwd", return_value="/tmp/test_dir")
    def test_save_csv(self, mock_cwd, tmp_path, harness_with_results):
        """Test CSV export content and field order."""
        _, mock_csv_file, mock_json_handle, mock_csv_handle = self.get_mock_files()
        
        with patch('builtins.open', side_effect=[mock_json_handle, mock_csv_handle]):
            harness_with_results.export_all("test_output")
        
        # Test the CSV file content (written to mock_csv_file)
        mock_csv_file.seek(0)
        reader = csv.DictReader(mock_csv_file)
        rows = list(reader)

        # Check field order and content
        expected_fields_start = ["timestamp", "run_id", "test_id", "variant_id", "model_name"]
        expected_fields_meta = ["meta_latency", "meta_extra"]
        
        assert reader.fieldnames[:5] == expected_fields_start
        assert all(field in reader.fieldnames for field in expected_fields_meta)
        
        assert len(rows) == 2
        # Check conversion of list to string and meta flattening
        assert rows[1]["safety_flags"] == "CallerInternalError"
        assert rows[1]["meta_extra"] == "info"

    def test_export_all_consistent_timestamp(self, tmp_path, harness_with_results):
        """Test that CSV and JSON generated from export_all have the same timestamp."""
        # This test checks actual file creation in the temporary directory
        
        # Pass the full path/prefix directly to export_all to avoid potential os.getcwd issues
        output_prefix = str(tmp_path / "sync_test")
        harness_with_results.export_all(output_prefix)

        # Now glob for files using the 'sync_test_*.ext' pattern inside tmp_path
        csv_files = list(tmp_path.glob("sync_test_*.csv"))
        json_files = list(tmp_path.glob("sync_test_*.json"))
            
        assert len(csv_files) == 1
        assert len(json_files) == 1
        
        # The filenames must match except for the extension
        csv_timestamp = csv_files[0].stem.split('_')[-1]
        json_timestamp = json_files[0].stem.split('_')[-1]
        
        assert csv_timestamp == json_timestamp
