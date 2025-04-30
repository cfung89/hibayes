import datetime
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
import pytz
from inspect_ai.log import (
    EvalDataset,
    EvalLog,
    EvalLogInfo,
    EvalPlan,
    EvalPlanStep,
    EvalResults,
    EvalSample,
    EvalSpec,
    EvalStats,
)
from inspect_ai.model import ChatMessage, ModelUsage
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.scorer import Score

from aisi_inspect_analyse.load.configs.config import DataLoaderConfig
from aisi_inspect_analyse.load.extractors import (
    BaseMetadataExtractor,
    CyberMetadataExtractor,
    MetadataExtractor,
    TokenExtractor,
    ToolsExtractor,
)
from aisi_inspect_analyse.load.load import (
    LogProcessor,
    get_file_list,
    get_sample_df_from_path_efficient,
    is_after_timestamp,
    process_eval_logs_parallel,
    row_generator,
)
from aisi_inspect_analyse.load.utils.patch_inspect_loader import LogSample


@pytest.fixture
def sample_config():
    """Create a sample DataLoaderConfig object for tests."""
    return DataLoaderConfig(
        enabled_extractors=["base"], custom_extractors=[], files_to_process=["test_dir"]
    )


@pytest.fixture
def sample_score():
    """Create a sample Score object."""
    return Score(value=0.75)


@pytest.fixture
def sample_message():
    """Create a sample ChatMessage."""
    return {"role": "user", "content": "test message"}


@pytest.fixture
def sample_model_output():
    """Create a sample ModelOutput."""
    return MagicMock(spec=ModelOutput)


@pytest.fixture
def sample_log_sample():
    """Create a sample LogSample object."""
    sample = MagicMock(spec=LogSample)
    sample.id = "test-sample-1"
    sample.epoch = 1
    return sample


@pytest.fixture
def sample_eval_sample(sample_score, sample_message, sample_model_output):
    """Create a sample EvalSample object for testing."""
    # Create a mock with the necessary attributes based on the EvalSample class structure
    sample = MagicMock(spec=EvalSample)
    sample.id = "test-sample-1"
    sample.epoch = 1
    sample.target = "test_target"
    sample.messages = [sample_message]
    sample.output = sample_model_output
    sample.metadata = {"challenge_metadata": {"category": "test_category"}}

    # Setup scores dictionary
    scores_dict = MagicMock()
    scores_dict.values.return_value = [sample_score]
    sample.scores = scores_dict

    # Add property for score
    sample.score = sample_score

    # Setup model usage
    sample.model_usage = {
        "test_model": ModelUsage(
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            input_tokens_cache_write=10,
            input_tokens_cache_read=5,
        )
    }

    return sample


@pytest.fixture
def sample_eval_spec():
    """Create a sample EvalSpec."""
    spec = MagicMock(spec=EvalSpec)
    spec.model = "test-model"
    spec.dataset = MagicMock(spec=EvalDataset)
    spec.dataset.name = "test_dataset"
    return spec


@pytest.fixture
def sample_eval_stats():
    """Create a sample EvalStats."""
    stats = MagicMock(spec=EvalStats)
    stats.completed_at = "2024-03-18T14:21:24+00:00"
    return stats


@pytest.fixture
def sample_eval_results():
    """Create a sample EvalResults."""
    return MagicMock(spec=EvalResults)


@pytest.fixture
def sample_eval_plan():
    """Create a sample EvalPlan."""
    plan = MagicMock(spec=EvalPlan)
    plan.steps = [
        EvalPlanStep(
            solver="use_tools",
            params={"tools": [{"name": "test_tool"}]},
        )
    ]
    return plan


@pytest.fixture
def sample_eval_log(
    sample_log_sample,
    sample_eval_spec,
    sample_eval_stats,
    sample_eval_results,
    sample_eval_plan,
):
    """Create a sample EvalLog object for testing."""
    log = MagicMock(spec=EvalLog)
    log.version = 2
    log.status = "success"
    log.eval = sample_eval_spec
    log.plan = sample_eval_plan
    log.results = sample_eval_results
    log.stats = sample_eval_stats
    log.samples = [sample_log_sample]
    log.reductions = None
    log.location = "test-location"

    return log


@pytest.fixture
def eval_log_info():
    log_info = MagicMock(spec=EvalLogInfo)
    log_info.name = "test_log.eval"

    return log_info


@pytest.fixture
def sample_config_yaml():
    """Sample YAML content for configuration."""
    return """
    extractors:
      enabled:
        - base
        - tools
        - tokens
    """


@pytest.fixture
def temp_config_file(sample_config_yaml):
    """Create a temporary YAML config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(sample_config_yaml)
        config_path = f.name

    yield config_path

    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)


@pytest.fixture
def temp_file_list():
    """Create temporary files for testing file_list."""
    files = []
    file_paths = []

    # Create temp files
    for i in range(3):
        fd, path = tempfile.mkstemp(suffix=".eval")
        os.close(fd)
        file_paths.append(path)

    for i in range(2):
        fd, path = tempfile.mkstemp(suffix=".eval")
        os.close(fd)
        files.append(path)

    # Create a file that contains paths
    fd, list_path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(files))

    file_paths.append(list_path)

    yield file_paths

    # Cleanup
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)


# Tests for DataLoaderConfig
def test_processor_config_init():
    """Test DataLoaderConfig initialisation."""
    config = DataLoaderConfig(
        enabled_extractors=["base", "tools"],
        custom_extractors=[MagicMock(spec=MetadataExtractor)],
        files_to_process=["test_dir"],
        yaml_path="test.yaml",
    )

    assert config.enabled_extractors == ["base", "tools"]
    assert len(config.custom_extractors) == 1
    assert config.yaml_path == "test.yaml"
    assert config.files_to_process == ["test_dir"]


def test_processor_config_from_yaml(temp_config_file):
    """Test loading DataLoaderConfig from YAML."""
    with patch(
        "yaml.safe_load",
        return_value={
            "extractors": {
                "enabled": ["base", "tools", "tokens"],
            },
            "paths": {
                "files_to_process": ["test_dir"],
            },
        },
    ):
        config = DataLoaderConfig.from_yaml(temp_config_file)

        assert config.enabled_extractors == ["base", "tools", "tokens"]
        assert config.yaml_path == temp_config_file
        assert config.custom_extractors == []
        assert config.files_to_process == ["test_dir"]


# Tests for extractors


def test_base_metadata_extractor(sample_eval_sample, sample_eval_log):
    """Test BaseMetadataExtractor functionality."""
    extractor = BaseMetadataExtractor()
    result = extractor.extract(sample_eval_sample, sample_eval_log)

    assert result["score"] == 0.75
    assert result["target"] == "test_target"
    assert result["model"] == "test-model"
    assert result["dataset"] == "test_dataset"
    assert result["task"] == "test-sample-1"
    assert result["epoch"] == 1
    assert result["num_messages"] == 1


def test_base_metadata_extractor_score_normalisation():
    """Test score normalisation in BaseMetadataExtractor."""
    extractor = BaseMetadataExtractor()

    assert extractor._normalise_score("I") == 0.0
    assert extractor._normalise_score("C") == 1.0
    assert extractor._normalise_score(0.5) == 0.5


def test_tools_extractor(sample_eval_sample, sample_eval_log):
    """Test ToolsExtractor functionality."""
    extractor = ToolsExtractor()
    result = extractor.extract(sample_eval_sample, sample_eval_log)

    assert "tools" in result
    assert result["tools"] == ["test_tool"]


def test_token_extractor(sample_eval_sample, sample_eval_log):
    """Test TokenExtractor functionality."""
    extractor = TokenExtractor()
    result = extractor.extract(sample_eval_sample, sample_eval_log)

    assert result["total_tokens"] == 100
    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 50
    assert result["cache_write_tokens"] == 10
    assert result["cache_read_tokens"] == 5


# Tests for LogProcessor


def test_log_processor_init(sample_config):
    """Test LogProcessor initialisation."""
    processor = LogProcessor(sample_config)
    assert len(processor.extractors) == 1
    assert isinstance(processor.extractors[0], BaseMetadataExtractor)


def test_log_processor_with_config():
    """Test LogProcessor with custom config."""
    config = DataLoaderConfig(
        enabled_extractors=["base", "tools"],
        custom_extractors=[],
        files_to_process=["test_dir"],
    )
    processor = LogProcessor(config)

    assert len(processor.extractors) == 2
    assert any(isinstance(ext, BaseMetadataExtractor) for ext in processor.extractors)
    assert any(isinstance(ext, ToolsExtractor) for ext in processor.extractors)


@patch("aisi_inspect_analyse.load.load.read_eval_log_sample")
def test_process_sample(
    mock_read_eval_log_sample,
    sample_log_sample,
    sample_eval_sample,
    sample_eval_log,
    eval_log_info,
):
    """Test processing a single sample."""

    mock_read_eval_log_sample.return_value = sample_eval_sample
    config = DataLoaderConfig(
        enabled_extractors=["base"], custom_extractors=[], files_to_process=["test_dir"]
    )
    processor = LogProcessor(config)
    result = processor.process_sample(sample_log_sample, sample_eval_log, eval_log_info)

    assert "score" in result
    assert "target" in result
    assert "model" in result
    assert "dataset" in result


# Tests for utility functions


def test_is_after_timestamp():
    """Test timestamp filtering function."""
    log = MagicMock()
    log.stats.completed_at = "2024-03-18T14:21:24+00:00"

    # Earlier timestamp should return True
    earlier = datetime.datetime(2024, 3, 17, 0, 0, 0)
    assert is_after_timestamp(earlier, log) is True

    # Later timestamp should return False
    later = datetime.datetime(2024, 3, 19, 0, 0, 0)
    assert is_after_timestamp(later, log) is False

    # None timestamp should return True
    assert is_after_timestamp(None, log) is True


def test_get_file_list(temp_file_list):
    """Test file list processing."""
    result = get_file_list(temp_file_list)

    assert len(result) == 5
    assert temp_file_list[2] in result  # Check random file from list


@patch("aisi_inspect_analyse.load.load.process_eval_logs_parallel")
@patch("aisi_inspect_analyse.load.load.list_eval_logs")
def test_row_generator(
    mock_list_eval_logs,
    mock_process_parallel,
    sample_eval_log,
    sample_eval_sample,
):
    """Test row generator with files_to_process."""
    # Setup mocks
    mock_list_eval_logs.return_value = ["log1", "log2"]
    mock_process_parallel.return_value = iter(
        [{"score": 0.75, "model": "test-model"}, {"score": 0.80, "model": "test-model"}]
    )

    # Create processor with config
    config = DataLoaderConfig(
        enabled_extractors=["base"], custom_extractors=[], files_to_process=["test_dir"]
    )
    processor = LogProcessor(config)

    with patch("os.path.exists", return_value=True):
        rows = list(row_generator(processor=processor, files_to_process=["test_dir"]))

    assert len(rows) == 2
    assert "score" in rows[0]
    assert rows[0]["score"] == 0.75

    mock_list_eval_logs.assert_called_once_with(log_dir="test_dir")
    mock_process_parallel.assert_called_once()


def test_row_generator_validation():
    """Test row generator validation checks."""
    config = DataLoaderConfig(
        enabled_extractors=["base"], custom_extractors=[], files_to_process=["test_dir"]
    )
    processor = LogProcessor(config)

    with pytest.raises(ValueError):
        next(
            row_generator(processor=processor, files_to_process=[])
        )  # Empty files_to_process


@patch("aisi_inspect_analyse.load.load.row_generator")
@patch("pandas.read_json")
def test_get_sample_df_from_path_efficient(mock_read_json, mock_row_generator):
    """Test efficient DataFrame creation function."""
    # Setup mock row generator to return one sample
    mock_row_generator.return_value = iter([{"score": 0.75, "target": "test_target"}])
    mock_read_json.return_value = pd.DataFrame(
        [{"score": 0.75, "target": "test_target"}]
    )

    # Create processor with config
    config = DataLoaderConfig(
        enabled_extractors=["base"], custom_extractors=[], files_to_process=["test_dir"]
    )
    processor = LogProcessor(config)

    # Create temporary output file
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "processed.jsonl")
        df = get_sample_df_from_path_efficient(
            processor=processor, files_to_process=["test_dir"], output_dir=temp_dir
        )

        assert not df.empty
        assert "score" in df.columns
        assert "target" in df.columns
        mock_read_json.assert_called_once()


@patch("aisi_inspect_analyse.load.load.read_eval_log")
@patch("aisi_inspect_analyse.load.load.read_eval_log_sample")
def test_process_eval_logs_parallel(
    mock_read_eval_log_sample,
    mock_read_eval_log,
    sample_eval_sample,
    sample_eval_log,
    sample_log_sample,
):
    """Test parallel processing of eval logs."""
    # Setup mocks
    mock_read_eval_log.return_value = sample_eval_log
    mock_read_eval_log_sample.return_value = sample_eval_sample

    # Create log info
    log_info = MagicMock(spec=EvalLogInfo)

    # Create processor with config
    config = DataLoaderConfig(
        enabled_extractors=["base"], custom_extractors=[], files_to_process=["test_dir"]
    )
    processor = LogProcessor(config)

    # Run the function
    results = list(
        process_eval_logs_parallel(
            eval_logs=[log_info], processor=processor, cutoff=None
        )
    )

    assert len(results) == 1
    assert "score" in results[0]
    assert "model" in results[0]
    assert results[0]["model"] == "test-model"


# Tests for edge cases
def test_log_processor_with_unknown_extractor():
    """Test LogProcessor behavior with unknown extractor name."""
    config = DataLoaderConfig(
        enabled_extractors=["base", "nonexistent_extractor"],
        custom_extractors=[],
        files_to_process=["test_dir"],
    )
    processor = LogProcessor(config)

    # Should only have the base extractor
    assert len(processor.extractors) == 1
    assert isinstance(processor.extractors[0], BaseMetadataExtractor)


def test_get_sample_df_with_empty_dir():
    """Test behavior with empty directory."""
    # Create empty temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the list_eval_logs to return empty list
        with patch("aisi_inspect_analyse.load.load.list_eval_logs", return_value=[]):
            config = DataLoaderConfig(
                enabled_extractors=["base"],
                custom_extractors=[],
                files_to_process=[temp_dir],
            )
            processor = LogProcessor(config)
            df = get_sample_df_from_path_efficient(
                processor=processor, files_to_process=[temp_dir]
            )
            assert df.empty


@patch("aisi_inspect_analyse.load.load.read_eval_log_sample")
def test_processor_with_multiple_custom_extractors(
    mock_read_eval_log_sample,
    sample_log_sample,
    sample_eval_sample,
    sample_eval_log,
    eval_log_info,
):
    """Test processor with multiple custom extractor."""

    mock_read_eval_log_sample.return_value = sample_eval_sample
    # Create custom mock extractors
    custom1 = MagicMock(spec=MetadataExtractor)
    custom1.extract.return_value = {"custom1": "value1"}
    custom1.__class__.__name__ = "CustomExtractor1"

    custom2 = MagicMock(spec=MetadataExtractor)
    custom2.extract.return_value = {"custom2": "value2"}
    custom2.__class__.__name__ = "CustomExtractor2"

    config = DataLoaderConfig(
        enabled_extractors=["base"],
        custom_extractors=[custom1, custom2],
        files_to_process=["test_dir"],
    )
    processor = LogProcessor(config)

    # Should have base + 2 custom extractors
    assert len(processor.extractors) == 3

    # Process a sample
    result = processor.process_sample(sample_log_sample, sample_eval_log, eval_log_info)

    # Both custom extractors should have been called
    custom1.extract.assert_called_once()
    custom2.extract.assert_called_once()

    # Result should contain values from both extractors
    assert "custom1" in result
    assert "custom2" in result
    assert result["custom1"] == "value1"
    assert result["custom2"] == "value2"


def test_get_file_list_with_empty_list():
    """Test get_file_list with empty input."""
    result = get_file_list([])
    assert result == []


@patch("aisi_inspect_analyse.load.load.read_eval_log")
def test_is_after_timestamp_with_timezone_aware_cutoff(mock_read_eval_log):
    """Test is_after_timestamp with timezone-aware cutoff datetime."""
    # Create a timezone-aware datetime
    log = MagicMock()
    log.stats.completed_at = "2024-03-18T14:21:24+00:00"

    # Create timezone-aware cutoff
    cutoff = datetime.datetime(2024, 3, 17, 0, 0, 0, tzinfo=pytz.UTC)

    assert is_after_timestamp(cutoff, log) is True


@patch("aisi_inspect_analyse.load.load.read_eval_log_sample")
def test_process_sample_with_error(
    mock_read_eval_log_sample, sample_log_sample, sample_eval_log, eval_log_info
):
    """Test process_sample handling of extractor errors."""
    # Setup sample
    mock_read_eval_log_sample.return_value = sample_log_sample

    # Create failing extractor
    failing_extractor = MagicMock(spec=MetadataExtractor)
    failing_extractor.extract.side_effect = Exception("Test error")
    failing_extractor.__class__.__name__ = "FailingExtractor"

    # Create config with the failing extractor
    config = DataLoaderConfig(
        enabled_extractors=["base"],
        custom_extractors=[failing_extractor],
        files_to_process=["test_dir"],
    )
    processor = LogProcessor(config)

    # Process the sample
    result = processor.process_sample(sample_log_sample, sample_eval_log, eval_log_info)

    # Should have recorded the error
    assert "processing_errors" in result
    assert "Error in FailingExtractor" in result["processing_errors"]


@patch("aisi_inspect_analyse.load.load.pd.read_json")
def test_get_sample_df_from_path_efficient_with_cached_data(mock_read_json):
    """Test get_sample_df_from_path_efficient using cached data."""
    # Create mock DataFrame
    mock_df = pd.DataFrame([{"score": 0.9, "model": "test-model"}])
    mock_read_json.return_value = mock_df

    # Create a temporary cached file
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as cached_file:
        # Call the function with cached_data_path
        config = DataLoaderConfig(
            enabled_extractors=["base"],
            custom_extractors=[],
            files_to_process=["test_dir"],
        )
        processor = LogProcessor(config)

        # Create patch for os.path.exists to return True for the cached file
        with patch("os.path.exists", return_value=True):
            result_df = get_sample_df_from_path_efficient(
                processor=processor,
                files_to_process=["test_dir"],
                cached_data_path=cached_file.name,
            )

            # Should have used the cached data
            mock_read_json.assert_called_once_with(cached_file.name, lines=True)
            assert result_df is mock_df


def test_process_sample_with_all_extractors(
    sample_log_sample, sample_eval_log, sample_eval_sample, eval_log_info
):
    """Test process_sample with all available extractors."""
    # Create config with all extractors
    config = DataLoaderConfig(
        enabled_extractors=["base", "tools", "tokens"],
        custom_extractors=[],
        files_to_process=["test_dir"],
    )
    processor = LogProcessor(config)

    # Set up the mock to return our sample
    with patch(
        "aisi_inspect_analyse.load.load.read_eval_log_sample",
        return_value=sample_eval_sample,
    ):
        # Process the sample
        result = processor.process_sample(
            sample_log_sample, sample_eval_log, eval_log_info
        )

    # Should have data from all extractors
    assert "score" in result  # From base extractor
    assert "tools" in result  # From tools extractor
    assert "total_tokens" in result  # From token extractor


@patch("aisi_inspect_analyse.load.load.list_eval_logs")
def test_row_generator_with_s3_path(mock_list_eval_logs):
    """Test row_generator with S3 path (which bypasses os.path.exists check)."""
    # Setup mocks
    mock_list_eval_logs.return_value = []

    # Create processor with config
    config = DataLoaderConfig(
        enabled_extractors=["base"],
        custom_extractors=[],
        files_to_process=["s3://bucket/path"],
    )
    processor = LogProcessor(config)

    # Call with S3 path
    list(row_generator(processor=processor, files_to_process=["s3://bucket/path"]))

    # Should have called list_eval_logs with the S3 path
    mock_list_eval_logs.assert_called_once_with(log_dir="s3://bucket/path")


@patch("aisi_inspect_analyse.load.load.read_eval_log_sample")
def test_log_processor_setup_extractors(mock_read_eval_log_sample):
    """Test LogProcessor._setup_extractors method."""
    # Create custom extractors
    custom_extractor = MagicMock(spec=MetadataExtractor)
    custom_extractor.extract.return_value = {"custom": "value"}

    # Create config with specific extractors
    config = DataLoaderConfig(
        enabled_extractors=["base", "tokens"],
        custom_extractors=[custom_extractor],
        files_to_process=["test_dir"],
    )

    # Create processor
    processor = LogProcessor(config)

    # Verify extractors are set up correctly
    assert len(processor.extractors) == 3
    assert any(isinstance(ext, BaseMetadataExtractor) for ext in processor.extractors)
    assert any(isinstance(ext, TokenExtractor) for ext in processor.extractors)
    assert custom_extractor in processor.extractors


def test_processor_setup_no_extractors():
    """Test LogProcessor with no enabled extractors."""
    config = DataLoaderConfig(
        enabled_extractors=[], custom_extractors=[], files_to_process=["test_dir"]
    )
    processor = LogProcessor(config)

    # Should have no extractors
    assert len(processor.extractors) == 0


@patch("aisi_inspect_analyse.load.load.list_eval_logs")
def test_row_generator_with_nonexistent_path(mock_list_eval_logs):
    """Test row_generator with a path that doesn't exist."""
    mock_list_eval_logs.return_value = []

    config = DataLoaderConfig(
        enabled_extractors=["base"],
        custom_extractors=[],
        files_to_process=["nonexistent_path"],
    )
    processor = LogProcessor(config)

    # Mock os.path.exists to return False
    with patch("os.path.exists", return_value=False):
        # Should log a warning but continue
        rows = list(
            row_generator(processor=processor, files_to_process=["nonexistent_path"])
        )
        assert len(rows) == 0


@patch("aisi_inspect_analyse.load.load.update_inspect_recorders")
def test_get_sample_df_patches_inspect_loaders(mock_update_inspect_recorders):
    """Test that get_sample_df_from_path_efficient updates inspect recorders."""
    config = DataLoaderConfig(
        enabled_extractors=["base"], custom_extractors=[], files_to_process=["test_dir"]
    )
    processor = LogProcessor(config)

    with patch("aisi_inspect_analyse.load.load.row_generator") as mock_row_gen:
        mock_row_gen.return_value = iter([])
        with patch("builtins.open", mock_open()):
            with patch("pandas.read_json", return_value=pd.DataFrame()):
                get_sample_df_from_path_efficient(
                    processor=processor, files_to_process=["test_dir"]
                )


# Integration test with real log files
@pytest.mark.parametrize("log_dir", ["logs_1", "logs_2"])
def test_real_log_processing(log_dir):
    """Test processing with actual log files from the test directory"""
    # Get the path to the test logs directory
    tests_dir = Path(__file__).parent
    logs_dir = tests_dir / log_dir

    if not logs_dir.exists() or not any(logs_dir.iterdir()):
        pytest.skip(f"No test log files available in {log_dir}")

    logs_dir = str(logs_dir)

    config = DataLoaderConfig(
        enabled_extractors=["base"], custom_extractors=[], files_to_process=[logs_dir]
    )
    processor = LogProcessor(config)
    df = get_sample_df_from_path_efficient(
        files_to_process=[logs_dir], processor=processor
    )

    # Verify we got some data
    assert not df.empty
    assert "score" in df.columns
    assert "model" in df.columns

    assert "claude-3-7-sonnet-20250219" in df["model"].unique(), "Missing model"
    assert "o3-mini" in df["model"].unique(), "Missing model"

    assert len(df) == 60, "Incorrect number of samples"
