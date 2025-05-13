import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
from inspect_ai.model import ModelUsage
from inspect_ai.scorer import Score

from hibayes.load.configs.config import DataLoaderConfig
from hibayes.load.extractors import (
    BaseMetadataExtractor,
    MetadataExtractor,
    TokenExtractor,
    ToolsExtractor,
)
from hibayes.load.load import (
    LogProcessor,
    get_file_list,
    get_sample_df,
    is_after_timestamp,
    process_eval_logs_parallel,
    row_generator,
)


@pytest.fixture
def score() -> Score:
    return Score(value=0.75)


@pytest.fixture
def eval_sample(score):
    s = MagicMock(spec=EvalSample)
    s.id = "sample‑1"
    s.epoch = 1
    s.target = "target‑1"
    s.messages = ["dummy‑msg"]
    s.scores = {"default": score}
    s.score = score
    s.metadata = {"challenge_metadata": {"category": "cat"}}
    s.model_usage = {
        "gpt‑4o": ModelUsage(
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            input_tokens_cache_write=10,
            input_tokens_cache_read=5,
        )
    }
    return s


@pytest.fixture
def eval_log(eval_sample):
    dataset = MagicMock(spec=EvalDataset)
    dataset.name = "unit‑dataset"

    spec = MagicMock(spec=EvalSpec)
    spec.model = "gpt‑4o"
    spec.dataset = dataset

    stats = MagicMock(spec=EvalStats)
    stats.completed_at = "2024-05-01T12:34:56+00:00"

    plan = EvalPlan(
        steps=[EvalPlanStep(solver="use_tools", params={"tools": [{"name": "grep"}]})]
    )

    log = MagicMock(spec=EvalLog)
    log.version = 2
    log.status = "success"
    log.eval = spec
    log.plan = plan
    log.results = MagicMock(spec=EvalResults)
    log.stats = stats
    log.samples = [eval_sample]
    log.location = "in‑mem.eval"
    return log


@pytest.fixture
def log_info():
    info = MagicMock(spec=EvalLogInfo)
    info.name = "in‑mem.eval"
    return info


@pytest.fixture
def dl_cfg(tmp_path: Path):
    """A default DataLoaderConfig pointing at a tmp logs dir."""
    return DataLoaderConfig(
        enabled_extractors=["base", "tools", "tokens"],
        custom_extractors=[],
        files_to_process=[str(tmp_path / "logs")],
    )


def test_dataloaderconfig_init(tmp_path: Path):
    cfg = DataLoaderConfig(
        enabled_extractors=["base", "tools"],
        custom_extractors=[],
        files_to_process=[str(tmp_path)],
    )
    assert cfg.enabled_extractors == ["base", "tools"]
    assert cfg.files_to_process == [str(tmp_path)]


def test_dataloaderconfig_from_yaml(tmp_path: Path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        """
extractors:
  enabled: [base, tokens]
paths:
  files_to_process: [/data]
"""
    )

    with patch(
        "yaml.safe_load",
        return_value={
            "extractors": {"enabled": ["base", "tokens"]},
            "paths": {"files_to_process": ["/data"]},
        },
    ):
        cfg = DataLoaderConfig.from_yaml(str(cfg_file))
    assert cfg.enabled_extractors == ["base", "tokens"]
    assert cfg.files_to_process == ["/data"]


def test_base_extractor(eval_sample, eval_log):
    row = BaseMetadataExtractor().extract(eval_sample, eval_log)
    assert row["score"] == 0.75
    assert row["dataset"] == "unit‑dataset"


def test_base_extractor_normalise_score():
    ext = BaseMetadataExtractor()
    assert ext._normalise_score("I") == 0.0
    assert ext._normalise_score("C") == 1.0
    assert ext._normalise_score(0.5) == 0.5


def test_tools_extractor(eval_sample, eval_log):
    row = ToolsExtractor().extract(eval_sample, eval_log)
    assert row["tools"] == ["grep"]


def test_token_extractor(eval_sample, eval_log):
    row = TokenExtractor().extract(eval_sample, eval_log)
    assert row["total_tokens"] == 100


def test_processor_setup(dl_cfg):
    proc = LogProcessor(dl_cfg)
    kinds = {type(e) for e in proc.extractors}
    assert kinds == {BaseMetadataExtractor, ToolsExtractor, TokenExtractor}


def test_processor_process_sample(eval_log, log_info, dl_cfg):
    proc = LogProcessor(dl_cfg)
    with patch(
        "hibayes.load.load.read_eval_log_sample", return_value=eval_log.samples[0]
    ):
        row = proc.process_sample(eval_log.samples[0], eval_log, log_info)
    assert row["tools"] == ["grep"]
    assert row["score"] == 0.75


def test_processor_unknown_extractor(tmp_path: Path):
    cfg = DataLoaderConfig(
        enabled_extractors=["base", "does‑not‑exist"],
        custom_extractors=[],
        files_to_process=[str(tmp_path)],
    )
    proc = LogProcessor(cfg)
    assert {type(e) for e in proc.extractors} == {BaseMetadataExtractor}


class _FailingExtractor(MagicMock, MetadataExtractor):  # type: ignore[misc]
    """Mock extractor that always raises to test error capture."""

    def extract(self, *_: Any, **__: Any):
        raise RuntimeError("bang")


def test_processor_error_capture(eval_log, log_info, dl_cfg):
    failing = _FailingExtractor(name="Fail")
    cfg = DataLoaderConfig(
        enabled_extractors=["base"],
        custom_extractors=[failing],
        files_to_process=["/logs"],
    )
    proc = LogProcessor(cfg)
    with patch(
        "hibayes.load.load.read_eval_log_sample", return_value=eval_log.samples[0]
    ):
        row = proc.process_sample(eval_log.samples[0], eval_log, log_info)
    assert "processing_errors" in row and "Fail" in row["processing_errors"]


def test_processor_with_multiple_custom_extractors(eval_log, log_info, dl_cfg):
    """Test processor with multiple custom extractors."""

    # Create custom mock extractors
    custom1 = MagicMock(spec=MetadataExtractor)
    custom1.extract.return_value = {"custom1": "value1"}
    custom1.__class__.__name__ = "CustomExtractor1"

    custom2 = MagicMock(spec=MetadataExtractor)
    custom2.extract.return_value = {"custom2": "value2"}
    custom2.__class__.__name__ = "CustomExtractor2"

    # Setup the config with base extractor and custom extractors
    config = DataLoaderConfig(
        enabled_extractors=["base"],  # Assuming "base" is a valid extractor
        custom_extractors=[custom1, custom2],
        files_to_process=["test_dir"],
    )

    proc = LogProcessor(config)

    assert len(proc.extractors) == 3

    with patch(
        "hibayes.load.load.read_eval_log_sample", return_value=eval_log.samples[0]
    ):
        row = proc.process_sample(eval_log.samples[0], eval_log, log_info)

    # Both custom extractors should have been called once
    custom1.extract.assert_called_once()
    custom2.extract.assert_called_once()

    assert "custom1" in row
    assert "custom2" in row
    assert row["custom1"] == "value1"
    assert row["custom2"] == "value2"


def test_is_after_timestamp(eval_log):
    before = datetime.datetime(2024, 4, 1, tzinfo=pytz.UTC)
    after = datetime.datetime(2025, 1, 1, tzinfo=pytz.UTC)
    assert is_after_timestamp(before, eval_log)
    assert not is_after_timestamp(after, eval_log)


def test_is_after_timestamp_tzaware(eval_log):
    cutoff = datetime.datetime(2024, 4, 30, tzinfo=pytz.UTC)
    assert is_after_timestamp(cutoff, eval_log)


def test_get_file_list(tmp_path: Path):
    p1 = tmp_path / "a.eval"
    p1.touch()
    p2 = tmp_path / "b.eval"
    p2.touch()
    manifest = tmp_path / "list.txt"
    manifest.write_text(f"{p1}\n{p2}\n")
    assert get_file_list([str(manifest)]) == [str(p1), str(p2)]


def test_get_file_list_empty():
    assert get_file_list([]) == []


def test_row_generator_flow(dl_cfg):
    proc = LogProcessor(dl_cfg)
    with (
        patch("hibayes.load.load.list_eval_logs", return_value=["foobar"]),
        patch(
            "hibayes.load.load.process_eval_logs_parallel",
            return_value=iter([{"score": 0.75, "model": "gpt‑4o"}]),
        ),
    ):
        rows = list(row_generator(processor=proc, files_to_process=["/fake"]))
    assert rows[0]["score"] == 0.75


def test_row_generator_empty_validation(dl_cfg):
    proc = LogProcessor(dl_cfg)
    with pytest.raises(ValueError):
        next(row_generator(processor=proc, files_to_process=[]))


def test_row_generator_s3_path(dl_cfg):
    proc = LogProcessor(dl_cfg)
    with patch("hibayes.load.load.list_eval_logs", return_value=[]):
        list(row_generator(processor=proc, files_to_process=["s3://bucket/key"]))


def test_row_generator_nonexistent_path(dl_cfg):
    proc = LogProcessor(dl_cfg)
    with (
        patch("os.path.exists", return_value=False),
        patch("hibayes.load.load.list_eval_logs", return_value=[]),
    ):
        rows = list(row_generator(processor=proc, files_to_process=["/nope"]))
    assert rows == []


def test_get_sample_df_from_path_cached(tmp_path: Path, dl_cfg):
    cached = tmp_path / "cached.jsonl"
    cached.write_text("{score: 0.9}\n")
    dl_cfg.cache_path = str(cached)
    with patch("pandas.read_json", return_value=pd.DataFrame([{"score": 0.9}])):
        df = get_sample_df(
            config=dl_cfg,
        )
    assert df.iloc[0]["score"] == 0.9


def test_process_eval_logs_parallel(eval_sample, eval_log):
    info = MagicMock(spec=EvalLogInfo)
    with (
        patch("hibayes.load.load.read_eval_log", return_value=eval_log),
        patch("hibayes.load.load.read_eval_log_sample", return_value=eval_sample),
    ):
        proc = LogProcessor(
            DataLoaderConfig(
                enabled_extractors=["base"],
                custom_extractors=[],
                files_to_process=["/"],
            )
        )
        rows = list(
            process_eval_logs_parallel(eval_logs=[info], processor=proc, cutoff=None)
        )
    assert rows[0]["score"] == 0.75


def test_end_to_end_pipeline(tmp_path: Path, eval_sample, eval_log, log_info, dl_cfg):
    with (
        patch("hibayes.load.load.list_eval_logs", return_value=[log_info]),
        patch("hibayes.load.load.read_eval_log", return_value=eval_log),
        patch("hibayes.load.load.read_eval_log_sample", return_value=eval_sample),
        patch("os.path.exists", return_value=True),
    ):
        df = get_sample_df(config=dl_cfg)
    assert len(df) == 1 and df.loc[0, "score"] == 0.75


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
    df = get_sample_df(config=config)

    # Verify we got some data
    assert not df.empty
    assert "score" in df.columns
    assert "model" in df.columns

    assert "claude-3-7-sonnet-20250219" in df["model"].unique(), "Missing model"
    assert "o3-mini" in df["model"].unique(), "Missing model"

    assert len(df) == 60, "Incorrect number of samples"
