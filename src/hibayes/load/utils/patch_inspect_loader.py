# patch inspect to extract sample ids when reading the header.
import argparse
import datetime
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
)
from zipfile import ZipFile

import ijson
import inspect_ai.log._recorders.eval
import pandas as pd
import pytz
import yaml
from ijson import IncompleteJSONError
from inspect_ai._util.constants import LOG_SCHEMA_VERSION
from inspect_ai._util.file import file, filesystem
from inspect_ai.log import (
    EvalError,
    EvalLog,
    EvalLogInfo,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSpec,
    EvalStats,
    list_eval_logs,
    read_eval_log,
    read_eval_log_sample,
)
from inspect_ai.log._recorders.create import _recorders
from inspect_ai.log._recorders.eval import (
    EvalRecorder,  # main recorder for .eval logs (current inspect default)
    EvalSampleReductions,
    _read_header,
    sort_samples,
)
from inspect_ai.log._recorders.json import (
    JSONRecorder,  # main recorder for .json logs (current inspect default)
    _validate_version,
)
from inspect_ai.log._recorders.recorder import Recorder
from pydantic import BaseModel
from pydantic_core import from_json
from typing_extensions import override

# We need to access the SAMPLES_DIR constant from the same module
SAMPLES_DIR = inspect_ai.log._recorders.eval.SAMPLES_DIR
REDUCTIONS_JSON = inspect_ai.log._recorders.eval.REDUCTIONS_JSON


def update_inspect_recorders():
    """Update inspect's recorder registry with patched versions for tracking sample ids."""

    for recorder_key in _recorders.keys():
        try:
            if recorder_key == "eval":
                patch_recorder = PatchedEvalRecorder
            elif recorder_key == "json":
                patch_recorder = PatchedJSONRecorder
            else:
                continue

            _recorders[recorder_key] = patch_recorder

        except KeyError:
            logging.error(
                f"Recorder {recorder_key} not found in inspect's recorder registry"
            )
        except Exception as e:
            logging.error(f"Error updating recorder {recorder_key}: {e}")


class LogSample(BaseModel):
    id: int | str
    epoch: int


# update eval recorder
class PatchedEvalRecorder(EvalRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @override
    async def read_log(cls, location: str, header_only: bool = False) -> EvalLog:
        # if the log is not stored in the local filesystem then download it first,
        # and then read it from a temp file (eliminates the possiblity of hundreds
        # of small fetches from the zip file streams)
        temp_log: str | None = None
        fs = filesystem(location)
        if not fs.is_local():  # see here still does this even if header_only is True
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp_log = temp.name
                fs.get_file(location, temp_log)

        # read log (use temp_log if we have it)
        try:
            with file(temp_log or location, "rb") as z:
                return patched_read_log(z, location, header_only)
        finally:
            if temp_log:
                os.unlink(temp_log)


def read_sample_header(zip: ZipFile, sample: str) -> LogSample:
    with zip.open(sample) as f:
        header = {}
        # Only parse until we find the fields we need
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if prefix == "id" and (event == "string" or event == "number"):
                header["id"] = value
            elif prefix == "epoch" and event == "number":
                header["epoch"] = value

            # Exit early once we have both values
            if len(header) == 2:
                break

        return LogSample(**header)


def patched_read_log(
    log: BinaryIO, location: str, header_only: bool = False
) -> EvalLog:
    start_time = time.time()

    with ZipFile(log, mode="r") as zip:
        evalLog = _read_header(zip, location)
        if REDUCTIONS_JSON in zip.namelist():
            with zip.open(REDUCTIONS_JSON, "r") as f:
                reductions = [
                    EvalSampleReductions.model_validate(reduction)
                    for reduction in json.load(f)
                ]
                if evalLog.results is not None:
                    evalLog.reductions = reductions

        samples: list[EvalSample] | None = None
        if not header_only:
            samples = []
            for name in zip.namelist():
                if name.startswith(f"{SAMPLES_DIR}/") and name.endswith(".json"):
                    with zip.open(name, "r") as f:
                        samples.append(
                            EvalSample.model_validate(json.load(f)),
                        )

            sort_samples(samples)
            evalLog.samples = samples
        else:
            samples = []
            for name in zip.namelist():
                if name.startswith(f"{SAMPLES_DIR}/") and name.endswith(".json"):
                    samples.append(read_sample_header(zip, name))

            # Add this information to the eval_log object
            evalLog.samples = samples
    return evalLog


# update json recorder


class PatchedJSONRecorder(JSONRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.info(
            "PatchedJSONRecorder is being used. This is an old log format and currently a very"
            " slow way to read logs. Please consider using the 'eval' log format instead."
        )

    @override
    @classmethod
    async def read_log(cls, location: str, header_only: bool = False) -> EvalLog:
        if header_only:
            try:
                return _patched_read_header_streaming(location)
            # The Python JSON serializer supports NaN and Inf, however
            # this isn't technically part of the JSON spec. The json-stream
            # library shares this limitation, so if we fail with an
            # invalid character then we move on and and parse w/ pydantic
            # (which does support NaN and Inf by default)
            except (ValueError, IncompleteJSONError) as ex:
                if (
                    str(ex).find("Invalid JSON character") != -1
                    or str(ex).find("invalid char in json text") != -1
                ):
                    pass
                else:
                    raise ValueError(f"Unable to read log file: {location}") from ex

        # full reads (and fallback to streaing reads if they encounter invalid json characters)
        with file(location, "r") as f:
            # parse w/ pydantic
            raw_data = from_json(f.read())
            log = EvalLog.model_validate(raw_data)
            log.location = location

            # fail for unknown version
            _validate_version(log.version)

            # set the version to the schema version we'll be returning
            log.version = LOG_SCHEMA_VERSION

            # prune if header_only
            if header_only:
                # exclude samples
                log.samples = None

                # prune sample reductions
                if log.results is not None:
                    log.results.sample_reductions = None
                    log.reductions = None

            # return log
            return log


def _get_sample_headers(samples: List[Dict[str, Any]]) -> List[LogSample]:
    sample_headers = []
    for sample in samples:
        sample_headers.append(LogSample(id=sample["id"], epoch=sample["epoch"]))
    return sample_headers


def _patched_read_header_streaming(log_file: str) -> EvalLog:
    with file(log_file, "rb") as f:
        # Do low-level parsing to get the version number and also
        # detect the presence of results or error sections
        version: int | None = None
        has_results = False
        has_error = False

        for prefix, event, value in ijson.parse(f):
            if (prefix, event) == ("version", "number"):
                version = value
            elif (prefix, event) == ("results", "start_map"):
                has_results = True
            elif (prefix, event) == ("error", "start_map"):
                has_error = True
            elif prefix == "samples":
                # we log the sample headers later.
                break

        if version is None:
            raise ValueError("Unable to read version of log format.")

        _validate_version(version)
        version = LOG_SCHEMA_VERSION

        # Rewind the file to the beginning to re-parse the contents of fields
        f.seek(0)

        # Parse the log file, stopping before parsing samples
        status: Literal["started", "success", "cancelled", "error"] | None = None
        eval: EvalSpec | None = None
        plan: EvalPlan | None = None
        results: EvalResults | None = None
        stats: EvalStats | None = None
        error: EvalError | None = None
        sample_headers: List[LogSample] | None = None
        for k, v in ijson.kvitems(f, ""):
            if k == "status":
                assert v in get_args(
                    Literal["started", "success", "cancelled", "error"]
                )
                status = v
            if k == "eval":
                eval = EvalSpec(**v)
            elif k == "plan":
                plan = EvalPlan(**v)
            elif k == "results":
                results = EvalResults(**v)
            elif k == "stats":
                stats = EvalStats(**v)
            elif k == "error":
                error = EvalError(**v)
            elif k == "samples":
                sample_headers = _get_sample_headers(v)

    assert status, "Must encounter a 'status'"
    assert eval, "Must encounter a 'eval'"
    assert plan, "Must encounter a 'plan'"
    assert stats, "Must encounter a 'stats'"
    assert sample_headers, "Must encounter a 'samples'"

    try:
        eval_log = EvalLog(
            eval=eval,
            plan=plan,
            results=results if has_results else None,
            stats=stats,
            status=status,
            version=version,
            error=error if has_error else None,
            location=log_file,
        )

        eval_log.samples = sample_headers
    except Exception as e:
        logging.error(f"Error creating EvalLog object: {e}")

    return eval_log
