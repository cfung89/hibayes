from typing import Any, Dict

from hibayes.load import MetadataExtractor
from inspect_ai.log import EvalLog


class GaiaMetadataExtractor(MetadataExtractor):
    def extract(self, sample: Any, eval_log: EvalLog) -> Dict[str, Any]:
        annotator_metadata = sample.metadata.get("Annotator Metadata", {})
        return {
            "level": sample.metadata["level"],
            "time_to_complete": annotator_metadata.get("How long did this take?", None),
            "number_steps": annotator_metadata.get("Number of steps", None),
        }
