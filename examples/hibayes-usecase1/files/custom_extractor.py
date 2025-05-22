from typing import Any, Dict

from inspect_ai.log import EvalLog, EvalSample

from hibayes.load import MetadataExtractor

DOMAINS = {
    "mbpp": "coding",
    "DS-1000": "coding",
    "boolq": "reasoning",
    "race-h": "reasoning",
}
SUB_DOMAINS = {
    "mbpp": "easy",
    "DS-1000": "hard",
    "boolq": "easy",
    "race-h": "hard",
}


class Domains(MetadataExtractor):
    def extract(self, sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        return {
            "domain": DOMAINS.get(eval_log.eval.task, "other"),
            "sub_domain": SUB_DOMAINS.get(eval_log.eval.task, "other"),
        }
