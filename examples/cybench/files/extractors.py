from typing import Any, Dict

from hibayes.load import MetadataExtractor
from inspect_ai.log import EvalLog


class LogFileNameExtractor(MetadataExtractor):
    def extract(self, sample: Any, eval_log: EvalLog) -> Dict[str, Any]:
        setup = eval_log.location.split("/")[-1].split(".")[0]
        
        return {"setup": setup}
