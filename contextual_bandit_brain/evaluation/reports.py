from __future__ import annotations
from typing import Dict
from ..reporting.report_generator import write_json_report, write_text_summary


def write_reports(out_dir: str, bis_score: float, metrics: Dict[str, float]) -> Dict[str, str]:
    jp = write_json_report(out_dir, bis_score, metrics)
    tp = write_text_summary(out_dir, bis_score, metrics)
    return {"json": jp, "text": tp}
