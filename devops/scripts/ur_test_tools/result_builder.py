"""Orchestrate parsers and reconciliation into a TestRunResult."""

from typing import List, Optional

from .models.test_results import TestRunResult
from .parsers.log_parser import LITLogParser
from .parsers.xml_parser import JUnitXMLParser
from .reconciliation import reconcile_test_results


def build_test_run_result(
    log_lines: List[str], xml_file: Optional[str] = None
) -> TestRunResult:
    log_data = LITLogParser(log_lines).parse_to_observations()
    xml_data = None
    if xml_file:
        xml_data = JUnitXMLParser(xml_file).parse_to_observations()
    return reconcile_test_results(log_data, xml_data)
