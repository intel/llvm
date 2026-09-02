"""Orchestrate parsers and reconciliation into a TestRunResult."""

from .models.config import SummaryConfigFromLines
from .models.test_results import TestRunResult
from .parsers.log_parser import LITLogParser
from .parsers.xml_parser import JUnitXMLParser
from .reconciliation import reconcile_test_results


def build_test_run_result(config: SummaryConfigFromLines) -> TestRunResult:
    log_data = LITLogParser(config.log_lines).parse_to_observations()
    xml_data = None
    if config.xml_file:
        xml_data = JUnitXMLParser(config.xml_file).parse_to_observations()
    return reconcile_test_results(log_data, xml_data)
