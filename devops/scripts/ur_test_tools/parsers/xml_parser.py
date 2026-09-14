"""Parse JUnit XML test results."""

import sys
from pathlib import Path
from typing import Optional

import defusedxml.ElementTree as ET

from ..constants import TEST_NOT_SELECTED_MSG
from ..models.test_results import TestStatus
from .parser_models import ParsedXMLData, ParsedTestObservation


def _format_test_name(classname: str, name: str) -> str:
    """Convert a JUnit name to LIT's display format."""
    if not classname:
        return name
    if not name:
        return classname

    suite_name, separator, path = classname.partition(".")
    if not separator or path == suite_name:
        return f"{suite_name} :: {name}"
    return f"{suite_name} :: {path}/{name}"


def _status_from_skipped_message(message: str) -> TestStatus:
    """Recover the LIT status encoded as a JUnit ``skipped`` element."""
    if message.startswith(TEST_NOT_SELECTED_MSG):
        return TestStatus.EXCLUDED
    if message == "User interrupt":
        return TestStatus.SKIPPED
    if message.startswith("Missing required feature(s):"):
        return TestStatus.UNSUPPORTED
    if message == "Unsupported configuration":
        return TestStatus.UNSUPPORTED

    return TestStatus.SKIPPED


class JUnitXMLParser:
    """Parse JUnit XML from LIT --xunit-xml-output."""

    def __init__(self, xml_path: Optional[str]):
        self.xml_path = xml_path
        self._root = None

    def parse(self) -> bool:
        if not self.xml_path:
            return False

        xml_file = Path(self.xml_path)
        if not xml_file.exists():
            return False

        try:
            tree = ET.parse(self.xml_path)
            self._root = tree.getroot()
            return True
        except ET.ParseError as e:
            print(
                f"Warning: Failed to parse XML file {self.xml_path}: {e}",
                file=sys.stderr,
            )
            return False
        except (OSError, ValueError) as e:
            print(
                f"Warning: Error reading XML file {self.xml_path}: {e}", file=sys.stderr
            )
            return False

    def parse_to_observations(self) -> ParsedXMLData:
        """Parse XML test results."""
        if not self.parse():
            return ParsedXMLData(tests=[])

        observations = []
        total_tests = 0
        total_time = 0.0

        for testcase in self._root.findall(".//testcase"):
            test_name = _format_test_name(
                testcase.get("classname", ""), testcase.get("name", "")
            )
            if not test_name:
                continue

            total_tests += 1

            duration_ms = None
            time_str = testcase.get("time")
            if time_str:
                try:
                    duration_sec = float(time_str)
                    duration_ms = duration_sec * 1000.0
                    total_time += duration_sec
                except ValueError:
                    pass

            status = self._determine_status_from_xml(testcase)

            observations.append(
                ParsedTestObservation(
                    name=test_name,
                    status=status,
                    duration_ms=duration_ms,
                )
            )

        return ParsedXMLData(
            tests=observations,
            total_tests=total_tests,
            total_time_seconds=total_time if total_time > 0 else None,
        )

    def _determine_status_from_xml(self, testcase) -> TestStatus:
        if testcase.find("failure") is not None:
            return TestStatus.FAIL

        error_elem = testcase.find("error")
        if error_elem is not None:
            message = error_elem.get("message", "").lower()
            if "timeout" in message:
                return TestStatus.TIMEOUT
            return TestStatus.UNRESOLVED

        skipped_elem = testcase.find("skipped")
        if skipped_elem is not None:
            return _status_from_skipped_message(skipped_elem.get("message", ""))

        # JUnit does not distinguish successful LIT statuses.
        return TestStatus.PASS
