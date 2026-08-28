"""Parse JUnit XML test results."""

import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import defusedxml.ElementTree as ET

from ..constants import TEST_NOT_SELECTED_MSG
from ..models.test_results import TestStatus
from .parser_models import ParsedXMLData, ParsedTestObservation


class ParsedXMLTests(NamedTuple):
    """Skipped and excluded tests from XML parsing."""

    skipped: List[str]
    excluded: List[str]


def _format_test_name(classname: str, name: str) -> str:
    if classname and name:
        return f"{classname}.{name}"
    return name


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

    def extract_tests_from_xml(self) -> ParsedXMLTests:
        if not self.parse():
            return ParsedXMLTests([], [])

        skipped = []
        excluded = []

        for testcase in self._root.findall(".//testcase"):
            skipped_elem = testcase.find("skipped")
            if skipped_elem is None:
                continue

            message = skipped_elem.get("message", "")
            test_name = _format_test_name(
                testcase.get("classname", ""), testcase.get("name", "")
            )

            if not test_name:
                continue

            # Separate by message type
            if TEST_NOT_SELECTED_MSG in message:
                excluded.append(test_name)
            else:
                skipped.append(test_name)

        return ParsedXMLTests(skipped=skipped, excluded=excluded)

    def extract_skipped_tests(self) -> List[str]:
        skipped, _ = self.extract_tests_from_xml()
        return skipped

    def extract_excluded_tests(self) -> List[str]:
        _, excluded = self.extract_tests_from_xml()
        return excluded

    def parse_to_observations(self) -> ParsedXMLData:
        """Parse XML into normalized observations.

        Returns:
            ParsedXMLData with test observations and metadata.
        """
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

            # Extract duration (in seconds from XML, convert to ms)
            duration_ms = None
            time_str = testcase.get("time")
            if time_str:
                try:
                    duration_sec = float(time_str)
                    duration_ms = duration_sec * 1000.0
                    total_time += duration_sec
                except ValueError:
                    pass

            # Determine status from child elements
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
        """Determine test status from XML testcase element.

        JUnit XML structure:
        - <failure> = FAIL
        - <error> = UNRESOLVED or TIMEOUT (depends on message)
        - <skipped message="..."> = SKIPPED or EXCLUDED (depends on message)
        - No child elements = PASS
        """
        # Check for failure
        if testcase.find("failure") is not None:
            return TestStatus.FAIL

        # Check for error (could be unresolved or timeout)
        error_elem = testcase.find("error")
        if error_elem is not None:
            message = error_elem.get("message", "").lower()
            if "timeout" in message:
                return TestStatus.TIMEOUT
            return TestStatus.UNRESOLVED

        # Check for skipped (could be skipped or excluded)
        skipped_elem = testcase.find("skipped")
        if skipped_elem is not None:
            message = skipped_elem.get("message", "")
            if TEST_NOT_SELECTED_MSG in message:
                return TestStatus.EXCLUDED
            return TestStatus.SKIPPED

        # No special markers = passed
        # Note: We can't distinguish PASS/FLAKYPASS/XFAIL/XPASS from XML alone
        # That requires log output. XML parser reports PASS for anything that succeeded.
        return TestStatus.PASS
