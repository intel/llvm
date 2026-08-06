"""Parse JUnit XML test results."""

import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import defusedxml.ElementTree as ET

from ..constants import TEST_NOT_SELECTED_MSG


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
                file=sys.stderr
            )
            return False
        except (OSError, ValueError) as e:
            print(
                f"Warning: Error reading XML file {self.xml_path}: {e}",
                file=sys.stderr
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


# Standalone function for backward compatibility
def extract_tests_from_xml(xml_path: str) -> ParsedXMLTests:
    parser = JUnitXMLParser(xml_path)
    return parser.extract_tests_from_xml()
