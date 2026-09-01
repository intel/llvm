"""Tests for parsing and reconciling LIT test results."""

import tempfile
import unittest
from pathlib import Path

from devops.scripts.ur_test_tools.models.test_results import TestStatus
from devops.scripts.ur_test_tools.parsers.parser_models import (
    ParsedLogData,
    ParsedTestObservation,
    ParsedXMLData,
)
from devops.scripts.ur_test_tools.parsers.xml_parser import JUnitXMLParser
from devops.scripts.ur_test_tools.reconciliation import reconcile_test_results


class JUnitXMLParserTest(unittest.TestCase):
    def test_recovers_lit_statuses_from_skipped_messages(self):
        xml = """\
<testsuites>
  <testsuite name="suite" tests="5">
    <testcase classname="Suite.path" name="excluded.cpp">
      <skipped message="Test not selected (--filter, --max-tests)"/>
    </testcase>
    <testcase classname="Suite.path" name="interrupted.cpp">
      <skipped message="User interrupt"/>
    </testcase>
    <testcase classname="Suite.path" name="missing-feature.cpp">
      <skipped message="Missing required feature(s): gpu"/>
    </testcase>
    <testcase classname="Suite.path" name="unsupported.cpp">
      <skipped message="Unsupported configuration"/>
    </testcase>
    <testcase classname="Suite.path" name="framework-skipped.cpp">
      <skipped message="Disabled by test framework"/>
    </testcase>
  </testsuite>
</testsuites>
"""
        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "results.xml"
            xml_path.write_text(xml, encoding="utf-8")
            result = JUnitXMLParser(str(xml_path)).parse_to_observations()

        self.assertEqual(
            [test.status for test in result.tests],
            [
                TestStatus.EXCLUDED,
                TestStatus.SKIPPED,
                TestStatus.UNSUPPORTED,
                TestStatus.UNSUPPORTED,
                TestStatus.SKIPPED,
            ],
        )
        self.assertEqual(result.tests[0].name, "Suite :: path/excluded.cpp")

    def test_formats_test_at_suite_root(self):
        xml = """\
<testsuites>
  <testsuite name="Suite" tests="1">
    <testcase classname="Suite.Suite" name="root.cpp"/>
  </testsuite>
</testsuites>
"""
        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "results.xml"
            xml_path.write_text(xml, encoding="utf-8")
            result = JUnitXMLParser(str(xml_path)).parse_to_observations()

        self.assertEqual(result.tests[0].name, "Suite :: root.cpp")


class ReconcileTestResultsTest(unittest.TestCase):
    def test_complete_log_category_takes_precedence_over_xml(self):
        log_data = ParsedLogData(
            tests=[
                ParsedTestObservation("Suite :: pass.cpp", TestStatus.PASS),
                ParsedTestObservation(
                    "Suite :: unsupported.cpp", TestStatus.UNSUPPORTED
                ),
            ],
            declared_counts={TestStatus.PASS: 1, TestStatus.UNSUPPORTED: 1},
        )
        xml_data = ParsedXMLData(
            tests=[
                ParsedTestObservation("Suite.pass.cpp", TestStatus.PASS),
                ParsedTestObservation(
                    "Suite.unsupported.cpp", TestStatus.UNSUPPORTED
                ),
                ParsedTestObservation("Suite.skipped.cpp", TestStatus.SKIPPED),
                ParsedTestObservation("Suite.excluded.cpp", TestStatus.EXCLUDED),
            ]
        )

        result = reconcile_test_results(log_data, xml_data)

        self.assertEqual(result.count_by_status(TestStatus.PASS), 1)
        self.assertEqual(result.count_by_status(TestStatus.UNSUPPORTED), 1)
        self.assertEqual(result.count_by_status(TestStatus.SKIPPED), 1)
        self.assertEqual(result.count_by_status(TestStatus.EXCLUDED), 1)

    def test_xml_fills_in_incomplete_log_category(self):
        log_data = ParsedLogData(
            tests=[], declared_counts={TestStatus.PASS: 1}
        )
        xml_data = ParsedXMLData(
            tests=[ParsedTestObservation("Suite :: pass.cpp", TestStatus.PASS)]
        )

        result = reconcile_test_results(log_data, xml_data)

        self.assertEqual(result.count_by_status(TestStatus.PASS), 1)


if __name__ == "__main__":
    unittest.main()