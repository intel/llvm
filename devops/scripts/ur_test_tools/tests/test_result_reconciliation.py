"""Tests for parsing and reconciling LIT test results."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from devops.scripts.ur_test_tools.models.config import SummaryConfigFromLines
from devops.scripts.ur_test_tools.models.test_results import TestStatus
from devops.scripts.ur_test_tools.parsers.parser_models import (
    ParsedLogData,
    ParsedTestObservation,
    ParsedXMLData,
)
from devops.scripts.ur_test_tools.parsers.log_parser import LITLogParser
from devops.scripts.ur_test_tools.parsers.stats_parser import parse_statistics
from devops.scripts.ur_test_tools.parsers.xml_parser import JUnitXMLParser
from devops.scripts.ur_test_tools.reconciliation import (
    build_test_run_result,
    reconcile_test_results,
)
from devops.scripts.ur_test_tools.summary_generator import SummaryReporter


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


class LITLogParserTest(unittest.TestCase):
    def test_parses_all_builtin_statuses(self):
        categories = [
            ("Passed", TestStatus.PASS),
            ("Passed With Retry", TestStatus.FLAKYPASS),
            ("Passed After Update", TestStatus.FIXED),
            ("Expectedly Failed", TestStatus.XFAIL),
            ("Unexpectedly Passed", TestStatus.XPASS),
            ("Failed", TestStatus.FAIL),
            ("Unresolved", TestStatus.UNRESOLVED),
            ("Unsupported", TestStatus.UNSUPPORTED),
            ("Timed Out", TestStatus.TIMEOUT),
            ("Skipped", TestStatus.SKIPPED),
            ("Excluded", TestStatus.EXCLUDED),
        ]
        lines = []
        for index, (label, _) in enumerate(categories):
            lines.extend([f"{label} Tests (1):\n", f"suite :: test-{index}\n", "\n"])

        result = LITLogParser(lines).parse_to_observations()

        self.assertEqual(
            [test.status for test in result.tests],
            [status for _, status in categories],
        )
        self.assertEqual(
            result.declared_counts,
            {status: 1 for _, status in categories},
        )

    def test_warns_about_unknown_category(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = LITLogParser(
                ["New Fancy Tests (1):\n", "suite :: test\n", "\n"]
            ).parse_to_observations()

        self.assertEqual(result.tests, [])
        self.assertIn("Unknown test category 'New Fancy'", stderr.getvalue())

    def test_parses_statistics_by_exact_label(self):
        result = parse_statistics(
            [
                "Total Discovered Tests: 5\n",
                "  Passed: 2\n",
                "  Unexpectedly Passed: 3\n",
            ]
        )

        self.assertEqual(result["Total Discovered"], 5)
        self.assertEqual(result["Passed"], 2)
        self.assertEqual(result["Unexpectedly Passed"], 3)

    def test_sums_repeated_category_counts(self):
        lines = [
            "Passed Tests (1):\n",
            "suite :: first\n",
            "\n",
            "Passed Tests (1):\n",
            "suite :: second\n",
            "\n",
        ]

        result = LITLogParser(lines).parse_to_observations()

        self.assertEqual(result.declared_counts[TestStatus.PASS], 2)
        self.assertEqual(len(result.tests), 2)


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
                ParsedTestObservation("Suite.unsupported.cpp", TestStatus.UNSUPPORTED),
                ParsedTestObservation("Suite.skipped.cpp", TestStatus.SKIPPED),
                ParsedTestObservation("Suite.excluded.cpp", TestStatus.EXCLUDED),
            ]
        )

        result = reconcile_test_results(log_data, xml_data)

        self.assertEqual(result.count_by_status(TestStatus.PASS), 1)
        self.assertEqual(result.count_by_status(TestStatus.UNSUPPORTED), 1)
        self.assertEqual(result.count_by_status(TestStatus.SKIPPED), 1)
        self.assertEqual(result.count_by_status(TestStatus.EXCLUDED), 1)

    def test_xml_does_not_guess_ambiguous_status(self):
        log_data = ParsedLogData(tests=[], declared_counts={TestStatus.PASS: 1})
        xml_data = ParsedXMLData(
            tests=[ParsedTestObservation("Suite :: pass.cpp", TestStatus.PASS)]
        )

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = reconcile_test_results(log_data, xml_data)

        self.assertEqual(result.tests, [])
        self.assertIn("Count mismatch for PASS", stderr.getvalue())

    def test_xml_fills_in_unambiguous_status(self):
        xml_data = ParsedXMLData(
            tests=[
                ParsedTestObservation("Suite :: unsupported", TestStatus.UNSUPPORTED)
            ]
        )

        result = reconcile_test_results(ParsedLogData(), xml_data)

        self.assertEqual(result.count_by_status(TestStatus.UNSUPPORTED), 1)

    def test_log_status_takes_precedence_and_xml_provides_duration(self):
        statuses = [
            TestStatus.XFAIL,
            TestStatus.XPASS,
            TestStatus.FLAKYPASS,
            TestStatus.FIXED,
        ]
        log_data = ParsedLogData(
            tests=[
                ParsedTestObservation(f"Suite :: test-{index}", status)
                for index, status in enumerate(statuses)
            ],
            declared_counts={status: 1 for status in statuses},
        )
        xml_data = ParsedXMLData(
            tests=[
                ParsedTestObservation(
                    f"Suite :: test-{index}", TestStatus.PASS, 123.4 + index
                )
                for index in range(len(statuses))
            ]
        )

        result = reconcile_test_results(log_data, xml_data)

        self.assertEqual([test.status for test in result.tests], statuses)
        self.assertEqual(result.tests[0].duration_ms, 123.4)

    def test_warns_about_statistics_count_mismatch(self):
        log_data = ParsedLogData(
            tests=[ParsedTestObservation("Suite :: pass", TestStatus.PASS)],
            statistics={"Passed": 2},
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            reconcile_test_results(log_data)

        self.assertIn(
            "Count mismatch for PASS: statistics 2, found 1", stderr.getvalue()
        )

    def test_warns_about_unknown_status_statistic(self):
        log_data = ParsedLogData(statistics={"New Fancy": 1})
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            reconcile_test_results(log_data)

        self.assertIn("Unknown test statistic 'New Fancy'", stderr.getvalue())


class SummaryReporterTest(unittest.TestCase):
    def test_displays_skipped_and_unsupported_separately(self):
        lines = [
            "Skipped Tests (2):\n",
            "suite :: skipped-1\n",
            "suite :: skipped-2\n",
            "\n",
            "Unsupported Tests (3):\n",
            "suite :: unsupported-1\n",
            "suite :: unsupported-2\n",
            "suite :: unsupported-3\n",
            "\n",
            "Total Discovered Tests: 5\n",
            "  Skipped: 2\n",
            "  Unsupported: 3\n",
            "Testing Time: 1.25s\n",
        ]
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            result = build_test_run_result(SummaryConfigFromLines(log_lines=lines))
            SummaryReporter(result).generate()

        output = stdout.getvalue()
        self.assertIn("::group::Skipped (2)", output)
        self.assertIn("::group::Unsupported (3)", output)
        self.assertIn("  Skipped: 2", output)
        self.assertIn("  Unsupported: 3", output)
        self.assertIn("Testing Time: 1.25s", output)


if __name__ == "__main__":
    unittest.main()
