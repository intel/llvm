import lit.Test
import lit.util
import lit.formats

import os
import re


class SYCLHeadersTest(lit.formats.TestFormat):
    def getTestsForPath(
        self, testSuite, path_in_suite, filepath, litConfig, localConfig
    ):
        yield lit.Test.Test(testSuite, path_in_suite, localConfig, file_path=filepath)

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        # We traverse build/sycl/include/sycl directory
        source_path = os.path.join(localConfig.sycl_include, "sycl")

        # Optional filter can be passed through command line options
        headers_filter = localConfig.sycl_headers_filter
        for dirpath, _, filenames in os.walk(source_path):
            relative_dirpath = dirpath[len(localConfig.sycl_include) + 1 :]
            for filename in filenames:
                if not filename.endswith(".hpp"):
                    continue
                filepath = os.path.join(dirpath, filename)

                if headers_filter is not None:
                    # Skip headers that doesn't match passed regexp
                    if re.search(headers_filter, filepath) is None:
                        continue
                for t in self.getTestsForPath(
                    testSuite,
                    path_in_suite
                    + (
                        relative_dirpath,
                        filename,
                    ),
                    filepath,
                    litConfig,
                    localConfig,
                ):
                    yield t

    def execute(self, test, litConfig):
        command = [
            test.config.clang,
            "-fsycl",
            "-fsyntax-only",
            "-include",
            test.file_path,
            os.path.join(
                test.suite.getSourcePath(("self-contained-headers",)),
                "Inputs",
                "test.cpp",
            ),
        ]

        is_xfail = False
        for path in test.config.sycl_headers_xfail:
            if test.file_path.endswith(path):
                is_xfail = True
                break

        try:
            out, err, exitCode = lit.util.executeCommand(
                command,
                # TODO: do we need to pass some non-default cwd argument here?
                env=test.config.environment,
                timeout=litConfig.maxIndividualTestTime,
            )
            if is_xfail:
                status = lit.Test.XPASS if exitCode == 0 else lit.Test.XFAIL
            else:
                status = lit.Test.PASS if exitCode == 0 else lit.Test.FAIL
            timeoutInfo = None
        except lit.util.ExecuteCommandTimeoutException as e:
            out, err, exitCode, timeoutInfo = e.out, e.err, e.exitCode, e.msg
            status = lit.Test.TIMEOUT

        commandStr = " ".join(command)
        output = f"Running command: {commandStr}\n"
        output += f"Exit Code: {exitCode}\n"
        if timeoutInfo is not None:
            output += f"Timeout: {timeoutInfo}\n"
        if is_xfail:
            output += "This test is marked as XFAIL in sycl/test/format.py\n"
        output += "\n"

        # Append the outputs, if present.
        if out:
            output += f"Command Output (stdout):\n--\n{err}\n--\n"
        if err:
            output += f"Command Output (stderr):\n--\n{err}\n--\n"

        return lit.Test.Result(status, output)
