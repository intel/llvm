import lit.Test
import lit.util
import lit.formats

import os
import re

class SYCLHeadersTest(lit.formats.TestFormat):
    def getTestsForPath(
        self, testSuite, path_in_suite, filepath, litConfig, localConfig
    ):
        # path_in_suite is a tuple like:
        #   ('self-contained-headers', 'path/to', 'file.hpp')
        test_path = testSuite.getSourcePath(path_in_suite) + ".cpp"
        if os.path.exists(test_path):
            # We have a dedicated special test for a header, let's use a file
            # from the suite itself

            # None is a special value we use to distinguish those two cases
            filepath = None
            # The actual file has .cpp extension as every other test
            path_in_suite = path_in_suite[:-1] + (path_in_suite[-1] + ".cpp",)
        else:
            # We don't have a dedicated special test for a header, therefore we
            # fallback to a generalized version of it

            # SYCL headers may depend on some generated files and therefore we
            # use headers from the build folder for testing
            filepath = os.path.join(localConfig.sycl_include, *path_in_suite[1:])

        yield lit.Test.Test(testSuite, path_in_suite, localConfig, file_path=filepath)

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        # To respect SYCL_LIB_DUMPS_ONLY mode
        if ".cpp" not in localConfig.suffixes:
            return

        # As we add more files and folders into 'self-contained-headers', this
        # method will be recursivelly called for them by lit discovery.
        # However, we don't use the test folder as the source of tests but
        # instead we use SYCL_SOURCE_DIR/include/sycl directory.
        # Therefore, we exit early from here if `path_in_suite` conatins more
        # than one element
        assert path_in_suite[0] == "self-contained-headers"
        if len(path_in_suite) > 1:
            return

        source_path = os.path.join(localConfig.sycl_include_source_dir, "sycl")

        # Optional filter can be passed through command line options
        headers_filter = localConfig.sycl_headers_filter
        for dirpath, _, filenames in os.walk(source_path):
            relative_dirpath = dirpath[len(localConfig.sycl_include_source_dir) + 1 :]
            for filename in filenames:
                suffix = os.path.splitext(filename)[1]
                # We only look at actual header files and not at their .inc/.def
                # components
                if suffix != ".hpp":
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
        if test.file_path is None:
            # It means that we have a special test case for a header and we need
            # to execute it as a regular lit sh test
            return lit.TestRunner.executeShTest(
                test,
                litConfig,
                False,  # execute_external
                [],  # extra_substitutions
                [],  # preamble_commands
            )

        # Otherwise we generate the test on the fly
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
