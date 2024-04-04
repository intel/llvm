import lit.Test
import lit.util
import lit.formats

import os

class CustomFormat(lit.formats.TestFormat):
    def getTestsForPath(self, testSuite, path_in_suite, filepath, litConfig, localConfig):
        yield lit.Test.Test(testSuite, path_in_suite, localConfig, file_path=filepath)

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        # We traverse build/sycl/include/sycl directory
        source_path = os.path.join(localConfig.sycl_include, 'sycl')
        for dirpath, _, filenames in os.walk(source_path):
            relative_dirpath = dirpath[len(localConfig.sycl_include) + 1:]
            for filename in filenames:
                if not filename.endswith('.hpp'):
                    continue
                filepath = os.path.join(dirpath, filename)
                for t in self.getTestsForPath(testSuite, path_in_suite + (relative_dirpath, filename,), filepath, litConfig, localConfig):
                    yield t

    def execute(self, test, litConfig):
        command = [test.config.clang, '-fsycl', '-fsyntax-only', '-include', test.file_path, os.path.join(test.suite.getSourcePath(('self-contained-headers')), 'Inputs', 'test.cpp')]
        try:
            out, err, exitCode = lit.util.executeCommand(
                command,
                # cwd=???,
                env=test.config.environment,
                timeout=litConfig.maxIndividualTestTime,
                )
            status = lit.Test.PASS if exitCode == 0 else lit.Test.FAIL
            timeoutInfo = None
        except lit.util.ExecuteCommandTimeoutException as e:
            out, err, exitCode, timeoutInfo = e.out, e.err, e.exitCode, e.msg
            status = lit.Test.TIMEOUT

        output = """Running command: %s\n""" % (' '.join(command),)
        output += f"Exit Code: {exitCode}\n"
        if timeoutInfo is not None:
            output += """Timeout: %s\n""" % (timeoutInfo,)
        output += "\n"

        # Append the outputs, if present.
        if out:
            output += """Command Output (stdout):\n--\n%s\n--\n""" % (out,)
        if err:
            output += """Command Output (stderr):\n--\n%s\n--\n""" % (err,)

        return lit.Test.Result(status, output)

