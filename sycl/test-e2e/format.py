import lit
import lit.formats

class SYCLEndToEndTest(lit.formats.ShTest):
    def execute(self, test, litConfig):
        filename = test.path_in_suite[-1]
        tmpDir, tmpBase = lit.TestRunner.getTempPaths(test)
        script = lit.TestRunner.parseIntegratedTestScript(test, require_script=True)
        if isinstance(script, lit.Test.Result):
            return script

        substitutions = lit.TestRunner.getDefaultSubstitutions(test, tmpDir, tmpBase)
        # -fsycl-targets is needed for CUDA/HIP, so just use it be default so
        # -that new tests by default would runnable there (unless they have
        # -other restrictions).
        substitutions.append(('%{build}', '%clangxx -fsycl -fsycl-targets=%sycl_triple %s'))

        devices_for_test = ['{}:{}'.format(test.config.sycl_be, dev)
                            for dev in test.config.target_devices.split(',')]

        new_script = []
        for directive in script:
            if not isinstance(directive, lit.TestRunner.CommandDirective):
                new_script.append(directive)
                continue

            if '%{run}' not in directive.command:
                new_script.append(directive)
                continue

            for sycl_device in devices_for_test:
                cmd = directive.command.replace(
                    '%{run}',
                    'env ONEAPI_DEVICE_SELECTOR={} {}'.format(sycl_device, test.config.run_launcher))

                new_script.append(
                    lit.TestRunner.CommandDirective(
                        directive.start_line_number,
                        directive.end_line_number,
                        directive.keyword,
                        cmd))
        script = new_script

        conditions = { feature: True for feature in test.config.available_features }
        script = lit.TestRunner.applySubstitutions(script, substitutions, conditions,
                                                   recursion_limit=test.config.recursiveExpansionLimit)
        useExternalSh = False
        return lit.TestRunner._runShTest(test, litConfig, useExternalSh, script, tmpBase)
