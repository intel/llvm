import lit
import lit.formats

import os

class SYCLEndToEndTest(lit.formats.ShTest):
    def execute(self, test, litConfig):
        filename = test.path_in_suite[-1]
        tmpDir, tmpBase = lit.TestRunner.getTempPaths(test)
        script = lit.TestRunner.parseIntegratedTestScript(test, require_script=True)
        if isinstance(script, lit.Test.Result):
            return script

        devices_for_test = ['{}:{}'.format(test.config.sycl_be, dev)
                            for dev in test.config.target_devices.split(',')]

        substitutions = lit.TestRunner.getDefaultSubstitutions(test, tmpDir, tmpBase)
        # -fsycl-targets is needed for CUDA/HIP, so just use it be default so
        # -that new tests by default would runnable there (unless they have
        # -other restrictions).
        substitutions.append(('%{build}', '%clangxx -fsycl -fsycl-targets=%sycl_triple %s'))

        def get_extra_env(sycl_devices):
            # Note: It's possible that the system has a device from below but
            # current llvm-lit invocation isn't configured to include it. We
            # don't use ONEAPI_DEVICE_SELECTOR for `%{run-unfiltered-devices}`
            # so that device might still be accessible to some of the tests yet
            # we won't set the environment variable below for such scenario.
            extra_env = []
            if 'ext_oneapi_level_zero:gpu' in sycl_devices and litConfig.params.get('ze_debug'):
                extra_env.append('ZE_DEBUG={}'.format(test.config.ze_debug))

            if 'ext_oneapi_cuda:gpu' in sycl_devices:
                extra_env.append('SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT=1')

            if 'ext_intel_esimd_emulator:gpu' in sycl_devices and not "CM_RT_PLATFORM" in os.environ:
                extra_env.append('CM_RT_PLATFORM=skl')

            return extra_env

        extra_env = get_extra_env(devices_for_test)

        run_unfiltered_substitution = ''
        if extra_env:
            run_unfiltered_substitution = 'env {} '.format(' '.join(extra_env))
        run_unfiltered_substitution += test.config.run_launcher

        substitutions.append(('%{run-unfiltered-devices}', run_unfiltered_substitution))

        new_script = []
        for directive in script:
            if not isinstance(directive, lit.TestRunner.CommandDirective):
                new_script.append(directive)
                continue

            if '%{run}' not in directive.command:
                new_script.append(directive)
                continue

            for sycl_device in devices_for_test:
                expanded = 'env'

                extra_env = get_extra_env([sycl_device])
                if extra_env:
                    expanded += ' {}'.format(' '.join(extra_env))

                expanded += ' ONEAPI_DEVICE_SELECTOR={} {}'.format(sycl_device, test.config.run_launcher)
                cmd = directive.command.replace('%{run}', expanded)
                # Expand device-specific condtions (%if ... %{ ... %}).
                tmp_script = [ cmd ]
                conditions = {x: True for x in sycl_device.split(':')}
                for os in ['linux', 'windows']:
                    if os in test.config.available_features:
                        conditions[os] = True

                tmp_script = lit.TestRunner.applySubstitutions(
                    tmp_script, [], conditions, recursion_limit=test.config.recursiveExpansionLimit)

                new_script.append(
                    lit.TestRunner.CommandDirective(
                        directive.start_line_number,
                        directive.end_line_number,
                        directive.keyword,
                        tmp_script[0]))
        script = new_script

        conditions = { feature: True for feature in test.config.available_features }
        script = lit.TestRunner.applySubstitutions(script, substitutions, conditions,
                                                   recursion_limit=test.config.recursiveExpansionLimit)
        useExternalSh = False
        return lit.TestRunner._runShTest(test, litConfig, useExternalSh, script, tmpBase)
