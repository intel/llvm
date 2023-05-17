import lit
import lit.formats

from lit.BooleanExpression import BooleanExpression

import os

def get_triple(test, backend):
    if backend == 'ext_oneapi_cuda':
        return 'nvptx64-nvidia-cuda'
    if backend == 'ext_oneapi_hip':
        if test.config.hip_platform == 'NVIDIA':
            return 'nvptx64-nvidia-cuda'
        else:
            return 'amdgcn-amd-amdhsa'
    return 'spir64'

class SYCLEndToEndTest(lit.formats.ShTest):
    def parseTestScript(self, test):
        """This is based on lit.TestRunner.parseIntegratedTestScript but we
        overload the semantics of REQUIRES/UNSUPPORTED/XFAIL directives so have
        to implement it manually."""

        # Parse the test sources and extract test properties
        try:
            parsed = lit.TestRunner._parseKeywords(test.getSourcePath(), require_script=True)
        except ValueError as e:
            return lit.Test.Result(Test.UNRESOLVED, str(e))
        script = parsed['RUN:'] or []
        assert parsed['DEFINE:'] == script
        assert parsed['REDEFINE:'] == script

        test.xfails += parsed['XFAIL:'] or []
        test.requires += test.config.required_features
        test.requires += parsed['REQUIRES:'] or []
        test.unsupported += test.config.unsupported_features
        test.unsupported += parsed['UNSUPPORTED:'] or []

        return script

    def getMatchedFromList(self, features, alist):
        try:
            return [item for item in alist
                    if BooleanExpression.evaluate(item, features)]
        except ValueError as e:
            raise ValueError('Error in UNSUPPORTED list:\n%s' % str(e))

    def select_devices_for_test(self, test):
        devices = []
        for d in test.config.sycl_devices:
            features = test.config.sycl_dev_features[d]
            if test.getMissingRequiredFeaturesFromList(features):
                continue

            if self.getMatchedFromList(features, test.unsupported):
                continue

            devices.append(d)

        if len(devices) <= 1:
            return devices

        # Treat XFAIL as UNSUPPORTED if the test is to be executed on multiple
        # devices.
        #
        # TODO: What if the entire list of devices consists of XFAILs only?

        if '*' in test.xfails:
            return []

        devices_without_xfail = [d for d in devices
                                 if not self.getMatchedFromList(test.config.sycl_dev_features[d], test.xfails)]

        return devices_without_xfail

    def execute(self, test, litConfig):
        if test.config.unsupported:
            return lit.Test.Result(lit.Test.UNSUPPORTED, 'Test is unsupported')

        filename = test.path_in_suite[-1]
        tmpDir, tmpBase = lit.TestRunner.getTempPaths(test)
        script = self.parseTestScript(test)
        if isinstance(script, lit.Test.Result):
            return script

        devices_for_test = self.select_devices_for_test(test)
        if not devices_for_test:
            return lit.Test.Result(lit.Test.UNSUPPORTED, 'No supported devices to run the test on')

        substitutions = lit.TestRunner.getDefaultSubstitutions(test, tmpDir, tmpBase)
        triples = set()
        for sycl_device in devices_for_test:
            (backend, _) = sycl_device.split(':')
            triples.add(get_triple(test, backend))

        substitutions.append(('%{sycl_triple}', format(','.join(triples))))
        # -fsycl-targets is needed for CUDA/HIP, so just use it be default so
        # -that new tests by default would runnable there (unless they have
        # -other restrictions).
        substitutions.append(('%{build}', '%clangxx -fsycl -fsycl-targets=%{sycl_triple} %s'))

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

            # ESIMD_EMULATOR backend uses CM_EMU library package for
            # multi-threaded execution on CPU, and the package emulates
            # multiple target platforms. In case user does not specify
            # what target platform to emulate, 'skl' is chosen by default.
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
                for op_sys in ['linux', 'windows']:
                    if op_sys in test.config.available_features:
                        conditions[op_sys] = True

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
        result = lit.TestRunner._runShTest(test, litConfig, useExternalSh, script, tmpBase)

        if (len(devices_for_test) > 1):
            return result

        # Single device - might be an XFAIL.
        device = devices_for_test[0]
        if '*' in test.xfails or self.getMatchedFromList(test.config.sycl_dev_features[device], test.xfails):
            if result.code is lit.Test.PASS:
                result.code = lit.Test.XPASS
            # fail -> expected fail
            elif result.code is lit.Test.FAIL:
                result.code = lit.Test.XFAIL
            return result

        return result
