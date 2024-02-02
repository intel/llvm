import lit
import lit.formats
import platform

from lit.BooleanExpression import BooleanExpression
from lit.TestRunner import (
    ParserKind,
    IntegratedTestKeywordParser,
    # parseIntegratedTestScript,
)

import os
import re


def get_triple(test, backend):
    if backend == "cuda":
        return "nvptx64-nvidia-cuda"
    if backend == "hip":
        if test.config.hip_platform == "NVIDIA":
            return "nvptx64-nvidia-cuda"
        else:
            return "amdgcn-amd-amdhsa"
    if backend == "native_cpu":
        return "native_cpu"
    return "spir64"


def parse_min_intel_driver_req(line_number, line, output):
    """
    Driver version looks like this for Intel devices:
          Linux/L0:       [1.3.26370]
          Linux/opencl:   [23.22.26370.18]
          Windows/L0:     [1.3.26370]
          Windows/opencl: [31.0.101.4502]
    Only "26370" and "101.4502" are interesting for us for the purpose of detecting
    if the driver has required changes or not. As such we refer to the former
    (5-digit) as "lin" format and as "win" for the latter."""
    if not output:
        output = {}

    lin = re.search("lin: *([0-9]{5})", line)
    if lin:
        if "lin" in output:
            raise ValueError('Multiple entries for "lin" version')
        output["lin"] = int(lin.group(1))

    win = re.search("win: *([0-9]{3}\.[0-9]{4})", line)
    if win:
        if "win" in output:
            raise ValueError('Multiple entries for "win" version')
        # Return "win" version as (101, 4502) to ease later comparison.
        output["win"] = tuple(map(int, win.group(1).split(".")))

    return output


class SYCLEndToEndTest(lit.formats.ShTest):
    def parseTestScript(self, test):
        """This is based on lit.TestRunner.parseIntegratedTestScript but we
        overload the semantics of REQUIRES/UNSUPPORTED/XFAIL directives so have
        to implement it manually."""

        # Parse the test sources and extract test properties
        try:
            parsed = lit.TestRunner._parseKeywords(
                test.getSourcePath(),
                additional_parsers=[
                    IntegratedTestKeywordParser(
                        "REQUIRES-INTEL-DRIVER:",
                        ParserKind.CUSTOM,
                        parse_min_intel_driver_req,
                    )
                ],
                require_script=True,
            )
        except ValueError as e:
            return lit.Test.Result(Test.UNRESOLVED, str(e))
        script = parsed["RUN:"] or []
        assert parsed["DEFINE:"] == script
        assert parsed["REDEFINE:"] == script

        test.xfails += parsed["XFAIL:"] or []
        test.requires += test.config.required_features
        test.requires += parsed["REQUIRES:"] or []
        test.unsupported += test.config.unsupported_features
        test.unsupported += parsed["UNSUPPORTED:"] or []

        test.intel_driver_req = parsed["REQUIRES-INTEL-DRIVER:"]

        return script

    def getMatchedFromList(self, features, alist):
        try:
            return [
                item for item in alist if BooleanExpression.evaluate(item, features)
            ]
        except ValueError as e:
            raise ValueError("Error in UNSUPPORTED list:\n%s" % str(e))

    def select_devices_for_test(self, test):
        devices = []
        for d in test.config.sycl_devices:
            features = test.config.sycl_dev_features[d]
            if test.getMissingRequiredFeaturesFromList(features):
                continue

            if self.getMatchedFromList(features, test.unsupported):
                continue

            driver_ok = True
            if test.intel_driver_req:
                for fmt in ["lin", "win"]:
                    if (
                        fmt in test.intel_driver_req
                        and fmt in test.config.intel_driver_ver[d]
                        and test.config.intel_driver_ver[d][fmt]
                        < test.intel_driver_req[fmt]
                    ):
                        driver_ok = False
            if not driver_ok:
                continue

            devices.append(d)

        if len(devices) <= 1:
            return devices

        # Treat XFAIL as UNSUPPORTED if the test is to be executed on multiple
        # devices.
        #
        # TODO: What if the entire list of devices consists of XFAILs only?

        if "*" in test.xfails:
            return []

        devices_without_xfail = [
            d
            for d in devices
            if not self.getMatchedFromList(
                test.config.sycl_dev_features[d], test.xfails
            )
        ]

        return devices_without_xfail

    def execute(self, test, litConfig):
        if test.config.unsupported:
            return lit.Test.Result(lit.Test.UNSUPPORTED, "Test is unsupported")

        filename = test.path_in_suite[-1]
        tmpDir, tmpBase = lit.TestRunner.getTempPaths(test)
        script = self.parseTestScript(test)
        if isinstance(script, lit.Test.Result):
            return script

        devices_for_test = self.select_devices_for_test(test)
        if not devices_for_test:
            return lit.Test.Result(
                lit.Test.UNSUPPORTED, "No supported devices to run the test on"
            )

        substitutions = lit.TestRunner.getDefaultSubstitutions(test, tmpDir, tmpBase)
        triples = set()
        for sycl_device in devices_for_test:
            (backend, _) = sycl_device.split(":")
            triples.add(get_triple(test, backend))

        substitutions.append(("%{sycl_triple}", format(",".join(triples))))
        # -fsycl-targets is needed for CUDA/HIP, so just use it be default so
        # -that new tests by default would runnable there (unless they have
        # -other restrictions).
        substitutions.append(
            ("%{build}", "%clangxx -fsycl -fsycl-targets=%{sycl_triple} %s")
        )
        if platform.system() == "Windows":
            substitutions.append(
                ("%{l0_leak_check}", "env UR_L0_LEAKS_DEBUG=1 SYCL_ENABLE_DEFAULT_CONTEXTS=0")
            )
        else:
            substitutions.append(("%{l0_leak_check}", "env UR_L0_LEAKS_DEBUG=1"))

        compilation_cmd_pthread = (
            "%clangxx -pthread -fsycl -fsycl-targets=%{sycl_triple} %s"
        )
        substitutions.append(("%{build_pthread_inc}", compilation_cmd_pthread))

        def get_extra_env(sycl_devices):
            # Note: It's possible that the system has a device from below but
            # current llvm-lit invocation isn't configured to include it. We
            # don't use ONEAPI_DEVICE_SELECTOR for `%{run-unfiltered-devices}`
            # so that device might still be accessible to some of the tests yet
            # we won't set the environment variable below for such scenario.
            extra_env = []
            if "level_zero:gpu" in sycl_devices and litConfig.params.get("ur_l0_debug"):
                extra_env.append("UR_L0_DEBUG={}".format(test.config.ur_l0_debug))

            if "level_zero:gpu" in sycl_devices and litConfig.params.get(
                "ur_l0_leaks_debug"
            ):
                extra_env.append(
                    "UR_L0_LEAKS_DEBUG={}".format(test.config.ur_l0_leaks_debug)
                )

            if "cuda:gpu" in sycl_devices:
                extra_env.append("SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT=1")

            return extra_env

        extra_env = get_extra_env(devices_for_test)

        run_unfiltered_substitution = ""
        if extra_env:
            run_unfiltered_substitution = "env {} ".format(" ".join(extra_env))
        run_unfiltered_substitution += test.config.run_launcher

        substitutions.append(("%{run-unfiltered-devices}", run_unfiltered_substitution))

        new_script = []
        for directive in script:
            if not isinstance(directive, lit.TestRunner.CommandDirective):
                new_script.append(directive)
                continue

            if "%{run}" not in directive.command:
                new_script.append(directive)
                continue

            for sycl_device in devices_for_test:
                expanded = "env"

                extra_env = get_extra_env([sycl_device])
                if extra_env:
                    expanded += " {}".format(" ".join(extra_env))

                expanded += " ONEAPI_DEVICE_SELECTOR={} {}".format(
                    sycl_device, test.config.run_launcher
                )
                cmd = directive.command.replace("%{run}", expanded)
                # Expand device-specific condtions (%if ... %{ ... %}).
                tmp_script = [cmd]
                conditions = {x: True for x in sycl_device.split(":")}
                for cond_features in [
                    "linux",
                    "windows",
                    "preview-breaking-changes-supported",
                ]:
                    if cond_features in test.config.available_features:
                        conditions[cond_features] = True

                tmp_script = lit.TestRunner.applySubstitutions(
                    tmp_script,
                    [],
                    conditions,
                    recursion_limit=test.config.recursiveExpansionLimit,
                )

                new_script.append(
                    lit.TestRunner.CommandDirective(
                        directive.start_line_number,
                        directive.end_line_number,
                        directive.keyword,
                        tmp_script[0],
                    )
                )
        script = new_script

        conditions = {feature: True for feature in test.config.available_features}
        script = lit.TestRunner.applySubstitutions(
            script,
            substitutions,
            conditions,
            recursion_limit=test.config.recursiveExpansionLimit,
        )
        useExternalSh = False
        result = lit.TestRunner._runShTest(
            test, litConfig, useExternalSh, script, tmpBase
        )

        if len(devices_for_test) > 1:
            return result

        # Single device - might be an XFAIL.
        device = devices_for_test[0]
        if "*" in test.xfails or self.getMatchedFromList(
            test.config.sycl_dev_features[device], test.xfails
        ):
            if result.code is lit.Test.PASS:
                result.code = lit.Test.XPASS
            # fail -> expected fail
            elif result.code is lit.Test.FAIL:
                result.code = lit.Test.XFAIL
            return result

        return result
