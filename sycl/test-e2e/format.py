import lit
import lit.formats
import platform

from lit.TestRunner import (
    ParserKind,
    IntegratedTestKeywordParser,
)
from E2EExpr import E2EExpr

import os
import re


def remove_level_zero_suffix(devices):
    return [device.replace("_v2", "").replace("_v1", "") for device in devices]


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

    lin = re.search(r"lin: *([0-9]{5})", line)
    if lin:
        if "lin" in output:
            raise ValueError('Multiple entries for "lin" version')
        output["lin"] = int(lin.group(1))

    win = re.search(r"win: *([0-9]{3}\.[0-9]{4})", line)
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
            return lit.Test.Result(lit.Test.UNRESOLVED, str(e))
        script = parsed["RUN:"] or []
        assert parsed["DEFINE:"] == script
        assert parsed["REDEFINE:"] == script

        test.xfails += parsed["XFAIL:"] or []
        test.requires += test.config.required_features
        test.requires += parsed["REQUIRES:"] or []
        test.unsupported += test.config.unsupported_features
        test.unsupported += parsed["UNSUPPORTED:"] or []
        if parsed["ALLOW_RETRIES:"]:
            test.allowed_retries = parsed["ALLOW_RETRIES:"][0]

        test.intel_driver_req = parsed["REQUIRES-INTEL-DRIVER:"]

        return script

    def getMatchedFromList(
        self, features, expression_list, build_only_mode, is_requires_directive
    ):
        try:
            return [
                item
                for item in expression_list
                if E2EExpr.evaluate(
                    item, features, build_only_mode, is_requires_directive
                )
                != is_requires_directive
            ]
        except ValueError as e:
            raise ValueError("Error in expression:\n%s" % str(e))

    BuildOnly = True
    BuildAndRun = False
    RequiresDirective = True
    UnsupportedDirective = False

    def getMissingRequires(self, features, expression_list):
        return self.getMatchedFromList(
            features, expression_list, self.BuildAndRun, self.RequiresDirective
        )

    def getMissingRequiresBuildOnly(self, features, expression_list):
        return self.getMatchedFromList(
            features, expression_list, self.BuildOnly, self.RequiresDirective
        )

    def getMatchedUnsupported(self, features, expression_list):
        return self.getMatchedFromList(
            features, expression_list, self.BuildAndRun, self.UnsupportedDirective
        )

    def getMatchedUnsupportedBuildOnly(self, features, expression_list):
        return self.getMatchedFromList(
            features, expression_list, self.BuildOnly, self.UnsupportedDirective
        )

    getMatchedXFail = getMatchedUnsupported

    def select_build_targets_for_test(self, test):
        supported_targets = set()
        for t in test.config.sycl_build_targets:
            features = test.config.available_features.union({t})
            if self.getMissingRequiresBuildOnly(features, test.requires):
                continue
            if self.getMatchedUnsupportedBuildOnly(features, test.unsupported):
                continue
            supported_targets.add(t)

        if len(supported_targets) <= 1:
            return supported_targets

        # Treat XFAIL as UNSUPPORTED if the test is to be compiled for multiple
        # triples.

        if "*" in test.xfails:
            return []

        triples_without_xfail = [
            t
            for t in supported_targets
            if not self.getMatchedXFail(
                test.config.available_features.union({t}), test.xfails
            )
        ]

        return triples_without_xfail

    def select_devices_for_test(self, test):
        devices = []
        for full_name in test.config.sycl_devices:
            features = test.config.sycl_dev_features[full_name]
            if self.getMissingRequires(features, test.requires):
                continue

            if self.getMatchedUnsupported(features, test.unsupported):
                continue

            driver_ok = True
            if test.intel_driver_req:
                for fmt in ["lin", "win"]:
                    if (
                        fmt in test.intel_driver_req
                        and fmt in test.config.intel_driver_ver[full_name]
                        and test.config.intel_driver_ver[full_name][fmt]
                        < test.intel_driver_req[fmt]
                    ):
                        driver_ok = False
            if not driver_ok:
                continue

            devices.append(full_name)

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
            if not self.getMatchedXFail(test.config.sycl_dev_features[d], test.xfails)
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

        devices_for_test = []
        build_targets = set()
        if test.config.test_mode == "build-only":
            build_targets = self.select_build_targets_for_test(test)
            if not build_targets:
                return lit.Test.Result(
                    lit.Test.UNSUPPORTED, "No supported triple to build for"
                )
        else:
            devices_for_test = self.select_devices_for_test(test)
            if not devices_for_test:
                return lit.Test.Result(
                    lit.Test.UNSUPPORTED, "No supported devices to run the test on"
                )

            for sycl_device in remove_level_zero_suffix(devices_for_test):
                (backend, _) = sycl_device.split(":")
                build_targets.add(test.config.backend_to_target[backend])

        triples = set(test.config.target_to_triple[t] for t in build_targets)
        test.config.available_features = test.config.available_features.union(
            build_targets
        )

        substitutions = lit.TestRunner.getDefaultSubstitutions(test, tmpDir, tmpBase)

        substitutions.append(("%{sycl_triple}", format(",".join(triples))))

        sycl_target_opts = "-fsycl-targets=%{sycl_triple}"
        if "target-amd" in build_targets:
            hip_arch_opts = (
                " -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch={}".format(
                    test.config.amd_arch
                )
            )
            sycl_target_opts += hip_arch_opts
            substitutions.append(("%{hip_arch_opts}", hip_arch_opts))
            substitutions.append(("%{amd_arch}", test.config.amd_arch))
        if (
            "target-spir" in build_targets
            and "spirv-backend" in test.config.available_features
        ):
            # TODO: Maybe that should be link-only option, so that we wouldn't
            # need to suppress the warning below for compile-only commands.
            sycl_target_opts += " -fsycl-use-spirv-backend-for-spirv-gen -Wno-unused-command-line-argument"
        substitutions.append(("%{sycl_target_opts}", sycl_target_opts))

        substitutions.append(
            (
                "%{build}",
                "%clangxx -fsycl %{sycl_target_opts} %verbose_print %s",
            )
        )
        if platform.system() == "Windows":
            substitutions.append(
                (
                    "%{l0_leak_check}",
                    "env UR_L0_LEAKS_DEBUG=1 SYCL_ENABLE_DEFAULT_CONTEXTS=0",
                )
            )
        else:
            substitutions.append(("%{l0_leak_check}", "env UR_L0_LEAKS_DEBUG=1"))

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
                extra_env.append("SYCL_UR_CUDA_ENABLE_IMAGE_SUPPORT=1")

            return extra_env

        extra_env = get_extra_env(remove_level_zero_suffix(devices_for_test))

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

            # Filter commands based on testing mode
            is_run_line = any(
                i in directive.command
                for i in ["%{run}", "%{run-unfiltered-devices}", "%{run-aux}"]
            )

            if (is_run_line and test.config.test_mode == "build-only") or (
                not is_run_line and test.config.test_mode == "run-only"
            ):
                continue

            if "%{run}" not in directive.command:
                new_script.append(directive)
                continue

            for full_dev_name, parsed_dev_name in zip(
                devices_for_test, remove_level_zero_suffix(devices_for_test)
            ):
                expanded = "env"

                extra_env = get_extra_env([parsed_dev_name])
                if extra_env:
                    expanded += " {}".format(" ".join(extra_env))

                if "level_zero_v2" in full_dev_name:
                    expanded += " env UR_LOADER_USE_LEVEL_ZERO_V2=1"
                elif "level_zero_v1" in full_dev_name:
                    expanded += " env UR_LOADER_USE_LEVEL_ZERO_V2=0"

                expanded += " ONEAPI_DEVICE_SELECTOR={} {}".format(
                    parsed_dev_name, test.config.run_launcher
                )
                cmd = directive.command.replace("%{run}", expanded)
                # Expand device-specific condtions (%if ... %{ ... %}).
                tmp_script = [cmd]
                conditions = {x: True for x in parsed_dev_name.split(":")}
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

        if len(script) == 0:
            return lit.Test.Result(lit.Test.UNSUPPORTED, "Lit script is empty")

        result = lit.TestRunner._runShTest(test, litConfig, False, script, tmpBase)

        # Single triple/device - might be an XFAIL.
        def map_result(features, code):
            if "*" in test.xfails or self.getMatchedXFail(features, test.xfails):
                if code is lit.Test.PASS:
                    code = lit.Test.XPASS
                elif code is lit.Test.FAIL:
                    code = lit.Test.XFAIL
            return code

        if len(triples) == 1 and test.config.test_mode == "build-only":
            result.code = map_result(test.config.available_features, result.code)
        if len(devices_for_test) == 1:
            device = devices_for_test[0]
            result.code = map_result(test.config.sycl_dev_features[device], result.code)

        # Set this to empty so internal lit code won't change our result if it incorrectly
        # thinks the test should XFAIL. This can happen when our XFAIL condition relies on
        # device features, since the internal lit code doesn't have knowledge of these.
        test.xfails = []

        return result
