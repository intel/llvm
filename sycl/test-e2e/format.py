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


# Intel devices report their driver version in one of a few formats, and which
# one is reported depends on the backend and the OS rather than on the hardware:
#
#       L0 (any OS):     [1.3.26370]          -> compute-runtime build "26370"
#       OpenCL/Linux:    [23.22.26370.18]     -> compute-runtime build "26370"
#       OpenCL/Windows:  [31.0.101.4502]      -> Windows package version "101.4502"
#       CPU (OpenCL):    [2024.18.12.0.05_160000]
#
# The compute-runtime build number (five digits) is the one reported by the
# Level Zero adapter everywhere, so it is the OS-agnostic way to gate a test.
# The Windows package version (101.XXXX) is *only* visible under the OpenCL
# backend on Windows. To make this distinction explicit (and to stop a
# Windows-only requirement from silently passing when the device only reports
# the L0 build number) each format has its own directive:
#
#       REQUIRES-L0-DRIVER:             26370       (Level Zero build number)
#       REQUIRES-INTEL-WINDOWS-DRIVER:  101.4502    (Windows OpenCL package)
#       REQUIRES-INTEL-CPU-DRIVER:      2024.18.12  (CPU OpenCL runtime)
#
# These correspond to the "l0", "intel_windows" and "intel_cpu" keys of the
# per-device version dictionary built in lit.cfg.py.


def parse_l0_driver_req(line_number, line, output):
    """Level Zero driver requirement, e.g. `REQUIRES-L0-DRIVER: 26370`. The
    value is the five-digit compute-runtime build number."""
    if output is not None:
        raise ValueError("Multiple REQUIRES-L0-DRIVER directives")
    m = re.search(r"([0-9]{5})", line)
    if not m:
        raise ValueError(
            "REQUIRES-L0-DRIVER expects a five-digit build number, e.g. 26370"
        )
    return int(m.group(1))


def parse_intel_windows_driver_req(line_number, line, output):
    """Windows package version requirement, e.g.
    `REQUIRES-INTEL-WINDOWS-DRIVER: 101.4502`. Returned as a tuple such as
    (101, 4502) to ease comparison."""
    if output is not None:
        raise ValueError("Multiple REQUIRES-INTEL-WINDOWS-DRIVER directives")
    m = re.search(r"([0-9]+)\.([0-9]+)", line)
    if not m:
        raise ValueError(
            "REQUIRES-INTEL-WINDOWS-DRIVER expects a package version, e.g. 101.4502"
        )
    return (int(m.group(1)), int(m.group(2)))


def parse_intel_cpu_driver_req(line_number, line, output):
    """CPU OpenCL runtime requirement, e.g. `REQUIRES-INTEL-CPU-DRIVER: 2026`.
    The value is compared as a string."""
    if output is not None:
        raise ValueError("Multiple REQUIRES-INTEL-CPU-DRIVER directives")
    m = re.search(r"(\S+)", line)
    if not m:
        raise ValueError("REQUIRES-INTEL-CPU-DRIVER expects a version value")
    return m.group(1)


def parse_run_if(line_number, line, output):
    """
    Parse RUN-IF directive in the format:
    // RUN-IF: condition, command
    where condition is a boolean expression and command is the command to execute.
    Returns a CommandDirective with the condition embedded in the command as %if.
    """
    if not output:
        output = []

    # Split on first comma to separate condition from command
    parts = line.split(",", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Line {line_number}: RUN-IF directive must have format: condition, command"
        )

    condition = parts[0].strip()
    command = parts[1].strip()

    if not condition or not command:
        raise ValueError(
            f"Line {line_number}: RUN-IF directive has empty condition or command"
        )

    # Strip outer %{ %} if present to avoid nested braces
    if command.startswith("%{") and command.endswith("%}"):
        command = command[2:-2].strip()

    # Convert RUN-IF to a RUN command with %if condition
    # This leverages lit's built-in conditional support
    conditional_command = f"%if {condition} %{{ {command} %}}"

    output.append(
        lit.TestRunner.CommandDirective(
            line_number, line_number, "RUN:", conditional_command
        )
    )
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
                        "REQUIRES-L0-DRIVER:",
                        ParserKind.CUSTOM,
                        parse_l0_driver_req,
                    ),
                    IntegratedTestKeywordParser(
                        "REQUIRES-INTEL-WINDOWS-DRIVER:",
                        ParserKind.CUSTOM,
                        parse_intel_windows_driver_req,
                    ),
                    IntegratedTestKeywordParser(
                        "REQUIRES-INTEL-CPU-DRIVER:",
                        ParserKind.CUSTOM,
                        parse_intel_cpu_driver_req,
                    ),
                    IntegratedTestKeywordParser(
                        "RUN-IF:", ParserKind.CUSTOM, parse_run_if
                    ),
                ],
                require_script=False,
            )
        except ValueError as e:
            return lit.Test.Result(lit.Test.UNRESOLVED, str(e))
        script = parsed["RUN:"] or []
        assert parsed["DEFINE:"] == script
        assert parsed["REDEFINE:"] == script

        # Add RUN-IF commands to the script
        if parsed["RUN-IF:"]:
            script.extend(parsed["RUN-IF:"])

        # Ensure we have at least one command (RUN or RUN-IF)
        if not script:
            return lit.Test.Result(
                lit.Test.UNRESOLVED, "Test has no 'RUN:' or 'RUN-IF:' line"
            )

        test.xfails += test.config.xfail_features
        test.xfails += parsed["XFAIL:"] or []
        test.requires += test.config.required_features
        test.requires += parsed["REQUIRES:"] or []
        test.unsupported += test.config.unsupported_features
        test.unsupported += parsed["UNSUPPORTED:"] or []
        if parsed["ALLOW_RETRIES:"]:
            test.allowed_retries = parsed["ALLOW_RETRIES:"][0]

        # Minimum driver versions this test requires, keyed by the same format
        # names used in config.intel_driver_ver (see lit.cfg.py). An entry is
        # present only if the corresponding directive appears in the test.
        test.intel_driver_req = {}
        if parsed["REQUIRES-L0-DRIVER:"] is not None:
            test.intel_driver_req["l0"] = parsed["REQUIRES-L0-DRIVER:"]
        if parsed["REQUIRES-INTEL-WINDOWS-DRIVER:"] is not None:
            test.intel_driver_req["intel_windows"] = parsed[
                "REQUIRES-INTEL-WINDOWS-DRIVER:"
            ]
        if parsed["REQUIRES-INTEL-CPU-DRIVER:"] is not None:
            test.intel_driver_req["intel_cpu"] = parsed["REQUIRES-INTEL-CPU-DRIVER:"]

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

            if test.intel_driver_req:
                # Empty for non-Intel devices, in which case an Intel-driver
                # requirement is simply inapplicable and does not gate the test.
                dev_ver = test.config.intel_driver_ver[full_name]
                if dev_ver:
                    # An Intel device reports its version in exactly one of
                    # these formats (which one depends on the backend and OS -
                    # see the comment in lit.cfg.py), so a test that needs to
                    # gate on multiple device configurations specifies each
                    # relevant format. Check every required format the device
                    # actually reports.
                    relevant = [fmt for fmt in test.intel_driver_req if fmt in dev_ver]
                    if not relevant:
                        # The device reports none of the required formats, so we
                        # cannot confirm its driver is new enough. Fail closed
                        # and skip rather than silently running the test - this
                        # is what a bare REQUIRES-INTEL-WINDOWS-DRIVER used to
                        # get wrong on Level Zero (see intel/llvm#23004).
                        continue
                    if any(
                        dev_ver[fmt] < test.intel_driver_req[fmt] for fmt in relevant
                    ):
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

            # At this point, liboffload does not work with multiple GPUs
            if "offload:gpu" in sycl_devices:
                extra_env.append("ZE_AFFINITY_MASK=1")

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
                    "gpu",
                ]:
                    if cond_features in test.config.available_features:
                        conditions[cond_features] = True

                # Add per-device features to conditions (gpu-intel-dg2, aspect-*, arch-*, etc.)
                for feature in test.config.sycl_dev_features[full_dev_name]:
                    conditions[feature] = True

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
