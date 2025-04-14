"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats
import subprocess
from os import path


class NopTestFormat(lit.formats.TestFormat):
    def getTestsForPath(*args, **kwargs):
        return
        yield

    def getTestsInDirectory(*args, **kwargs):
        return
        yield


config.name = "Unified Runtime Conformance"
config.test_format = lit.formats.GoogleTest("", "-test")
config.test_source_root = config.binary_dir
config.test_exec_root = config.binary_dir

if config.using_l0_v2:
    config.environment["UR_LOADER_USE_LEVEL_ZERO_V2"] = "1"
else:
    config.environment["UR_LOADER_USE_LEVEL_ZERO_V2"] = "0"

if "UR_CTS_ALSO_RUN_KNOWN_FAILURES" in os.environ:
    config.environment["UR_CTS_ALSO_RUN_KNOWN_FAILURES"] = os.environ[
        "UR_CTS_ALSO_RUN_KNOWN_FAILURES"
    ]

if "selector" in lit_config.params:
    # Specified via passing `-Dselector="backend:*"` to lit
    config.environment["ONEAPI_DEVICE_SELECTOR"] = lit_config.params["selector"]
elif os.environ.get("ONEAPI_DEVICE_SELECTOR", ""):
    lit_config.warning(
        "ONEAPI_DEVICE_SELECTOR set in environment, using that for UR conformance tests"
    )
    config.environment["ONEAPI_DEVICE_SELECTOR"] = os.environ["ONEAPI_DEVICE_SELECTOR"]
else:
    ods = ""
    for a in config.adapters_built:
        if a != "mock":
            ods += f"{a}:*;"
    config.environment["ONEAPI_DEVICE_SELECTOR"] = ods

selector = config.environment["ONEAPI_DEVICE_SELECTOR"]
if selector == "":
    lit_config.warning(
        "No device selector is set (are any adapters enabled?), skipping conformance tests"
    )
    config.test_format = NopTestFormat()
else:
    lit_config.note(
        f"Running Unified Runtime conformance tests with ONEAPI_DEVICE_SELECTOR='{selector}':"
    )
    urinfo = subprocess.run(
        path.join(config.runtime_dir, "urinfo"),
        text=True,
        env=config.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for l in urinfo.stdout.split("\n"):
        if l:
            lit_config.note(f"  {l}")
