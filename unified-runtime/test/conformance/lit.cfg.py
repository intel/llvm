"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats
import subprocess
from os import path

config.name = "Unified Runtime Conformance"
config.test_format = lit.formats.GoogleTest("", "-test")
config.test_source_root = config.binary_dir
config.test_exec_root = config.binary_dir

config.environment["UR_LOADER_USE_LEVEL_ZERO_V2"] = "1" if config.using_l0_v2 else "0"

if "UR_CTS_ALSO_RUN_KNOWN_FAILURES" in os.environ:
    config.environment["UR_CTS_ALSO_RUN_KNOWN_FAILURES"] = os.environ[
        "UR_CTS_ALSO_RUN_KNOWN_FAILURES"
    ]

if from_param := lit_config.params.get("selector"):
    # Specified via passing `-Dselector="backend:*"` to lit
    selector = from_param
elif from_env := os.environ.get("ONEAPI_DEVICE_SELECTOR"):
    lit_config.warning(
        "ONEAPI_DEVICE_SELECTOR set in environment, using that for UR conformance tests"
    )
    selector = from_env
elif config.default_selector:
    selector = config.default_selector
else:
    selector = ";".join([f"{a}:*" for a in config.adapters_built if a != "mock"])
config.environment["ONEAPI_DEVICE_SELECTOR"] = selector

if not selector:
    lit_config.warning(
        "No device selector is set (are any adapters enabled?), conformance tests may not run as expected"
    )

lit_config.note(
    f"Running Unified Runtime conformance tests with ONEAPI_DEVICE_SELECTOR='{selector}':"
)
urinfo = subprocess.check_output(
    path.join(config.runtime_dir, "urinfo"),
    text=True,
    env=config.environment,
    stderr=subprocess.DEVNULL,
)
for l in urinfo.splitlines():
    if l:
        lit_config.note(f"  {l}")
