"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

from os import path
import lit.formats

config.suffixes = [".test", ".cpp"]

if config.l0_v1_enabled:
    config.substitutions.append((r"%maybe-v1", r"%with-v1"))
    config.substitutions.append((r"%with-v1", "UR_LOADER_USE_LEVEL_ZERO_V2=0"))
    config.available_features.add("v1")
else:
    config.substitutions.append(
        (r"%maybe-v1", "echo 'Level Zero V1 is not available' || ")
    )

if config.l0_v2_enabled:
    config.substitutions.append((r"%maybe-v2", r"%with-v2"))
    config.substitutions.append((r"%with-v2", "UR_LOADER_USE_LEVEL_ZERO_V2=1"))
    config.available_features.add("v2")
else:
    config.substitutions.append(
        (r"%maybe-v2", "echo 'Level Zero V2 is not available' || ")
    )

if config.l0_static_link:
    config.available_features.add("static-link")

config.environment["ONEAPI_DEVICE_SELECTOR"] = "level_zero:*"
