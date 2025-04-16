"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

config.suffixes = [".cpp"]

config.substitutions.append(
    (
        r"%validate",
        r"UR_ENABLE_LAYERS=UR_LAYER_FULL_VALIDATION UR_LOG_VALIDATION=level:debug\;flush:debug\;output:stdout",
    )
)
