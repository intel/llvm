"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

config.substitutions.append(
    (
        r"%xptienable",
        f"UR_ENABLE_LAYERS=UR_LAYER_TRACING XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER={config.shlibpre}xptifw{config.shlibext}",
    )
)
