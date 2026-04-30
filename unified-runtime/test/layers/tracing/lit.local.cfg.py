"""

Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

config.substitutions.append(
    (
        r"%xptienable",
        f"UR_ENABLE_LAYERS=UR_LAYER_TRACING XPTI_TRACE_ENABLE=1 XPTI_FRAMEWORK_DISPATCHER={config.shlibpre}xptifw{config.shlibext}",
    )
)
