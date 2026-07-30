"""

Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

config.suffixes = [".cpp"]

config.substitutions.append(
    (
        r"%validate",
        r"UR_ENABLE_LAYERS=UR_LAYER_FULL_VALIDATION UR_LOG_VALIDATION=level:debug\;flush:debug\;output:stdout",
    )
)
