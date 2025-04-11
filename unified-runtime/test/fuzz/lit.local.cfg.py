"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

config.substitutions.append(
    (
        r"%fuzz-options",
        " ".join(
            [
                "NEOReadDebugKeys=1"
                "DisableDeepBind=1"
                "UBSAN_OPTIONS=print_stacktrace=1"
            ]
        ),
    )
)
