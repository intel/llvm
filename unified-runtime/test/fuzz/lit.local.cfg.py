"""

Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
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
