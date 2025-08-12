"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

config.substitutions.append(
    (
        r"%trace",
        f"urtrace --stdout --flush info --mock --libpath {config.main_lib_dir}",
    )
)
