"""

Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

from os import path

config.suffixes = [".cpp"]

config.substitutions.append(
    (r"%cwd", path.join(config.test_exec_root, "loader", "adapter_registry"))
)
