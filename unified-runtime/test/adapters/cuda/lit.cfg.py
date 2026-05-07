"""

Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats

config.name = "Unified Runtime Adapter (CUDA)"
config.test_format = lit.formats.GoogleTest("", "-test")
config.test_source_root = config.binary_dir
config.test_exec_root = config.binary_dir

config.environment["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
