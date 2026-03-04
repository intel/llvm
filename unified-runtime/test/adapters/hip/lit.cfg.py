"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats

config.name = "Unified Runtime Adapter (HIP)"
config.test_format = lit.formats.GoogleTest("", "-test")
config.test_source_root = config.binary_dir
config.test_exec_root = config.binary_dir

config.environment["ONEAPI_DEVICE_SELECTOR"] = "hip:*"
