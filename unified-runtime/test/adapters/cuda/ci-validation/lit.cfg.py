"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats
import os

config.name = "Unified Runtime Adapter (CUDA) - CI Validation"
config.test_format = lit.formats.ShTest()
config.suffixes = [".test"]

# Use source directory for tests (where .test files are)
config.test_source_root = os.path.dirname(__file__)
# Use binary_dir for test execution (set by lit.site.cfg.py)
config.test_exec_root = config.binary_dir

config.environment["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
