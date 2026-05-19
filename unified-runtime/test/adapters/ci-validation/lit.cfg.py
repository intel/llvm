"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats

config.name = "Unified Runtime CI Validation"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".test"]
config.test_source_root = config.binary_dir
config.test_exec_root = config.binary_dir
