"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

import lit.formats
import os
import re

config.name = "Unified Runtime Adapter (CUDA) - CI Validation"
config.test_format = lit.formats.ShTest()
config.suffixes = [".test"]

# Use the directory where lit.cfg.py is located (source directory) for tests
config.test_source_root = os.path.dirname(__file__)
# Use binary_dir for test execution (set by lit.site.cfg.py)
config.test_exec_root = config.binary_dir

# Inherit environment from parent CUDA config
if not hasattr(config, 'environment'):
    config.environment = {}
config.environment["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"

# Add FileCheck substitution - use hardcoded path for now
# TODO: Use proper cmake variable to find FileCheck
filecheck_path = "/home/kkaczmax/llvm/build_sycl/bin/FileCheck"
if os.path.exists(filecheck_path):
    # Add word boundary matching like LLVM does
    def word_match(key, subst):
        regex = re.compile(rf"\b{key}\b")
        return (regex, subst)
    if not hasattr(config, 'substitutions'):
        config.substitutions = []
    config.substitutions.append(word_match("FileCheck", filecheck_path))
