"""
Copyright (C) 2025 Intel Corporation

Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.TXT
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""

# Skip adapter testing for adapters that have not been built
if "cuda" not in config.adapters_built:
    config.excludes.add("cuda")
if "hip" not in config.adapters_built:
    config.excludes.add("hip")
if "level_zero" not in config.adapters_built:
    config.excludes.add("level_zero")
