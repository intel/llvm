# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CI validation test suite for CUDA adapter
# These tests intentionally generate different test outcomes
# to validate CI logging and categorization

add_ur_lit_testsuite(cuda-ci-validation)
