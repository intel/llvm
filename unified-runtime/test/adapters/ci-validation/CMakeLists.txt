# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CI validation test suite - intentionally generates different test outcomes
# to validate CI logging and categorization

add_ur_lit_testsuite(adapter-ci-validation)

# NOTE: CI validation tests are now integrated into adapter-specific test suites
# (e.g., cuda/ci-validation/) to appear together in test results.
# This standalone suite is kept for compatibility but not added as dependency.
