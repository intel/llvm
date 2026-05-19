# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CI validation test suite - intentionally generates different test outcomes
# to validate CI logging and categorization

add_ur_lit_testsuite(adapter-ci-validation)

# Add as dependency to main adapter tests target so it runs automatically
add_dependencies(check-unified-runtime-adapter check-unified-runtime-adapter-ci-validation)
