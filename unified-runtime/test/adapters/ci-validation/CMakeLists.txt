# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CI validation test suite - intentionally generates different test outcomes
# to validate CI logging and categorization

add_custom_target(check-unified-runtime-adapter-ci-validation
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/llvm/utils/lit/lit.py
            --show-pass
            --show-unsupported
            --show-xfail
            --no-progress-bar
            --succinct
            --timeout 10
            ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running CI validation test suite (generates intentional failures/timeouts)"
    USES_TERMINAL
)

# Add as dependency to main adapter tests target so it runs automatically
add_dependencies(check-unified-runtime-adapter check-unified-runtime-adapter-ci-validation)
