# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(test-loader-handles
    urLoaderHandles.cpp
)

target_link_libraries(test-loader-handles
    PRIVATE
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::loader
    gmock
    GTest::gtest_main
)

add_test(NAME loader-handles
    COMMAND test-loader-handles
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_tests_properties(loader-handles PROPERTIES
    LABELS "loader"
    ENVIRONMENT "UR_ENABLE_LOADER_INTERCEPT=1;UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_null>\""
)
