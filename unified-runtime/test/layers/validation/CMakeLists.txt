# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_VALIDATION_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_gtest_test(parameters parameters.cpp)
add_gtest_test(leaks leaks.cpp)
add_gtest_test(leaks_mt leaks_mt.cpp)
add_gtest_test(lifetime lifetime.cpp)
