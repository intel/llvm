# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(UR_VALIDATION_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_gtest_test(parameters parameters.cpp)
add_gtest_test(leaks leaks.cpp)
add_gtest_test(leaks_mt leaks_mt.cpp)
add_gtest_test(lifetime lifetime.cpp)
