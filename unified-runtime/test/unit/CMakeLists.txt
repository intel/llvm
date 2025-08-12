# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(unit)
add_gtest_test(getenv getenv.cpp)
add_gtest_test(params params.cpp)
add_gtest_test(print print.cpp)
add_gtest_test(helpers helpers.cpp)
add_gtest_test(mock mock.cpp)
