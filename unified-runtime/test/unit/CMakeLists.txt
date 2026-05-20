# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(unit)
add_gtest_test(getenv getenv.cpp)
add_gtest_test(params params.cpp)
add_gtest_test(print print.cpp)
add_gtest_test(helpers helpers.cpp)
add_gtest_test(mock mock.cpp)
