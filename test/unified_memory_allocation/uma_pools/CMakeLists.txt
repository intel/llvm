# Copyright (C) 2022-2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_uma_test(disjointPool disjoint_pool.cpp)
target_include_directories(uma_test-disjointPool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(uma_test-disjointPool PRIVATE ${PROJECT_NAME}::common)
