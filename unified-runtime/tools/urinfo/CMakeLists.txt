# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_executable(urinfo
    urinfo.hpp
    utils.hpp
    urinfo.cpp
)
target_compile_definitions(urinfo PRIVATE
    UR_VERSION="${PROJECT_VERSION}"
)
target_include_directories(urinfo PRIVATE
    ${PROJECT_SOURCE_DIR}/source/common
)
target_link_libraries(urinfo PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::loader
)
