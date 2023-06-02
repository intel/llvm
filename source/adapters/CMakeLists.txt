# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

add_subdirectory(null)

if(UR_BUILD_ADAPTER_L0)
add_subdirectory(level_zero)
endif()
