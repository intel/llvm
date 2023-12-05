# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(urinfo)
if(UR_ENABLE_TRACING)
    add_subdirectory(urtrace)
endif()
