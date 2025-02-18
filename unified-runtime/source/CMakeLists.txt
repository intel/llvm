# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_definitions(-DUR_VERSION="${PROJECT_VERSION_MAJOR}")
add_definitions(-DUR_VALIDATION_LAYER_SUPPORTED_VERSION="${PROJECT_VERSION_MAJOR}")

add_subdirectory(common)
add_subdirectory(loader)
add_subdirectory(mock)
add_subdirectory(adapters)
