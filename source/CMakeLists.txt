# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

add_definitions(-DUR_VERSION="${PROJECT_VERSION_MAJOR}")
add_definitions(-DUR_VALIDATION_LAYER_SUPPORTED_VERSION="${PROJECT_VERSION_MAJOR}")

add_subdirectory(common)
add_subdirectory(loader)
#add_subdirectory(layers)
add_subdirectory(adapters)
