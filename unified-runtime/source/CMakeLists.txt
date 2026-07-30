# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_definitions(-DUR_VERSION="${PROJECT_VERSION_MAJOR}")
add_definitions(-DUR_VALIDATION_LAYER_SUPPORTED_VERSION="${PROJECT_VERSION_MAJOR}")

add_subdirectory(common)
add_subdirectory(loader)
add_subdirectory(mock)
add_subdirectory(adapters)
