# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(loader DEPENDS ur_loader hello_world)

add_subdirectory(adapter_registry)
add_subdirectory(loader_config)
add_subdirectory(loader_lifetime)
add_subdirectory(platforms)
add_subdirectory(handles)
