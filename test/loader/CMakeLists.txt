# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_test(NAME example-hello-world COMMAND hello_world DEPENDS hello_world)
set_tests_properties(example-hello-world PROPERTIES LABELS "loader"
    ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_null>\""
)

add_subdirectory(adapter_registry)
add_subdirectory(loader_config)
add_subdirectory(loader_lifetime)
add_subdirectory(platforms)
add_subdirectory(handles)
