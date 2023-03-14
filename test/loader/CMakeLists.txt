# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

add_test(NAME example-hello-world COMMAND hello_world DEPENDS hello_world)
set_tests_properties(example-hello-world PROPERTIES LABELS "loader"
    ENVIRONMENT "UR_ADAPTERS_FORCE_LOAD=$<TARGET_FILE:ur_adapter_null>"
)
