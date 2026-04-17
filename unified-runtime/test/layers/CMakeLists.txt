# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(layers DEPENDS
    ur_adapter_mock
    hello_world)

add_subdirectory(validation)

if(UR_ENABLE_TRACING)
    add_subdirectory(tracing)
    add_dependencies(deps_check-unified-runtime-layers ur_collector test_collector collector xptifw)
endif()

if(UR_ENABLE_SANITIZER)
    add_subdirectory(sanitizer)
endif()
