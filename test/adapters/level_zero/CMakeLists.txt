# Copyright (C) 2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT UR_DPCXX)
    # Tests that require kernels can't be used if we aren't generating
    # device binaries
    message(WARNING
        "UR_DPCXX is not defined, skipping some adapter tests for level_zero")
else()
    add_adapter_test(level_zero
        FIXTURE KERNELS
        SOURCES
            urProgramLink.cpp
            urKernelCreateWithNativeHandle.cpp
        ENVIRONMENT
            "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero>\""
    )
    # TODO: valgrind tests require very new environment.
    # Enable once all L0 runners are updated.
    # add_adapter_memcheck_test(level_zero
    #    ENVIRONMENT
    #         "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero>\""
    # )

    target_link_libraries(test-adapter-level_zero PRIVATE
        LevelZeroLoader
        LevelZeroLoader-Headers
    )

    target_include_directories(test-adapter-level_zero PRIVATE
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/source/adapters/level_zero
        LevelZeroLoader-Headers
    )

    add_dependencies(test-adapter-level_zero
        generate_device_binaries kernel_names_header)
endif()

if(NOT WIN32)
    # Make L0 use CallMap from a seprate shared lib so that we can access the map
    # from the tests. This only seems to work on linux
    add_library(zeCallMap SHARED zeCallMap.cpp)
    target_compile_definitions(ur_adapter_level_zero PRIVATE UR_L0_CALL_COUNT_IN_TESTS)
    target_link_libraries(ur_adapter_level_zero PRIVATE zeCallMap)

    add_adapter_test(level_zero_ze_calls
        FIXTURE DEVICES
        SOURCES
            event_cache_tests.cpp
        ENVIRONMENT
            "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero>\""
            "UR_L0_LEAKS_DEBUG=1"
    )

    target_link_libraries(test-adapter-level_zero_ze_calls PRIVATE zeCallMap)

    add_adapter_test(level_zero_multi_queue
        FIXTURE DEVICES
        SOURCES
            multi_device_event_cache_tests.cpp
        ENVIRONMENT
            "UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero>\""
            "UR_L0_LEAKS_DEBUG=1"
    )

    target_link_libraries(test-adapter-level_zero_multi_queue PRIVATE zeCallMap)
endif()

add_subdirectory(v2)
