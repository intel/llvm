# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_fuzz_test name label)
    set(TEST_TARGET_NAME fuzztest-${name})

    set(ENV_VARS "")
    if(UR_USE_UBSAN)
        list(APPEND ENV_VARS
                UBSAN_OPTIONS=print_stacktrace=1)
    endif()

    # The XPTI collector is leaking memory, likely unrelated to UR.
    #if(UR_ENABLE_TRACING)
    #    list(APPEND ENV_VARS
    #            XPTI_TRACE_ENABLE=1
    #            XPTI_FRAMEWORK_DISPATCHER=$<TARGET_FILE:xptifw>
    #            XPTI_SUBSCRIBERS=$<TARGET_FILE:collector>
    #            UR_ENABLE_LAYERS=UR_LAYER_TRACING)
    #endif()

    if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_ALL)
        list(APPEND ENV_VARS
                UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_level_zero>\")
        if(UR_USE_ASAN)
            list(APPEND ENV_VARS
                    NEOReadDebugKeys=1
                    DisableDeepBind=1)
        endif()
    else()
        list(APPEND ENV_VARS UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_mock>\")
    endif()
    
    add_test(NAME ${TEST_TARGET_NAME}
        COMMAND fuzztest-base ${ARGN} -verbosity=1 -detect_leaks=0
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set_tests_properties(${TEST_TARGET_NAME} PROPERTIES
        LABELS ${label}
        ENVIRONMENT "${ENV_VARS}")
endfunction()

# Create a single binary
add_ur_executable(fuzztest-base
    urFuzz.cpp)
target_link_libraries(fuzztest-base
    PRIVATE
    ${PROJECT_NAME}::loader
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    -fsanitize=fuzzer)
# When built with -g and -flto (which is required by some hardening flags), this causes a segfault in (upstream)
# LLVM 14-15 while linking when CMAKE_BUILD_TYPE is Release
target_compile_options(fuzztest-base PRIVATE -fsanitize=fuzzer)
target_compile_definitions(fuzztest-base PRIVATE -DKERNEL_IL_PATH="${UR_CONFORMANCE_DEVICE_BINARIES_DIR}/fill/spir64.bin.0")
target_include_directories(fuzztest-base PRIVATE ${UR_CONFORMANCE_DEVICE_BINARIES_DIR})
add_dependencies(fuzztest-base generate_device_binaries)

# Add long test
add_fuzz_test(base fuzz-long -max_total_time=600 -seed=1)

# Add short tests
set(CORPUS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/corpus)
file(GLOB files "${CORPUS_DIR}/*")

foreach(file ${files})
    get_filename_component(filename ${file} NAME)
    add_fuzz_test(${filename} fuzz-short ${file})
endforeach()
