# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_fuzz_test name label)
    set(TEST_TARGET_NAME fuzztest-${name})
    add_ur_executable(${TEST_TARGET_NAME}
        urFuzz.cpp)
    target_link_libraries(${TEST_TARGET_NAME}
        PRIVATE
        ${PROJECT_NAME}::loader
        ${PROJECT_NAME}::headers
        ${PROJECT_NAME}::common
        -fsanitize=fuzzer)
    add_test(NAME ${TEST_TARGET_NAME}
        COMMAND ${TEST_TARGET_NAME} ${ARGN}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(${TEST_TARGET_NAME} PROPERTIES
        LABELS ${label}
        ENVIRONMENT
            "UR_ENABLE_LAYERS=UR_LAYER_FULL_VALIDATION;UR_ADAPTERS_FORCE_LOAD=\"$<TARGET_FILE:ur_adapter_null>\"")

    # TODO: Should we check if this sanitizer flag is available?
    target_compile_options(${TEST_TARGET_NAME} PRIVATE -g -fsanitize=fuzzer)
    target_compile_definitions(${TEST_TARGET_NAME} PRIVATE -DKERNEL_IL_PATH="${UR_CONFORMANCE_DEVICE_BINARIES_DIR}/bar/sycl_spir641.spv")
    target_include_directories(${TEST_TARGET_NAME} PRIVATE ${UR_CONFORMANCE_DEVICE_BINARIES_DIR})

    add_dependencies(${TEST_TARGET_NAME} generate_device_binaries)
endfunction()

# Add long test
add_fuzz_test(base fuzz-long
    -max_total_time=600 -seed=1 -verbosity=1)

# Add short tests
set(CORPUS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/corpus)
file(GLOB files "${CORPUS_DIR}/*")

foreach(file ${files})
    get_filename_component(filename ${file} NAME)
    add_fuzz_test(${filename} fuzz-short ${file} -verbosity=1)
endforeach()
