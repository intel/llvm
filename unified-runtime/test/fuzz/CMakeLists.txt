# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(fuzz)

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

add_dependencies(deps_check-unified-runtime-fuzz fuzztest-base ur_adapter_mock)
if(UR_BUILD_ADAPTER_L0 OR UR_BUILD_ADAPTER_ALL)
    add_dependencies(deps_check-unified-runtime-fuzz ur_adapter_level_zero)
endif()
