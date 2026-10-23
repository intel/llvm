# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_ur_lit_testsuite(hip)
add_dependencies(check-unified-runtime-adapter check-unified-runtime-hip)

# PTX is loaded through the HIP adapter, so this does not need a SYCL
# device-binary generator. On NVIDIA it launches one byte past the default
# shared-memory pool, which is the cuFuncSetAttribute path.
add_conformance_devices_test(adapter-hip-local-mem
    fixtures.h
    local_mem_optin.cpp
)

target_include_directories(adapter-hip-local-mem-test PRIVATE
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/adapters/hip
)

get_target_property(HIP_COMPILE_DEFINITIONS
    ur_adapter_hip COMPILE_DEFINITIONS)
target_compile_definitions(adapter-hip-local-mem-test PRIVATE
    ${HIP_COMPILE_DEFINITIONS}
)

# The kernels test below needs the same driver libraries. On NVIDIA, rocmdrv
# is the AMD runtime and does not satisfy the HIP-on-NVIDIA link.
if("${UR_HIP_PLATFORM}" STREQUAL "NVIDIA")
    set(UR_HIP_TEST_LIBS cudadrv cudart ${PROJECT_NAME}::umf)
else()
    set(UR_HIP_TEST_LIBS rocmdrv ${PROJECT_NAME}::umf)
endif()

target_link_libraries(adapter-hip-local-mem-test PRIVATE ${UR_HIP_TEST_LIBS})

if(NOT UR_FOUND_DPCXX)
    message(WARNING
        "UR_DPCXX is not defined, skipping HIP adapter tests that need SYCL device binaries")
    return()
endif()

add_conformance_kernels_test(adapter-hip
    fixtures.h
    urContextGetNativeHandle.cpp
    urDeviceGetNativeHandle.cpp
    urEventGetNativeHandle.cpp
    test_context.cpp
    test_event.cpp
    kernel_tests.cpp
)

target_include_directories(adapter-hip-test PRIVATE
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/adapters/hip
)

get_target_property(HIP_COMPILE_DEFINITIONS
    ur_adapter_hip COMPILE_DEFINITIONS)
target_compile_definitions(adapter-hip-test PRIVATE
    ${HIP_COMPILE_DEFINITIONS}
)

target_link_libraries(adapter-hip-test PRIVATE ${UR_HIP_TEST_LIBS})
