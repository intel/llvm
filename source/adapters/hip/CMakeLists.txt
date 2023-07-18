# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(HIP_DIR "${SYCL_ADAPTER_DIR}/sycl/plugins/unified_runtime/ur/adapters/hip")

set(TARGET_NAME ur_adapter_hip)

# Set default UR HIP platform to AMD
set(UR_HIP_PLATFORM "AMD" CACHE STRING "UR HIP platform, AMD or NVIDIA")

# Set default ROCm installation directory
set(UR_HIP_ROCM_DIR "/opt/rocm" CACHE STRING "ROCm installation dir")

set(UR_HIP_INCLUDE_DIR "${UR_HIP_ROCM_DIR}/include")

set(UR_HIP_HSA_INCLUDE_DIR "${UR_HIP_ROCM_DIR}/hsa/include")

# Set HIP lib dir
set(UR_HIP_LIB_DIR "${UR_HIP_ROCM_DIR}/hip/lib")

# Check if HIP library path exists (AMD platform only)
if("${UR_HIP_PLATFORM}" STREQUAL "AMD")
    if(NOT EXISTS "${UR_HIP_LIB_DIR}")
    message(FATAL_ERROR "Couldn't find the HIP library directory at '${UR_HIP_LIB_DIR}',"
                        " please check ROCm installation.")
    endif()
    # Check if HIP include path exists
    if(NOT EXISTS "${UR_HIP_INCLUDE_DIR}")
    message(FATAL_ERROR "Couldn't find the HIP include directory at '${UR_HIP_INCLUDE_DIR}',"
                        " please check ROCm installation.")
    endif()

    # Check if HSA include path exists
    if(NOT EXISTS "${UR_HIP_HSA_INCLUDE_DIR}")
    message(FATAL_ERROR "Couldn't find the HSA include directory at '${UR_HIP_HSA_INCLUDE_DIR}',"
                        " please check ROCm installation.")
    endif()
endif()

# Set includes used in added library (rocmdrv)
set(HIP_HEADERS "${UR_HIP_INCLUDE_DIR};${UR_HIP_HSA_INCLUDE_DIR}")

add_ur_adapter(${TARGET_NAME}
    SHARED
    ${HIP_DIR}/ur_interface_loader.cpp
    ${HIP_DIR}/common.hpp
    ${HIP_DIR}/common.cpp
    ${HIP_DIR}/context.hpp
    ${HIP_DIR}/context.cpp
    ${HIP_DIR}/device.hpp
    ${HIP_DIR}/device.cpp
    ${HIP_DIR}/enqueue.cpp
    ${HIP_DIR}/event.hpp
    ${HIP_DIR}/event.cpp
    ${HIP_DIR}/kernel.hpp
    ${HIP_DIR}/kernel.cpp
    ${HIP_DIR}/memory.hpp
    ${HIP_DIR}/memory.cpp
    ${HIP_DIR}/platform.hpp
    ${HIP_DIR}/platform.cpp
    ${HIP_DIR}/program.hpp
    ${HIP_DIR}/program.cpp
    ${HIP_DIR}/queue.hpp
    ${HIP_DIR}/queue.cpp
    ${HIP_DIR}/sampler.hpp
    ${HIP_DIR}/sampler.cpp
    ${HIP_DIR}/usm.cpp
    ${HIP_DIR}/../../ur.cpp
    ${HIP_DIR}/../../ur.hpp
    ${HIP_DIR}/../../usm_allocator.cpp
    ${HIP_DIR}/../../usm_allocator.hpp
    ${HIP_DIR}/../../usm_allocator_config.cpp
    ${HIP_DIR}/../../usm_allocator_config.hpp
)

if(NOT MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE
        -Wno-deprecated-declarations
    )
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

if("${UR_HIP_PLATFORM}" STREQUAL "AMD")
    # Import HIP runtime library
    add_library(rocmdrv SHARED IMPORTED GLOBAL)

    set_target_properties(
    rocmdrv PROPERTIES
        IMPORTED_LOCATION                    "${UR_HIP_LIB_DIR}/libamdhip64.so"
        INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    rocmdrv
    )

    # Set HIP define to select AMD platform
    target_compile_definitions(${TARGET_NAME} PRIVATE __HIP_PLATFORM_AMD__)
elseif("${UR_HIP_PLATFORM}" STREQUAL "NVIDIA")
    # Import CUDA libraries
    find_package(CUDA REQUIRED)
    find_package(Threads REQUIRED)

    list(APPEND HIP_HEADERS ${CUDA_INCLUDE_DIRS})

    # cudadrv may be defined by the CUDA plugin
    if(NOT TARGET cudadrv)
    add_library(cudadrv SHARED IMPORTED GLOBAL)
    set_target_properties(
        cudadrv PROPERTIES
        IMPORTED_LOCATION                    ${CUDA_CUDA_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
    )
    endif()

    add_library(cudart SHARED IMPORTED GLOBAL)
    set_target_properties(
    cudart PROPERTIES
        IMPORTED_LOCATION                    ${CUDA_CUDART_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    Threads::Threads
    cudadrv
    cudart
    )

    # Set HIP define to select NVIDIA platform
    target_compile_definitions(${TARGET_NAME} PRIVATE __HIP_PLATFORM_NVIDIA__)
else()
    message(FATAL_ERROR "Unspecified UR HIP platform please set UR_HIP_PLATFORM to 'AMD' or 'NVIDIA'")
endif()

target_include_directories(${TARGET_NAME} PRIVATE
    ${HIP_DIR}/../../../
)
