# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_NAME ur_adapter_hip)

# Set default UR HIP platform to AMD
set(UR_HIP_PLATFORM "AMD" CACHE STRING "UR HIP platform, AMD or NVIDIA")

# Set default ROCm installation directory
set(UR_HIP_ROCM_DIR "/opt/rocm" CACHE STRING "ROCm installation dir")
# Allow custom location of HIP/HSA include and HIP library directories
set(UR_HIP_INCLUDE_DIR "${UR_HIP_ROCM_DIR}/include" CACHE PATH
  "Custom ROCm HIP include dir")
set(UR_HIP_HSA_INCLUDE_DIRS
  "${UR_HIP_ROCM_DIR}/hsa/include;${UR_HIP_ROCM_DIR}/include" CACHE FILEPATH
  "Custom ROCm HSA include dir")
set(UR_HIP_LIB_DIR "${UR_HIP_ROCM_DIR}/lib" CACHE PATH
  "Custom ROCm HIP library dir")

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

    # Check if HSA include path exists. In rocm-6.0.0 the layout of HSA
    # directory has changed, check for the new location as well.
    foreach(D IN LISTS UR_HIP_HSA_INCLUDE_DIRS)
        if(EXISTS "${D}")
            set(UR_HIP_HSA_INCLUDE_DIR "${D}")
            break()
        endif()
    endforeach()
    if(NOT UR_HIP_HSA_INCLUDE_DIR)
        message(FATAL_ERROR "Couldn't find the HSA include directory in any of "
        "these paths: '${UR_HIP_HSA_INCLUDE_DIRS}'. Please check ROCm "
        "installation.")
    endif()
endif()

# Set includes used in added library (rocmdrv)
set(HIP_HEADERS "${UR_HIP_INCLUDE_DIR};${UR_HIP_HSA_INCLUDE_DIR}")

add_ur_adapter(${TARGET_NAME}
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/ur_interface_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physical_mem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/platform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/virtual_mem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.hpp
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

    if(UR_ENABLE_COMGR)
        set(UR_COMGR_VERSION5_HEADER "${UR_HIP_INCLUDE_DIR}/amd_comgr/amd_comgr.h")
        set(UR_COMGR_VERSION4_HEADER "${UR_HIP_INCLUDE_DIR}/amd_comgr.h")
        # The COMGR header changed location between ROCm versions 4 and 5.
        # Check for existence in the version 5 location or fallback to version 4
        if(NOT EXISTS "${UR_COMGR_VERSION5_HEADER}")
            if(NOT EXISTS "${UR_COMGR_VERSION4_HEADER}")
                message(FATAL_ERROR "Could not find AMD COMGR header at "
                                    "${UR_COMGR_VERSION5_HEADER} or"
                                    "${UR_COMGR_VERSION4_HEADER}, "
                                    "check ROCm installation")
            else()
                target_compile_definitions(${TARGET_NAME} PRIVATE UR_COMGR_VERSION4_INCLUDE)
            endif()
        endif()

        add_library(amd_comgr SHARED IMPORTED GLOBAL)
        set_target_properties(
        amd_comgr PROPERTIES
            IMPORTED_LOCATION                    "${UR_HIP_LIB_DIR}/libamd_comgr.so"
            INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
        )
        target_link_libraries(${TARGET_NAME} PUBLIC amd_comgr)
        target_compile_definitions(${TARGET_NAME} PRIVATE SYCL_ENABLE_KERNEL_FUSION)
    endif(UR_ENABLE_COMGR)

    target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::unified_malloc_framework
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
    "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)
