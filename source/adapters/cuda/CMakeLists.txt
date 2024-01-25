# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TARGET_NAME ur_adapter_cuda)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/image.hpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/tracing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/usm_p2p.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/virtual_mem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ur/ur.hpp
)

set_target_properties(${TARGET_NAME} PROPERTIES
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)

find_package(Threads REQUIRED)
find_package(CUDA 10.1 REQUIRED)

# Make imported library global to use it within the project.
add_library(cudadrv SHARED IMPORTED GLOBAL)

if (WIN32)
  set_target_properties(
    cudadrv PROPERTIES 
      IMPORTED_IMPLIB               ${CUDA_CUDA_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
  )
else()
  set_target_properties(
    cudadrv PROPERTIES 
      IMPORTED_LOCATION             ${CUDA_CUDA_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
  )
endif()

if(UMF_ENABLE_POOL_TRACKING)
  target_compile_definitions("ur_adapter_cuda" PRIVATE UMF_ENABLE_POOL_TRACKING)
else()
  message(WARNING "CUDA adapter USM pools are disabled, set UMF_ENABLE_POOL_TRACKING to enable them")
endif()

if (CUDA_cupti_LIBRARY)
  target_compile_definitions("ur_adapter_cuda" PRIVATE CUPTI_LIB_PATH="${CUDA_cupti_LIBRARY}")
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    ${PROJECT_NAME}::unified_malloc_framework
    Threads::Threads
    cudadrv
)

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)
