# Copyright (C) 2022 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(CUDA_DIR "${SYCL_ADAPTER_DIR}/sycl/plugins/unified_runtime/ur/adapters/cuda")

set(TARGET_NAME ur_adapter_cuda)

add_ur_adapter(${TARGET_NAME}
    SHARED
    ${CUDA_DIR}/ur_interface_loader.cpp
    ${CUDA_DIR}/common.hpp
    ${CUDA_DIR}/common.cpp
    ${CUDA_DIR}/context.hpp
    ${CUDA_DIR}/context.cpp
    ${CUDA_DIR}/device.hpp
    ${CUDA_DIR}/device.cpp
    ${CUDA_DIR}/enqueue.cpp
    ${CUDA_DIR}/event.hpp
    ${CUDA_DIR}/event.cpp
    ${CUDA_DIR}/kernel.hpp
    ${CUDA_DIR}/kernel.cpp
    ${CUDA_DIR}/memory.hpp
    ${CUDA_DIR}/memory.cpp
    ${CUDA_DIR}/platform.hpp
    ${CUDA_DIR}/platform.cpp
    ${CUDA_DIR}/program.hpp
    ${CUDA_DIR}/program.cpp
    ${CUDA_DIR}/queue.hpp
    ${CUDA_DIR}/queue.cpp
    ${CUDA_DIR}/sampler.hpp
    ${CUDA_DIR}/sampler.cpp
    ${CUDA_DIR}/tracing.cpp
    ${CUDA_DIR}/usm.cpp
    ${CUDA_DIR}/../../ur.cpp
    ${CUDA_DIR}/../../ur.hpp
    ${CUDA_DIR}/../../usm_allocator.cpp
    ${CUDA_DIR}/../../usm_allocator.hpp
    ${CUDA_DIR}/../../usm_allocator_config.cpp
    ${CUDA_DIR}/../../usm_allocator_config.hpp
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

target_link_libraries(${TARGET_NAME} PRIVATE
    ${PROJECT_NAME}::headers
    ${PROJECT_NAME}::common
    Threads::Threads
    cudadrv
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CUDA_DIR}/../../../
)
