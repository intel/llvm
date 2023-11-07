# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

macro(add_device_binary SOURCE_FILE)
    get_filename_component(KERNEL_NAME ${SOURCE_FILE} NAME_WE)
    set(DEVICE_BINARY_DIR "${UR_CONFORMANCE_DEVICE_BINARIES_DIR}/${KERNEL_NAME}")
    file(MAKE_DIRECTORY ${DEVICE_BINARY_DIR})
    if(UR_SYCL_LIBRARY_DIR)
        if(CMAKE_SYSTEM_NAME STREQUAL Linux)
            set(EXTRA_ENV LD_LIBRARY_PATH=${UR_SYCL_LIBRARY_DIR})
        elseif(CMAKE_SYSTEM_NAME STREQUAL Windows)
            set(EXTRA_ENV PATH=${UR_SYCL_LIBRARY_DIR};$ENV{PATH})
        else()
            set(EXTRA_ENV DYLD_FALLBACK_LIBRARY_PATH=${UR_SYCL_LIBRARY_DIR})
        endif()
    endif()
    foreach(TRIPLE ${TARGET_TRIPLES})
        set(EXE_PATH "${DEVICE_BINARY_DIR}/${KERNEL_NAME}_${TRIPLE}")
        add_custom_command(OUTPUT ${EXE_PATH}
            COMMAND ${UR_DPCXX} -fsycl -fsycl-targets=${TRIPLE} -fsycl-device-code-split=off
            ${SOURCE_FILE} -o ${EXE_PATH}
            COMMAND ${CMAKE_COMMAND} -E env ${EXTRA_ENV} SYCL_DUMP_IMAGES=true
            ${EXE_PATH} || exit 0
            WORKING_DIRECTORY "${DEVICE_BINARY_DIR}"
            DEPENDS ${SOURCE_FILE}
        )
        add_custom_target(generate_${KERNEL_NAME}_${TRIPLE} DEPENDS ${EXE_PATH})
        add_dependencies(generate_device_binaries generate_${KERNEL_NAME}_${TRIPLE})
    endforeach()
    list(APPEND DEVICE_CODE_SOURCES ${SOURCE_FILE})
endmacro()

add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/bar.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/device_global.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/fill.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/fill_2d.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/fill_3d.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/fill_usm.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/foo.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/image_copy.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/mean.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/spec_constant.cpp)

set(KERNEL_HEADER ${UR_CONFORMANCE_DEVICE_BINARIES_DIR}/kernel_entry_points.h)
add_custom_command(OUTPUT ${KERNEL_HEADER}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts
    COMMAND ${Python3_EXECUTABLE} generate_kernel_header.py
    --dpcxx_path ${UR_DPCXX} -o ${KERNEL_HEADER} ${DEVICE_CODE_SOURCES}
    DEPENDS ${PROJECT_SOURCE_DIR}/scripts/generate_kernel_header.py
    ${DEVICE_CODE_SOURCES})
add_custom_target(kernel_names_header DEPENDS ${KERNEL_HEADER})
add_dependencies(generate_device_binaries kernel_names_header)
