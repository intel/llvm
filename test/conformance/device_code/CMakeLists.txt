# Copyright (C) 2023 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(add_device_binary SOURCE_FILE)
    get_filename_component(KERNEL_NAME ${SOURCE_FILE} NAME_WE)
    set(DEVICE_BINARY_DIR "${UR_CONFORMANCE_DEVICE_BINARIES_DIR}/${KERNEL_NAME}")
    file(MAKE_DIRECTORY ${DEVICE_BINARY_DIR})
    foreach(TRIPLE ${TARGET_TRIPLES})
        add_custom_target(${KERNEL_NAME}_${TRIPLE}_device_binary
            ${UR_DPCXX} -fsycl -fsycl-targets=${TRIPLE} -fsycl-device-code-split=off
            ${SOURCE_FILE} -o "${DEVICE_BINARY_DIR}/${KERNEL_NAME}_${TRIPLE}"
            COMMAND ${CMAKE_COMMAND} -E env SYCL_DUMP_IMAGES=true
            "${DEVICE_BINARY_DIR}/${KERNEL_NAME}_${TRIPLE}" || (exit 0)
            WORKING_DIRECTORY "${DEVICE_BINARY_DIR}"
            COMMAND ${CMAKE_COMMAND} -E remove
            "${DEVICE_BINARY_DIR}/${KERNEL_NAME}_${TRIPLE}"
        )
        add_dependencies(generate_device_binaries ${KERNEL_NAME}_${TRIPLE}_device_binary)
    endforeach()
endfunction()

add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/bar.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/fill.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/foo.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/image_copy.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/mean.cpp)
add_device_binary(${CMAKE_CURRENT_SOURCE_DIR}/spec_constant.cpp)
