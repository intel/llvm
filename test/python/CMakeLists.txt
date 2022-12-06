# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

add_test(NAME python-basic
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/basic.py
)

# python uses LD_LIBRARY_PATH to search for dynamic libraries, so set it to
# the location where it can find the loader.
if(UNIX)
        set_property(TEST python-basic PROPERTY
                ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()