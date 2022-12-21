# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

add_test(NAME python-basic
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${Python3_EXECUTABLE} -B -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/basic.py
)

# python uses LD_LIBRARY_PATH (PATH on Windows) to search for dynamic libraries,
# so set it to the location where it can find the loader.
if(UNIX)
        set_property(TEST python-basic PROPERTY
                ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
else()
        set_property(TEST python-basic PROPERTY
                     ENVIRONMENT_MODIFICATION "PATH=cmake_list_prepend:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

# this is for importing the include/ur.py module in other python files
set_property(TEST python-basic APPEND PROPERTY
             ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}")
