# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT

function(add_python_test name)
    set(TEST_NAME python-${name})
    add_test(NAME ${TEST_NAME}
        COMMAND ${Python3_EXECUTABLE} -B -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/${name}.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "python")
    # python uses LD_LIBRARY_PATH (PATH on Windows) to search for dynamic libraries,
    # so set it to the location where it can find the loader.
    if(UNIX)
        set_property(TEST ${TEST_NAME} PROPERTY
            ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    else()
        set_property(TEST ${TEST_NAME} PROPERTY
            ENVIRONMENT_MODIFICATION "PATH=cmake_list_prepend:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()
    # this is for importing the include/ur.py module in other python files
    set_property(TEST ${TEST_NAME} APPEND PROPERTY
       ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}" "UR_ADAPTERS_FORCE_LOAD=$<TARGET_FILE:ur_adapter_null>")
endfunction()

add_python_test(basic)
