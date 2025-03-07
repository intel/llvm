# Copyright (C) 2025 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


# There are two modes to finding lit and filecheck
# If we are a standalone build, use the python packages
# If we are not a standalone build, use the one provided by LLVM
# Note that the python filecheck re-implementation isn't 100% compatible, but hopefully for simple tests it should be
# good enough

if(UR_STANDALONE_BUILD)
# Standalone build

find_program(URLIT_LIT_BINARY "lit")
if(NOT URLIT_LIT_BINARY)
  message(WARNING "No lit binary was found in path; ensure you are in a Python venv and have installed `third_party/requirements.txt`")
endif()
find_program(URLIT_FILECHECK_BINARY "filecheck")
if(NOT URLIT_FILECHECK_BINARY)
  message(WARNING "No filecheck binary was found in path; ensure you are in a Python venv and have installed `third_party/requirements.txt`")
endif()

function(add_lit_test target)
configure_file(lit.site.cfg.py.in lit.site.cfg.py)
add_custom_target(check-${target}
  COMMAND "${URLIT_LIT_BINARY}" "${CMAKE_CURRENT_BINARY_DIR}" -v
)
add_test(NAME lit COMMAND ${URLIT_LIT_BINARY} "${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

else()
# In source build

# This will be in the $PATH
set(URLIT_FILECHECK_BINARY "FileCheck")

function(add_lit_test target)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

add_lit_testsuite(check-${target} "Running Unified Runtime tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS FileCheck
  )
endfunction()

endif()
