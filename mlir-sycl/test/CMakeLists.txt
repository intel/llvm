# Copyright (C) Codeplay Software Limited

#===--- CMakeLists.txt -----------------------------------------------------===#
#
# MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIR_SYCL_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        )

add_lit_testsuite(check-mlir-sycl "Running the mlir-sycl regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MLIR_SYCL_TEST_DEPENDS}
        )
set_target_properties(check-mlir-sycl PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_SYCL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MLIR_SYCL_TEST_DEPENDS})

