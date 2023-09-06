add_subdirectory(lib)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIR_SYCL_TEST_DEPENDS
  FileCheck count not
  sycl-mlir-opt
)

add_lit_testsuite(check-mlir-sycl "Running the mlir-sycl regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_SYCL_TEST_DEPENDS}
)
set_target_properties(check-mlir-sycl PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_SYCL ${CMAKE_CURRENT_SOURCE_DIR} 
  DEPENDS ${MLIR_SYCL_TEST_DEPENDS}
)
