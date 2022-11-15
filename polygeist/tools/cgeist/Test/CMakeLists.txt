set(MLIR_CLANG_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_CLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg
)

list(APPEND MLIR_CLANG_TEST_DEPS
  llvm-config 
  llvm-dis
  FileCheck count not
  cgeist
  split-file
  clang
  opt
  sycl-toolchain
  )

add_lit_testsuite(check-cgeist "Running the clang-to-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} 
  DEPENDS ${MLIR_CLANG_TEST_DEPS}
  )

add_lit_testsuite(check-cgeist-single "Running the clang-to-mlir regression tests"
  ${CMAKE_CURRENT_BINARY_DIR} 
  DEPENDS ${MLIR_CLANG_TEST_DEPS}
  ARGS -j 1
  )

set_target_properties(check-cgeist PROPERTIES FOLDER "clang-to-mlir tests")
