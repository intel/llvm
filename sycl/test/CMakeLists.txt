set(LLVM_BUILD_LIBRARY_DIRS "${LLVM_BINARY_DIR}/lib/")
set(LLVM_BUILD_BINARY_DIRS "${LLVM_BINARY_DIR}/bin/")
set(LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}/bin/")
set(CLANG_IN_BUILD "${LLVM_BINARY_DIR}/bin/clang")
set(CLANGXX_IN_BUILD "${LLVM_BINARY_DIR}/bin/clang++")

set(SYCL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SYCL_TESTS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCL_TESTS_BINARY_DIR ${SYCL_BINARY_DIR}/test)

set(RT_TEST_ARGS ${RT_TEST_ARGS} "-v")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg
  )

add_lit_testsuite(check-sycl "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  DEPENDS sycl-toolchain FileCheck get_device_count_by_type llvm-config
  )
set_target_properties(check-sycl PROPERTIES FOLDER "SYCL tests")
