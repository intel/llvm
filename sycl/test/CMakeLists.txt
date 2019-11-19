set(LLVM_BUILD_LIBRARY_DIRS "${LLVM_BINARY_DIR}/lib/")
set(LLVM_BUILD_BINARY_DIRS "${LLVM_BINARY_DIR}/bin/")
set(LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}/bin/")
set(CLANG_IN_BUILD "${LLVM_BINARY_DIR}/bin/clang")
set(CLANGXX_IN_BUILD "${LLVM_BINARY_DIR}/bin/clang++")
set(CLANGCL_IN_BUILD "${LLVM_BINARY_DIR}/bin/clang-cl")

get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)
get_target_property(SYCL_SOURCE_DIR sycl-toolchain SOURCE_DIR)

set(SYCL_INCLUDE "${dst_dir}")

set(RT_TEST_ARGS ${RT_TEST_ARGS} "-v")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

list(APPEND SYCL_TEST_DEPS
  sycl-toolchain
  FileCheck
  get_device_count_by_type
  llvm-config
  )

add_lit_testsuite(check-sycl "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  DEPENDS ${SYCL_TEST_DEPS}
  )
set_target_properties(check-sycl PROPERTIES FOLDER "SYCL tests")

add_lit_testsuites(SYCL ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${SYCL_TEST_DEPS}
  )
