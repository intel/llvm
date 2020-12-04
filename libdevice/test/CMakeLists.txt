set(LIBDEVICE_TEST_DEPENDS
  FileCheck
  clang-offload-bundler
  count
  llvm-dis
  llvm-spirv
  not
  )

# Enable new IN_LIST operator.
cmake_policy(SET CMP0057 NEW)
if (libdevice IN_LIST LLVM_ENABLE_PROJECTS AND
    (sycl IN_LIST LLVM_ENABLE_PROJECTS OR
     sycl IN_LIST LLVM_EXTERNAL_PROJECTS))
  set(LLVM_HAS_LIBSYCLDEVICE TRUE)
  list(APPEND LIBDEVICE_TEST_DEPENDS libsycldevice)
endif()

llvm_canonicalize_cmake_booleans(
  LLVM_HAS_LIBSYCLDEVICE
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "CLANG_TOOLS_DIR"
  "CMAKE_CURRENT_SOURCE_DIR"
  "LIBDEVICE_SOURCE_DIR"
  "LIBDEVICE_BINARY_DIR"
  "LIBDEVICE_LIBRARY_DIR"
  "LIBDEVICE_RUNTIME_DIR"
  "LLVM_TOOLS_DIR"
  )

add_custom_target(libdevice-test-depends DEPENDS ${LIBDEVICE_TEST_DEPENDS})
set_target_properties(libdevice-test-depends PROPERTIES FOLDER "Tests")

if(LLVM_BUILD_TOOLS AND
   LLVM_HAS_LIBSYCLDEVICE)
  set(exclude_from_check_all "")
else()
  set(exclude_from_check_all "EXCLUDE_FROM_CHECK_ALL")
endif()

add_lit_testsuite(check-libdevice "Running the libdevice regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ${exclude_from_check_all}
  DEPENDS ${LIBDEVICE_TEST_DEPENDS}
  )
set_target_properties(check-libdevice PROPERTIES FOLDER "Tests")
