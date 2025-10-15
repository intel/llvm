configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "CMAKE_OSX_SYSROOT"
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "LLVM_LIBS_DIR"
  "SHLIBDIR"
  )

# TODO: Consider adding to check-sycl if this is every moved to under llvm/tests
# Add a target to invoke tests via Ninja/make.
add_lit_testsuite(check-sycl-vecz-tests "Running SYCL vecz lit tests"
    "${CMAKE_CURRENT_BINARY_DIR}"

    DEPENDS
    veczc
    FileCheck
)

add_custom_target(check-sycl-vecz)
add_dependencies(check-sycl-vecz check-sycl-vecz-tests)
