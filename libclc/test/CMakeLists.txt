# required by lit.site.cfg.py.in
set(LIBCLC_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBCLC_PERTARGET_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/targets)
set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LIBCLC_TEST_DEPS
  FileCheck
  ${llvm-dis_target}
  not
  clang
  count
  libclc-opencl-builtins
)

umbrella_lit_testsuite_begin(check-libclc)

foreach( t ${LIBCLC_TARGETS_TO_BUILD} )
    if (t STREQUAL native_cpu)
      continue()
    endif()

    # Each target gets own lit testsuite directory.
    # This is done because all suites passed to add_lit_testsuite
    # are collected into a single invocation of lit, for example:
    #   add_lit_testsuite(check-a dir PARAMS "foo=1")
    #   add_lit_testsuite(check-b dir PARAMS "foo=2" "bar=3")
    # results in a call to lit like this (for `check-all`):
    #   llvm-lit --param=foo=1 --param=foo=2 --param=bar=3 dir dir
    # This means we can't reuse the same directory and rely on
    # different parameter values to distinguish between targets, because
    # parameters with the same name overwrite each other.
    # The solution is to create a separate directory for each target,
    # and copy the lit.site.cfg.py file there; the name of the
    # directory will be used as the target name.
    file(MAKE_DIRECTORY ${LIBCLC_PERTARGET_TEST_DIR}/${t})
    file(COPY_FILE ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                   ${LIBCLC_PERTARGET_TEST_DIR}/${t}/lit.site.cfg.py)

    add_lit_testsuite(check-libclc-spirv-${t}
      "Running libclc spirv-${t} regression tests"
      ${LIBCLC_PERTARGET_TEST_DIR}/${t}
      DEPENDS
        ${LIBCLC_TEST_DEPS}
        libspirv-builtins
      )
endforeach( t )

umbrella_lit_testsuite_end(check-libclc)
