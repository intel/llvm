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

if(LIBCLC_GENERATE_REMANGLED_VARIANTS)
  # Now that check-libclc is defined make sure that all remangler targets depend
  # on it.
  foreach(remangler-test ${libclc-remangler-tests})
    set_property(GLOBAL APPEND PROPERTY LLVM_LIBCLC_ADDITIONAL_TEST_TARGETS ${remangler-test})
    set_property(GLOBAL APPEND PROPERTY LLVM_ALL_ADDITIONAL_TEST_TARGETS ${remangler-test})
  endforeach()
endif()

# Testing unresolved symbols.
foreach(t ${LIBCLC_TARGETS_TO_BUILD})
  string(REPLACE "-" ";" TRIPLE ${t})
  list(GET TRIPLE 0 ARCH)

  # Skip nvptx, clspv, spirv targets
  if(ARCH MATCHES "^(nvptx|clspv)(64)?$" OR ARCH MATCHES "^spirv(64)?$")
    continue()
  endif()

  # Get the output file from the target property
  set(target_file "$<TARGET_PROPERTY:libclc-${t},TARGET_FILE>")

  set(LIBCLC_TARGET_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/${t})
  file(MAKE_DIRECTORY ${LIBCLC_TARGET_TEST_DIR})
  file(GENERATE OUTPUT ${LIBCLC_TARGET_TEST_DIR}/check-external-funcs.test
    CONTENT "; RUN: llvm-nm -u \"${target_file}\" | FileCheck %s --allow-empty\n\n; CHECK-NOT: {{.+}}\n"
  )

  configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
      ${LIBCLC_TARGET_TEST_DIR}/lit.site.cfg.py
      MAIN_CONFIG
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

  add_lit_testsuite(check-libclc-external-funcs-${t} "Running ${t} tests"
    ${LIBCLC_TARGET_TEST_DIR}
    DEPENDS ${LIBCLC_TEST_DEPS}
  )
  set_target_properties(check-libclc-external-funcs-${t} PROPERTIES FOLDER "libclc tests")
endforeach()

umbrella_lit_testsuite_end(check-libclc)
