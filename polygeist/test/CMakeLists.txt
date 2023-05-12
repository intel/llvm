add_subdirectory(lib)

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

set(MLIR_POLYGEIST_TEST_DEPENDS
  FileCheck count not split-file
  polygeist-opt
)

list(APPEND MLIR_POLYGEIST_TEST_DEPENDS MLIRPolygeistUnitTests)

# This target can be used to just build the dependencies
# for the check-polygeist target without executing the tests.
# This is useful for bots when splitting the build step
# from the testing step.
add_custom_target(check-polygeist-build-only
  DEPENDS ${MLIR_POLYGEIST_TEST_DEPENDS}
)

add_lit_testsuite(check-polygeist "Running the Polygeist regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MLIR_POLYGEIST_TEST_DEPENDS}
    ARGS -v
)
set_target_properties(check-polygeist PROPERTIES FOLDER "Tests")

add_lit_testsuites(polygeist ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_POLYGEIST_TEST_DEPENDS}
)
