configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-polygeist-opt "Verify Polygeist passes perform correctly"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS polygeist-opt
    ARGS -v
)

set_target_properties(check-polygeist-opt PROPERTIES FOLDER "Tests")
