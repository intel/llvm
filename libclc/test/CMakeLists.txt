# required by lit.site.cfg.py.in
set(LIBCLC_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LIBCLC_TEST_DEPS
  FileCheck
  llvm-dis
  not
  clang
  count
)

add_custom_target(check-libclc)

foreach( t ${LIBCLC_TARGET_TO_TEST} )
  foreach( d ${${t}_devices} )
    if( ${d} STREQUAL "none" )
      set( mcpu )
      set( arch_suffix "${t}" )
    else()
      set( mcpu "cpu=${d}" )
      set( arch_suffix "${d}-${t}" )
    endif()
    message( " Testing : ${arch_suffix}" )

    add_lit_testsuite(check-libclc-spirv-${arch_suffix}
      "Running libclc spirv-${arch_suffix} regression tests"
      ${CMAKE_CURRENT_BINARY_DIR}
      ARGS
        --verbose
      PARAMS "target=${t}" ${mcpu} "libclc-arch-suffix=${arch_suffix}"
             "builtins=libspirv-${arch_suffix}.bc"
      DEPENDS
        ${LIBCLC_TEST_DEPS}
        libspirv-builtins
      )

    add_dependencies(check-libclc check-libclc-spirv-${arch_suffix})

  endforeach( d )
endforeach( t )

if(LIBCLC_GENERATE_REMANGLED_VARIANTS)
  #Now that check-libclc is defined make sure that all remangler targets depend
  # on it.
  foreach(remangler-test ${libclc-remangler-tests})
    add_dependencies(check-libclc ${remangler-test})
  endforeach()
endif()
