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

    add_lit_testsuite(check-libclc-spirv-${arch_suffix} "Running libclc spirv-${arch_suffix} regression tests"
      ${CMAKE_CURRENT_BINARY_DIR}
      ARGS
        --verbose
      PARAMS "target=${t}" ${mcpu} "builtins=libspirv-${arch_suffix}.bc"
      DEPENDS
        ${LIBCLC_TEST_DEPS}
        libspirv-builtins
      )

    add_dependencies(check-libclc check-libclc-spirv-${arch_suffix})

  endforeach( d )
endforeach( t )

if(LIBCLC_GENERATE_REMANGLED_VARIANTS)
  # Run remangler in test mode if generating remangled variants and make sure
  # it depends on check-libclc target.
  # Both `long_widths` and `char_signedness` are set in AddLibclc.cmake and can
  # be used here.
  foreach(long_width ${long_widths})
    foreach(signedness ${char_signedness})
      # In `-t` (TestRun) the remangler does not perform any substitutions, it
      # needs to make sure that the remangled name matches the original mangled
      # one.
      set (test_target_name "test-remangled-${long_width}-${signedness}_char")
      add_custom_target(${test_target_name}
        COMMAND libclc-remangler
        --long-width=${long_width}
        --char-signedness=${signedness}
        --input-ir="$<TARGET_PROPERTY:prepare-${obj_suffix},TARGET_FILE>"
        ${dummy_in} -t -o -
        DEPENDS "${builtins_obj_path}" "prepare-${obj_suffix}" "${dummy_in}" libclc-remangler)

      add_dependencies(check-libclc ${test_target_name})
    endforeach()
  endforeach()
endif()
