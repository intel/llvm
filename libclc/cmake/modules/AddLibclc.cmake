function(add_libclc_alias alias target)
  cmake_parse_arguments(ARG "" "" PARENT_TARGET "" ${ARGN})

  if(CMAKE_HOST_UNIX AND NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(LIBCLC_LINK_OR_COPY create_symlink)
  else()
    set(LIBCLC_LINK_OR_COPY copy)
  endif()

  add_custom_command(
      OUTPUT ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${alias_suffix}
      COMMAND ${CMAKE_COMMAND} -E
        ${LIBCLC_LINK_OR_COPY} ${target}.bc
        ${alias_suffix}
      WORKING_DIRECTORY
        ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
      DEPENDS "prepare-${target}"
    )
  add_custom_target( alias-${alias_suffix} ALL
    DEPENDS "${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${alias_suffix}" )
  add_dependencies(${ARG_PARENT_TARGET} alias-${alias_suffix})

  install( FILES ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${alias_suffix}
           DESTINATION ${CMAKE_INSTALL_DATADIR}/clc )

endfunction(add_libclc_alias alias target)

# add_libclc_builtin_set(arch_suffix
#   TRIPLE string
#     Triple used to compile
#   TARGET_ENV string
#     "clc" or "libspirv"
#   FILES string ...
#     List of file that should be built for this library
#   ALIASES string ...
#     List of alises
#   COMPILE_OPT
#     Compilation options
#   LIB_DEP
#     Library to include to the builtin set
#   )
macro(add_libclc_builtin_set arch_suffix)
  cmake_parse_arguments(ARG
    ""
    "TRIPLE;TARGET_ENV;LIB_DEP;PARENT_TARGET"
    "FILES;ALIASES;GENERATE_TARGET;COMPILE_OPT;OPT_FLAGS"
    ${ARGN})

  if (DEFINED ${ARG_LIB_DEP})
    set(LIB_DEP ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${ARG_LIB_DEP}.bc)
    set(TARGET_DEP prepare-${ARG_LIB_DEP}.bc)
  endif()

  add_library( builtins.link.${arch_suffix}
    STATIC ${ARG_FILES} ${LIB_DEP})
  # Make sure we depend on the pseudo target to prevent
  # multiple invocations
  add_dependencies( builtins.link.${arch_suffix}
    ${ARG_GENERATE_TARGET} ${TARGET_DEP})
  # Add dependency to used tools
  add_dependencies( builtins.link.${arch_suffix}
    llvm-as llvm-link opt clang )
  # CMake will turn this include into absolute path
  target_include_directories( builtins.link.${arch_suffix} PRIVATE
    "generic/include" )
  target_compile_definitions( builtins.link.${arch_suffix} PRIVATE
    "__CLC_INTERNAL" )
  target_compile_options( builtins.link.${arch_suffix} PRIVATE
    -target ${ARG_TRIPLE} ${ARG_COMPILE_OPT} -fno-builtin -nostdlib )
  set_target_properties( builtins.link.${arch_suffix} PROPERTIES
    LINKER_LANGUAGE CLC )
  set_output_directory(builtins.link.${arch_suffix} LIBRARY_DIR ${LIBCLC_LIBRARY_OUTPUT_INTDIR})

  set( obj_suffix ${arch_suffix}.bc )

  # Add opt target
  set( builtins_opt_path "${LIBCLC_LIBRARY_OUTPUT_INTDIR}/builtins.opt.${obj_suffix}" )
  add_custom_command( OUTPUT "${builtins_opt_path}"
    COMMAND ${LLVM_OPT} ${ARG_OPT_FLAGS} -o
    "${builtins_opt_path}"
    "${LIBCLC_LIBRARY_OUTPUT_INTDIR}/builtins.link.${obj_suffix}"
    DEPENDS opt "builtins.link.${arch_suffix}" )
  add_custom_target( "opt.${obj_suffix}" ALL
    DEPENDS "${builtins_opt_path}" )
  set_target_properties("opt.${obj_suffix}"
    PROPERTIES TARGET_FILE "${builtins_opt_path}")

  # Add prepare target
  set( builtins_obj_path "${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}" )
  add_custom_command( OUTPUT "${builtins_obj_path}"
    COMMAND prepare_builtins -o
    "${builtins_obj_path}"
    "$<TARGET_PROPERTY:opt.${obj_suffix},TARGET_FILE>"
    DEPENDS "${builtins_opt_path}" "opt.${obj_suffix}"
            prepare_builtins )
  add_custom_target( "prepare-${obj_suffix}" ALL
    DEPENDS "${builtins_obj_path}" )
  set_target_properties("prepare-${obj_suffix}"
    PROPERTIES TARGET_FILE "${builtins_obj_path}")

  # Add dependency to top-level pseudo target to ease making other
  # targets dependent on libclc.
  add_dependencies(${ARG_PARENT_TARGET} "prepare-${obj_suffix}")

  install(
    FILES ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/clc )
  
  # Generate remangled variants if requested
  if( LIBCLC_GENERATE_REMANGLED_VARIANTS )
    set(long_widths l32 l64)
    set(char_signedness signed unsigned)
    # All permutations of [l32, l64] and [signed, unsigned]
    foreach(long_width ${long_widths})
      foreach(signedness ${char_signedness})
        # Remangle
        set( builtins_remangle_path
            "${LIBCLC_LIBRARY_OUTPUT_INTDIR}/remangled-${long_width}-${signedness}_char.${obj_suffix}" )
        add_custom_command( OUTPUT "${builtins_remangle_path}"
          COMMAND libclc-remangler
          -o "${builtins_remangle_path}"
          --long-width=${long_width}
          --char-signedness=${signedness}
          "$<TARGET_PROPERTY:prepare-${obj_suffix},TARGET_FILE>"
          DEPENDS "${builtins_obj_path}" "prepare-${obj_suffix}" libclc-remangler )
        add_custom_target( "remangled-${long_width}-${signedness}_char.${obj_suffix}" ALL
          DEPENDS "${builtins_remangle_path}" )
        set_target_properties("remangled-${long_width}-${signedness}_char.${obj_suffix}"
          PROPERTIES TARGET_FILE "${builtins_remangle_path}")

        # Add dependency to top-level pseudo target to ease making other
        # targets dependent on libclc.
        add_dependencies(${ARG_PARENT_TARGET} "remangled-${long_width}-${signedness}_char.${obj_suffix}")

        # Keep remangled variants
        install(
          FILES ${builtins_remangle_path}
          DESTINATION ${CMAKE_INSTALL_DATADIR}/clc )
      endforeach()
    endforeach()
  endif()

  # nvptx-- targets don't include workitem builtins
  if( NOT ${t} MATCHES ".*ptx.*--$" )
    add_test( NAME external-calls-${obj_suffix}
      COMMAND ./check_external_calls.sh ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
      WORKING_DIRECTORY ${LIBCLC_LIBRARY_OUTPUT_INTDIR} )
    set_tests_properties( external-calls-${obj_suffix}
      PROPERTIES ENVIRONMENT "LLVM_CONFIG=${LLVM_CONFIG}" )
  endif()

  foreach( a ${$ARG_ALIASES} )
    set( alias_suffix "${ARG_TARGET_ENV}-${a}-${ARG_TRIPLE}.bc" )
    add_libclc_alias( ${alias_suffix}
      ${arch_suffix}
      PARENT_TARGET ${ARG_PARENT_TARGET})
  endforeach( a )

endmacro(add_libclc_builtin_set arch_suffix)

function(libclc_configure_lib_source OUT_LIST)
  cmake_parse_arguments(ARG
    ""
    "LIB_DIR"
    "DIRS;DEPS"
    ${ARGN})

  # Enumerate SOURCES* files
  set( source_list )
  foreach( l ${ARG_DIRS} )
    foreach( s "SOURCES" "SOURCES_${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}" )
      file( TO_CMAKE_PATH ${l}/${ARG_LIB_DIR}/${s} file_loc )
      file( TO_CMAKE_PATH ${LIBCLC_ROOT_DIR}/${file_loc} loc )
      # Prepend the location to give higher priority to
      # specialized implementation
      if( EXISTS ${loc} )
        # Make cmake configuration depends on the SOURCE file
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${loc})
        set( source_list ${loc} ${source_list} )
      endif()
    endforeach()
  endforeach()

  # Add the generated convert.cl here to prevent adding
  # the one listed in SOURCES
  set( rel_files ${ARG_DEPS} )
  set( objects ${ARG_DEPS} )
  if( NOT ENABLE_RUNTIME_SUBNORMAL )
    if( EXISTS generic/${ARG_LIB_DIR}/subnormal_use_default.ll )
      list( APPEND rel_files generic/${ARG_LIB_DIR}/subnormal_use_default.ll )
    endif()
  endif()

  foreach( l ${source_list} )
    file( READ ${l} file_list )
    string( REPLACE "\n" ";" file_list ${file_list} )
    get_filename_component( dir ${l} DIRECTORY )
    foreach( f ${file_list} )
      list( FIND objects ${f} found )
      if( found EQUAL  -1 )
        list( APPEND objects ${f} )
        list( APPEND rel_files ${dir}/${f} )
        # FIXME: This should really go away
        file( TO_CMAKE_PATH ${dir}/${f} src_loc )
        get_filename_component( fdir ${src_loc} DIRECTORY )

        set_source_files_properties( ${dir}/${f}
          PROPERTIES COMPILE_FLAGS "-I ${fdir}" )
      endif()
    endforeach()
  endforeach()

  set( ${OUT_LIST} ${rel_files} PARENT_SCOPE )

endfunction(libclc_configure_lib_source OUT_LIST)
