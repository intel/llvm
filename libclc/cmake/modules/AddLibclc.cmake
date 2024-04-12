<<<<<<< HEAD
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
    set(dummy_in "${CMAKE_BINARY_DIR}/lib/clc/libclc_dummy_in.cc")
    add_custom_command( OUTPUT ${dummy_in}
      COMMAND ${CMAKE_COMMAND} -E touch ${dummy_in} )
    set(long_widths l32 l64)
    set(char_signedness signed unsigned)
    if( ${obj_suffix} STREQUAL "libspirv-nvptx64--nvidiacl.bc")
      set( obj_suffix_mangled "libspirv-nvptx64-nvidia-cuda.bc")
    elseif( ${obj_suffix} STREQUAL "libspirv-amdgcn--amdhsa.bc")
      set( obj_suffix_mangled "libspirv-amdgcn-amd-amdhsa.bc")
    else()
      set( obj_suffix_mangled "${obj_suffix}")
    endif()
    # All permutations of [l32, l64] and [signed, unsigned]
    foreach(long_width ${long_widths})
      foreach(signedness ${char_signedness})
        # Remangle
        set( builtins_remangle_path
            "${LIBCLC_LIBRARY_OUTPUT_INTDIR}/remangled-${long_width}-${signedness}_char.${obj_suffix_mangled}" )
        add_custom_command( OUTPUT "${builtins_remangle_path}"
          COMMAND libclc-remangler
          -o "${builtins_remangle_path}"
          --long-width=${long_width}
          --char-signedness=${signedness}
          --input-ir="$<TARGET_PROPERTY:prepare-${obj_suffix},TARGET_FILE>"
          ${dummy_in}
          DEPENDS "${builtins_obj_path}" "prepare-${obj_suffix}" libclc-remangler ${dummy_in})
        add_custom_target( "remangled-${long_width}-${signedness}_char.${obj_suffix_mangled}" ALL
          DEPENDS "${builtins_remangle_path}" "${dummy_in}")
        set_target_properties("remangled-${long_width}-${signedness}_char.${obj_suffix_mangled}"
          PROPERTIES TARGET_FILE "${builtins_remangle_path}")

        # Add dependency to top-level pseudo target to ease making other
        # targets dependent on libclc.
        add_dependencies(${ARG_PARENT_TARGET} "remangled-${long_width}-${signedness}_char.${obj_suffix_mangled}")

        # Keep remangled variants
        install(
          FILES ${builtins_remangle_path}
          DESTINATION ${CMAKE_INSTALL_DATADIR}/clc )
      endforeach()
    endforeach()

    # For remangler tests we do not care about long_width, or signedness, as it
    # performs no substitutions.
    # Collect all remangler tests in libclc-remangler-tests to later add
    # dependency against check-libclc.
    set(libclc-remangler-tests)
    set(libclc-remangler-test-no 0)
    set(libclc-remangler-target-ir
         "$<TARGET_PROPERTY:opt.${obj_suffix},TARGET_FILE>"
         "${LIBCLC_LIBRARY_OUTPUT_INTDIR}/builtins.link.${obj_suffix}"
         "$<TARGET_PROPERTY:prepare-${obj_suffix},TARGET_FILE>")
    foreach(target-ir ${libclc-remangler-target-ir})
      math(EXPR libclc-remangler-test-no "${libclc-remangler-test-no}+1")
      set(current-test "libclc-remangler-test-${obj_suffix}-${libclc-remangler-test-no}")
      add_custom_target(${current-test}
        COMMAND libclc-remangler
        --long-width=l32
        --char-signedness=signed
        --input-ir=${target-ir}
        ${dummy_in} -t -o -
        DEPENDS "${builtins_obj_path}" "prepare-${obj_suffix}" "${dummy_in}" libclc-remangler)
      list(APPEND libclc-remangler-tests ${current-test})
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
=======
# Compiles an OpenCL C - or assembles an LL file - to bytecode
#
# Arguments:
# * TRIPLE <string>
#     Target triple for which to compile the bytecode file.
# * INPUT <string>
#     File to compile/assemble to bytecode
# * OUTPUT <string>
#     Bytecode file to generate
# * EXTRA_OPTS <string> ...
#     List of compiler options to use. Note that some are added by default.
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
#
# Depends on the libclc::clang and libclc::llvm-as targets for compiling and
# assembling, respectively.
function(compile_to_bc)
  cmake_parse_arguments(ARG
    ""
    "TRIPLE;INPUT;OUTPUT"
    "EXTRA_OPTS;DEPENDENCIES"
    ${ARGN}
  )

  # If this is an LLVM IR file (identified soley by its file suffix),
  # pre-process it with clang to a temp file, then assemble that to bytecode.
  set( TMP_SUFFIX )
  get_filename_component( FILE_EXT ${ARG_INPUT} EXT )
  if( NOT ${FILE_EXT} STREQUAL ".ll" )
    # Pass '-c' when not running the preprocessor
    set( PP_OPTS -c )
  else()
    set( PP_OPTS -E;-P )
    set( TMP_SUFFIX .tmp )
  endif()

  set( TARGET_ARG )
  if( ARG_TRIPLE )
    set( TARGET_ARG "-target" ${ARG_TRIPLE} )
  endif()

  add_custom_command(
    OUTPUT ${ARG_OUTPUT}${TMP_SUFFIX}
    COMMAND libclc::clang
      ${TARGET_ARG}
      ${PP_OPTS}
      ${ARG_EXTRA_OPTS}
      -MD -MF ${ARG_OUTPUT}.d -MT ${ARG_OUTPUT}${TMP_SUFFIX}
      # LLVM 13 enables standard includes by default - we don't want
      # those when pre-processing IR. We disable it unconditionally.
      $<$<VERSION_GREATER_EQUAL:${LLVM_PACKAGE_VERSION},13.0.0>:-cl-no-stdinc>
      -emit-llvm
      -o ${ARG_OUTPUT}${TMP_SUFFIX}
      -x cl
      ${ARG_INPUT}
    DEPENDS
      libclc::clang
      ${ARG_INPUT}
      ${ARG_DEPENDENCIES}
    DEPFILE ${ARG_OUTPUT}.d
  )

  if( ${FILE_EXT} STREQUAL ".ll" )
    add_custom_command(
      OUTPUT ${ARG_OUTPUT}
      COMMAND libclc::llvm-as -o ${ARG_OUTPUT} ${ARG_OUTPUT}${TMP_SUFFIX}
      DEPENDS libclc::llvm-as ${ARG_OUTPUT}${TMP_SUFFIX}
    )
  endif()
endfunction()

# Links together one or more bytecode files
#
# Arguments:
# * TARGET <string>
#     Custom target to create
# * INPUT <string> ...
#     List of bytecode files to link together
function(link_bc)
  cmake_parse_arguments(ARG
    ""
    "TARGET"
    "INPUTS"
    ${ARGN}
  )

  add_custom_command(
    OUTPUT ${ARG_TARGET}.bc
    COMMAND libclc::llvm-link -o ${ARG_TARGET}.bc ${ARG_INPUTS}
    DEPENDS libclc::llvm-link ${ARG_INPUTS}
  )

  add_custom_target( ${ARG_TARGET} ALL DEPENDS ${ARG_TARGET}.bc )
  set_target_properties( ${ARG_TARGET} PROPERTIES TARGET_FILE ${ARG_TARGET}.bc )
endfunction()

# Decomposes and returns variables based on a libclc triple and architecture
# combination. Returns data via one or more optional output variables.
#
# Arguments:
# * TRIPLE <string>
#     libclc target triple to query
# * DEVICE <string>
#     libclc device to query
#
# Optional Arguments:
# * CPU <var>
#     Variable name to be set to the target CPU
# * ARCH_SUFFIX <var>
#     Variable name to be set to the triple/architecture suffix
# * CLANG_TRIPLE <var>
#     Variable name to be set to the normalized clang triple
function(get_libclc_device_info)
  cmake_parse_arguments(ARG
    ""
    "TRIPLE;DEVICE;CPU;ARCH_SUFFIX;CLANG_TRIPLE"
    ""
    ${ARGN}
  )

  if( NOT ARG_TRIPLE OR NOT ARG_DEVICE )
    message( FATAL_ERROR "Must provide both TRIPLE and DEVICE" )
  endif()

  string( REPLACE "-" ";" TRIPLE  ${ARG_TRIPLE} )
  list( GET TRIPLE 0 ARCH )

  # Some targets don't have a specific device architecture to target
  if( ARG_DEVICE STREQUAL none OR ARCH STREQUAL spirv OR ARCH STREQUAL spirv64 )
    set( cpu )
    set( arch_suffix "${ARG_TRIPLE}" )
  else()
    set( cpu "${ARG_DEVICE}" )
    set( arch_suffix "${ARG_DEVICE}-${ARG_TRIPLE}" )
  endif()

  if( ARG_CPU )
    set( ${ARG_CPU} ${cpu} PARENT_SCOPE )
  endif()

  if( ARG_ARCH_SUFFIX )
    set( ${ARG_ARCH_SUFFIX} ${arch_suffix} PARENT_SCOPE )
  endif()

  # Some libclc targets are not real clang triples: return their canonical
  # triples.
  if( ARCH STREQUAL spirv OR ARCH STREQUAL clspv )
    set( ARG_TRIPLE "spir--" )
  elseif( ARCH STREQUAL spirv64 OR ARCH STREQUAL clspv64 )
    set( ARG_TRIPLE "spir64--" )
  endif()

  if( ARG_CLANG_TRIPLE )
    set( ${ARG_CLANG_TRIPLE} ${ARG_TRIPLE} PARENT_SCOPE )
  endif()
endfunction()
>>>>>>> 72f9881c3ffcf4be6361c3e4312d91c9c8d94a98
