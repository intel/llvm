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

  # FIXME: Unlike upstream LLVM, passing "-x cl" to these files across the
  # board causes too many changes to the resulting bytecode library. This needs
  # investigation. It's still required for the preprocessor step, though.
  set( XCL_OPT )
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
    set( XCL_OPT -x;cl )
  endif()


  set( TARGET_ARG )
  if( ARG_TRIPLE )
    set( TARGET_ARG "-target" ${ARG_TRIPLE} )
  endif()

  # Ensure the directory we are told to output to exists
  get_filename_component( ARG_OUTPUT_DIR ${ARG_OUTPUT} DIRECTORY )
  file( MAKE_DIRECTORY ${ARG_OUTPUT_DIR} )

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
      ${XCL_OPT}
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
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
function(link_bc)
  cmake_parse_arguments(ARG
    ""
    "TARGET"
    "INPUTS;DEPENDENCIES"
    ${ARGN}
  )

  set( LINK_INPUT_ARG ${ARG_INPUTS} )
  if( WIN32 OR CYGWIN )
    # Create a response file in case the number of inputs exceeds command-line
    # character limits on certain platforms.
    file( TO_CMAKE_PATH ${LIBCLC_ARCH_OBJFILE_DIR}/${ARG_TARGET}.rsp RSP_FILE )
    # Turn it into a space-separate list of input files
    list( JOIN ARG_INPUTS " " RSP_INPUT )
    file( WRITE ${RSP_FILE} ${RSP_INPUT} )
    # Ensure that if this file is removed, we re-run CMake
    set_property( DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${RSP_FILE}
    )
    set( LINK_INPUT_ARG "@${RSP_FILE}" )
  endif()

  add_custom_command(
    OUTPUT ${ARG_TARGET}.bc
    COMMAND libclc::llvm-link -o ${ARG_TARGET}.bc ${LINK_INPUT_ARG}
    DEPENDS libclc::llvm-link ${ARG_DEPENDENCIES} ${ARG_INPUTS} ${RSP_FILE}
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
    if( ARCH STREQUAL amdgcn )
      # AMDGCN needs libclc to be compiled to high bc version since all atomic
      # clang builtins need to be accessible
      set( cpu gfx940)
    else()
      set( cpu )
    endif()
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

function(add_libclc_alias alias target)
  cmake_parse_arguments(ARG "" "" PARENT_TARGET "" ${ARGN})

  if(CMAKE_HOST_UNIX AND NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(LIBCLC_LINK_OR_COPY create_symlink)
  else()
    set(LIBCLC_LINK_OR_COPY copy)
  endif()

  add_custom_command(
      OUTPUT ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${alias_suffix}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
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
    "GEN_FILES;FILES;ALIASES;GENERATE_TARGET;COMPILE_OPT;OPT_FLAGS"
    ${ARGN})

  string( TOUPPER "CLC_${ARCH}" CLC_TARGET_DEFINE )

  list( APPEND ARG_COMPILE_OPT
    -D__CLC_INTERNAL
    -D${CLC_TARGET_DEFINE}
    -I${PROJECT_SOURCE_DIR}/generic/include
    # FIXME: Fix libclc to not require disabling this noisy warning
    -Wno-bitwise-conditional-parentheses
  )

  set( bytecode_files "" )
  foreach( file IN LISTS ARG_GEN_FILES ARG_FILES )
    # We need to take each file and produce an absolute input file, as well
    # as a unique architecture-specific output file. We deal with a mix of
    # different input files, which makes this trickier.
    if( ${file} IN_LIST ARG_GEN_FILES )
      # Generated files are given just as file names, which we must make
      # absolute to the binary directory.
      set( input_file ${CMAKE_CURRENT_BINARY_DIR}/${file} )
      set( output_file "${LIBCLC_ARCH_OBJFILE_DIR}/${file}.bc" )
    else()
      # Other files are originally relative to each SOURCE file, which are
      # then make relative to the libclc root directory. We must normalize
      # the path (e.g., ironing out any ".."), then make it relative to the
      # root directory again, and use that relative path component for the
      # binary path.
      get_filename_component( abs_path ${file} ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR} )
      file( RELATIVE_PATH root_rel_path ${PROJECT_SOURCE_DIR} ${abs_path} )
      set( input_file ${PROJECT_SOURCE_DIR}/${file} )
      set( output_file "${LIBCLC_ARCH_OBJFILE_DIR}/${root_rel_path}.bc" )
    endif()

    get_filename_component( file_dir ${file} DIRECTORY )

    compile_to_bc(
      TRIPLE ${ARG_TRIPLE}
      INPUT ${input_file}
      OUTPUT ${output_file}
      EXTRA_OPTS -fno-builtin -nostdlib
          "${ARG_COMPILE_OPT}" -I${PROJECT_SOURCE_DIR}/${file_dir}
      DEPENDENCIES generate_convert.cl clspv-generate_convert.cl
    )
    list(APPEND bytecode_files ${output_file})
  endforeach()

  set( builtins_comp_lib_tgt builtins.comp.${arch_suffix} )
  add_custom_target( ${builtins_comp_lib_tgt}
    DEPENDS ${bytecode_files}
  )

  set( builtins_link_lib_tgt builtins.link.${arch_suffix} )
  link_bc(
    TARGET ${builtins_link_lib_tgt}
    INPUTS ${bytecode_files}
    DEPENDENCIES ${builtins_comp_lib_tgt}
  )

  set( builtins_link_lib $<TARGET_PROPERTY:${builtins_link_lib_tgt},TARGET_FILE> )

  set( builtins_opt_lib_tgt builtins.opt.${arch_suffix} )

  # Add opt target
  add_custom_command( OUTPUT ${builtins_opt_lib_tgt}.bc
    COMMAND libclc::opt ${ARG_OPT_FLAGS} -o ${builtins_opt_lib_tgt}.bc
      ${builtins_link_lib}
    DEPENDS libclc::opt ${builtins_link_lib} ${builtins_link_lib_tgt}
  )
  add_custom_target( ${builtins_opt_lib_tgt}
    ALL DEPENDS ${builtins_opt_lib_tgt}.bc
  )
  set_target_properties( ${builtins_opt_lib_tgt}
    PROPERTIES TARGET_FILE ${builtins_opt_lib_tgt}.bc
  )

  set( builtins_opt_lib $<TARGET_PROPERTY:${builtins_opt_lib_tgt},TARGET_FILE> )

  # Add prepare target
  set( obj_suffix ${arch_suffix}.bc )
  add_custom_command( OUTPUT ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
    COMMAND prepare_builtins -o ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
      ${builtins_opt_lib}
    DEPENDS ${builtins_opt_lib} ${builtins_opt_lib_tgt} prepare_builtins )
  add_custom_target( prepare-${obj_suffix} ALL
    DEPENDS ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
  )
  set_target_properties( prepare-${obj_suffix}
    PROPERTIES TARGET_FILE ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/${obj_suffix}
  )

  # Add dependency to top-level pseudo target to ease making other
  # targets dependent on libclc.
  add_dependencies(${ARG_PARENT_TARGET} prepare-${obj_suffix})
  set( builtins_lib $<TARGET_PROPERTY:prepare-${obj_suffix},TARGET_FILE> )

  install( FILES ${builtins_lib} DESTINATION ${CMAKE_INSTALL_DATADIR}/clc )

  # Generate remangled variants if requested
  if( LIBCLC_GENERATE_REMANGLED_VARIANTS )
    set( dummy_in ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/libclc_dummy_in.cc )
    add_custom_command( OUTPUT ${dummy_in}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
      COMMAND ${CMAKE_COMMAND} -E touch ${dummy_in}
    )
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
          COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
          COMMAND libclc::libclc-remangler
          -o "${builtins_remangle_path}"
          --long-width=${long_width}
          --char-signedness=${signedness}
          --input-ir=${builtins_lib}
          ${dummy_in}
          DEPENDS ${builtins_lib} libclc::libclc-remangler ${dummy_in})
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
    foreach(target-ir ${builtins_opt_lib} ${builtins_link_lib} ${builtins_lib})
      math(EXPR libclc-remangler-test-no "${libclc-remangler-test-no}+1")
      set(current-test "libclc-remangler-test-${obj_suffix}-${libclc-remangler-test-no}")
      add_custom_target(${current-test}
        COMMAND libclc::libclc-remangler
        --long-width=l32
        --char-signedness=signed
        --input-ir=${target-ir}
        ${dummy_in} -t -o -
        DEPENDS ${builtins_lib} "${dummy_in}" libclc::libclc-remangler)
      list(APPEND libclc-remangler-tests ${current-test})
    endforeach()
  endif()

  # nvptx-- targets don't include workitem builtins
  if( NOT ${t} MATCHES ".*ptx.*--$" )
    add_test( NAME external-calls-${obj_suffix}
      COMMAND ./check_external_calls.sh ${builtins-lib}
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

function(libclc_configure_lib_source OUT_LIST OUT_GEN_LIST)
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
      file( TO_CMAKE_PATH ${PROJECT_SOURCE_DIR}/${file_loc} loc )
      # Prepend the location to give higher priority to
      # specialized implementation
      if( EXISTS ${loc} )
        set( source_list ${file_loc} ${source_list} )
      endif()
    endforeach()
  endforeach()

  ## Add the generated convert files here to prevent adding the ones listed in
  ## SOURCES
  set( objects ${ARG_DEPS} )   # A "set" of already-added input files
  set( rel_files )             # Source directory input files, relative to the root dir
  set( gen_files ${ARG_DEPS} ) # Generated binary input files, relative to the binary dir

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
      # Only add each file once, so that targets can 'specialize' builtins
      if( NOT ${f} IN_LIST objects )
        list( APPEND objects ${f} )
        list( APPEND rel_files ${dir}/${f} )
      endif()
    endforeach()
  endforeach()

  set( ${OUT_LIST} ${rel_files} PARENT_SCOPE )
  set( ${OUT_GEN_LIST} ${gen_files} PARENT_SCOPE )

endfunction(libclc_configure_lib_source OUT_LIST OUT_GEN_LIST)
