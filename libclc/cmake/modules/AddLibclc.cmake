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
# Depends on the clang, llvm-as, and llvm-link targets for compiling,
# assembling, and linking, respectively.
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
    COMMAND ${clang_exe}
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
      ${clang_target}
      ${ARG_INPUT}
      ${ARG_DEPENDENCIES}
    DEPFILE ${ARG_OUTPUT}.d
  )

  if( ${FILE_EXT} STREQUAL ".ll" )
    add_custom_command(
      OUTPUT ${ARG_OUTPUT}
      COMMAND ${llvm-as_exe} -o ${ARG_OUTPUT} ${ARG_OUTPUT}${TMP_SUFFIX}
      DEPENDS ${llvm-as_target} ${ARG_OUTPUT}${TMP_SUFFIX}
    )
  endif()
endfunction()

# Links together one or more bytecode files
#
# Arguments:
# * INTERNALIZE
#     Set if -internalize flag should be passed when linking
# * TARGET <string>
#     Custom target to create
# * INPUT <string> ...
#     List of bytecode files to link together
# * RSP_DIR <string>
#     Directory where a response file should be placed
#     (Only needed for WIN32 or CYGWIN)
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
function(link_bc)
  cmake_parse_arguments(ARG
    "INTERNALIZE"
    "TARGET;RSP_DIR"
    "INPUTS;DEPENDENCIES"
    ${ARGN}
  )

  set( LINK_INPUT_ARG ${ARG_INPUTS} )
  if( WIN32 OR CYGWIN )
    # Create a response file in case the number of inputs exceeds command-line
    # character limits on certain platforms.
    file( TO_CMAKE_PATH ${ARG_RSP_DIR}/${ARG_TARGET}.rsp RSP_FILE )
    # Turn it into a space-separate list of input files
    list( JOIN ARG_INPUTS " " RSP_INPUT )
    file( GENERATE OUTPUT ${RSP_FILE} CONTENT ${RSP_INPUT} )
    # Ensure that if this file is removed, we re-run CMake
    set_property( DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${RSP_FILE}
    )
    set( LINK_INPUT_ARG "@${RSP_FILE}" )
  endif()

  add_custom_command(
    OUTPUT ${ARG_TARGET}.bc
    COMMAND ${llvm-link_exe} $<$<BOOL:${ARG_INTERNALIZE}>:--internalize> -o ${ARG_TARGET}.bc ${LINK_INPUT_ARG}
    DEPENDS ${llvm-link_target} ${ARG_DEPENDENCIES} ${ARG_INPUTS} ${RSP_FILE}
  )

  add_custom_target( ${ARG_TARGET} ALL DEPENDS ${ARG_TARGET}.bc )
  set_target_properties( ${ARG_TARGET} PROPERTIES
    TARGET_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}.bc
    FOLDER "libclc/Device IR/Linking"
  )
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

# Runs opt and prepare-builtins on a bitcode file specified by lib_tgt
#
# ARGUMENTS:
# * LIB_TGT string
#     Target name that becomes dependent on the out file named LIB_TGT.bc
# * IN_FILE string
#     Target name of the input bytecode file
# * OUT_DIR string
#     Name of the directory where the output should be placed
# *  DEPENDENCIES <string> ...
#     List of extra dependencies to inject
function(process_bc out_file)
  cmake_parse_arguments(ARG
    ""
    "LIB_TGT;IN_FILE;OUT_DIR"
    "OPT_FLAGS;DEPENDENCIES"
    ${ARGN})
  add_custom_command( OUTPUT ${ARG_LIB_TGT}.bc
    COMMAND ${opt_exe} ${ARG_OPT_FLAGS} -o ${ARG_LIB_TGT}.bc
    ${ARG_IN_FILE}
    DEPENDS ${opt_target} ${ARG_IN_FILE} ${ARG_DEPENDENCIES}
  )
  add_custom_target( ${ARG_LIB_TGT}
    ALL DEPENDS ${ARG_LIB_TGT}.bc
    )
  set_target_properties( ${ARG_LIB_TGT}
    PROPERTIES TARGET_FILE ${ARG_LIB_TGT}.bc
    )

  set( builtins_opt_lib $<TARGET_PROPERTY:${ARG_LIB_TGT},TARGET_FILE> )

  # Add prepare target
  add_custom_command( OUTPUT ${ARG_OUT_DIR}/${out_file}
    COMMAND ${prepare_builtins_exe} -o ${ARG_OUT_DIR}/${out_file}
      ${builtins_opt_lib}
      DEPENDS ${builtins_opt_lib} ${ARG_LIB_TGT} ${prepare_builtins_target} )
  add_custom_target( prepare-${out_file} ALL
    DEPENDS ${ARG_OUT_DIR}/${out_file}
  )
  set_target_properties( prepare-${out_file}
    PROPERTIES TARGET_FILE ${ARG_OUT_DIR}/${out_file}
  )
endfunction()

# Compiles a list of library source files (provided by LIB_FILES/GEN_FILES) and
# compiles them to LLVM bytecode (or SPIR-V), links them together and optimizes
# them.
#
# For bytecode libraries, a list of ALIASES may optionally be provided to
# produce additional symlinks.
#
# Arguments:
#  * ARCH <string>
#      libclc architecture being built
#  * ARCH_SUFFIX <string>
#      libclc architecture/triple suffix
#  * TRIPLE <string>
#      Triple used to compile
#
# Optional Arguments:
# * CLC_INTERNAL
#     Pass if compiling the internal CLC builtin libraries, which are not
#     optimized and do not have aliases created.
#  * LIB_FILES <string> ...
#      List of files that should be built for this library
#  * GEN_FILES <string> ...
#      List of generated files (in build dir) that should be built for this library
#  * COMPILE_FLAGS <string> ...
#      Compilation options (for clang)
#  * OPT_FLAGS <string> ...
#      Optimization options (for opt)
#  * TARGET_ENV <string>
#      Prefix to give the final builtin library aliases
#  * ALIASES <string> ...
#      List of aliases
#  * INTERNAL_LINK_DEPENDENCIES <string> ...
#      A list of extra bytecode files to link into the builtin library. Symbols
#      from these link dependencies will be internalized during linking.
function(add_libclc_builtin_set)
  cmake_parse_arguments(ARG
    "CLC_INTERNAL"
    "ARCH;TRIPLE;ARCH_SUFFIX;TARGET_ENV;PARENT_TARGET"
    "LIB_FILES;GEN_FILES;COMPILE_FLAGS;OPT_FLAGS;ALIASES;INTERNAL_LINK_DEPENDENCIES"
    ${ARGN}
  )

  if( NOT ARG_ARCH OR NOT ARG_ARCH_SUFFIX OR NOT ARG_TRIPLE )
    message( FATAL_ERROR "Must provide ARCH, ARCH_SUFFIX, and TRIPLE" )
  endif()

  set( bytecode_files "" )
  foreach( file IN LISTS ARG_GEN_FILES ARG_LIB_FILES )
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
      get_filename_component( abs_path ${file} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
      file( RELATIVE_PATH root_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${abs_path} )
      set( input_file ${CMAKE_CURRENT_SOURCE_DIR}/${file} )
      set( output_file "${LIBCLC_ARCH_OBJFILE_DIR}/${root_rel_path}.bc" )
    endif()

    get_filename_component( file_dir ${file} DIRECTORY )

    compile_to_bc(
      TRIPLE ${ARG_TRIPLE}
      INPUT ${input_file}
      OUTPUT ${output_file}
      EXTRA_OPTS -fno-builtin -nostdlib
        "${ARG_COMPILE_FLAGS}" -I${CMAKE_CURRENT_SOURCE_DIR}/${file_dir}
      DEPENDENCIES generate_convert.cl clspv-generate_convert.cl
    )
    list( APPEND bytecode_files ${output_file} )
  endforeach()

  set( builtins_comp_lib_tgt builtins.comp.${ARG_ARCH_SUFFIX} )
  add_custom_target( ${builtins_comp_lib_tgt}
    DEPENDS ${bytecode_files}
  )
  set_target_properties( ${builtins_comp_lib_tgt} PROPERTIES FOLDER "libclc/Device IR/Comp" )

  if( NOT bytecode_files )
    message(FATAL_ERROR "Cannot create an empty builtins library")
  endif()

  set( builtins_link_lib_tgt builtins.link.${ARG_ARCH_SUFFIX} )

  if( NOT ARG_INTERNAL_LINK_DEPENDENCIES )
    link_bc(
      TARGET ${builtins_link_lib_tgt}
      INPUTS ${bytecode_files}
      RSP_DIR ${LIBCLC_ARCH_OBJFILE_DIR}
      DEPENDENCIES ${builtins_comp_lib_tgt}
    )
  else()
    # If we have libraries to link while internalizing their symbols, we need
    # two separate link steps; the --internalize flag applies to all link
    # inputs but the first.
    set( builtins_link_lib_tmp_tgt builtins.link.pre-deps.${ARG_ARCH_SUFFIX} )
    link_bc(
      TARGET ${builtins_link_lib_tmp_tgt}
      INPUTS ${bytecode_files}
      RSP_DIR ${LIBCLC_ARCH_OBJFILE_DIR}
      DEPENDENCIES ${builtins_comp_lib_tgt}
    )
    link_bc(
      INTERNALIZE
      TARGET ${builtins_link_lib_tgt}
      INPUTS $<TARGET_PROPERTY:${builtins_link_lib_tmp_tgt},TARGET_FILE>
        ${ARG_INTERNAL_LINK_DEPENDENCIES}
      RSP_DIR ${LIBCLC_ARCH_OBJFILE_DIR}
      DEPENDENCIES ${builtins_link_lib_tmp_tgt}
    )
  endif()

  # For the CLC internal builtins, exit here - we only optimize the targets'
  # entry points once we've linked the CLC buitins into them
  if( ARG_CLC_INTERNAL )
    return()
  endif()

  set( builtins_link_lib $<TARGET_PROPERTY:${builtins_link_lib_tgt},TARGET_FILE> )

  add_custom_command( OUTPUT ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
    DEPENDS ${builtins_link_lib} prepare_builtins )

  if( ARG_ARCH STREQUAL spirv OR ARG_ARCH STREQUAL spirv64 )
    set( spv_suffix ${ARG_ARCH_SUFFIX}.spv )
    add_custom_command( OUTPUT ${spv_suffix}
      COMMAND ${llvm-spirv_exe} ${spvflags} -o ${spv_suffix} ${builtins_link_lib}
      DEPENDS ${llvm-spirv_target} ${builtins_link_lib} ${builtins_link_lib_tgt}
    )
    add_custom_target( "prepare-${spv_suffix}" ALL DEPENDS "${spv_suffix}" )
    set_target_properties( "prepare-${spv_suffix}" PROPERTIES FOLDER "libclc/Device IR/Prepare" )
    install( FILES ${CMAKE_CURRENT_BINARY_DIR}/${spv_suffix}
       DESTINATION "${CMAKE_INSTALL_DATADIR}/clc" )

    return()
  endif()

  set( builtins_opt_lib_tgt builtins.opt.${ARG_ARCH_SUFFIX} )

  process_bc(${ARG_ARCH_SUFFIX}.bc
    LIB_TGT ${builtins_opt_lib_tgt}
    IN_FILE ${builtins_link_lib}
    OUT_DIR ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
    OPT_FLAGS ${ARG_OPT_FLAGS}
    DEPENDENCIES ${builtins_link_lib_tgt} ${LIBCLC_LIBRARY_OUTPUT_INTDIR})

  # Add dependency to top-level pseudo target to ease making other
  # targets dependent on libclc.
  set( obj_suffix ${ARG_ARCH_SUFFIX}.bc )
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
          COMMAND ${libclc-remangler_exe}
          -o "${builtins_remangle_path}"
          --long-width=${long_width}
          --char-signedness=${signedness}
          --input-ir=${builtins_lib}
          ${dummy_in}
          DEPENDS ${builtins_lib} ${libclc-remangler_target} ${dummy_in})
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
        COMMAND ${libclc-remangler_exe}
        --long-width=l32
        --char-signedness=signed
        --input-ir=${target-ir}
        ${dummy_in} -t -o -
        DEPENDS ${builtins_lib} "${dummy_in}" ${libclc-remangler_target})
      list(APPEND libclc-remangler-tests ${current-test})
    endforeach()
  endif()

  # nvptx-- targets don't include workitem builtins
  if( NOT ARG_TRIPLE MATCHES ".*ptx.*--$" )
    add_test( NAME external-calls-${obj_suffix}
      COMMAND ./check_external_calls.sh ${builtins-lib}
      WORKING_DIRECTORY ${LIBCLC_LIBRARY_OUTPUT_INTDIR} )
    set_tests_properties( external-calls-${obj_suffix}
      PROPERTIES ENVIRONMENT "LLVM_CONFIG=${LLVM_CONFIG}" )
  endif()

  foreach( a ${$ARG_ALIASES} )
    set( alias_suffix "${ARG_TARGET_ENV}${a}-${ARG_TRIPLE}.bc" )
    add_libclc_alias( ${alias_suffix}
      ${arch_suffix}
      PARENT_TARGET ${ARG_PARENT_TARGET})
  endforeach( a )

endfunction(add_libclc_builtin_set)

# Produces a list of libclc source files by walking over SOURCES files in a
# given directory. Outputs the list of files in LIB_FILE_LIST.
#
# LIB_FILE_LIST may be pre-populated and is appended to.
#
# Arguments:
# * CLC_INTERNAL
#     Pass if compiling the internal CLC builtin libraries, which have a
#     different directory structure.
# * LIB_ROOT_DIR <string>
#     Root directory containing target's lib files, relative to libclc root
#     directory. If not provided, is set to '.'.
# * LIB_DIR <string>
#     Name of the directory containing the target's lib files. If not provided,
#     is set to 'lib'.
# * DIRS <string> ...
#     List of directories under LIB_ROOT_DIR to walk over searching for SOURCES
#     files
function(libclc_configure_lib_source LIB_FILE_LIST)
  cmake_parse_arguments(ARG
    "CLC_INTERNAL"
    "LIB_DIR;LIB_ROOT_DIR"
    "DIRS"
    ${ARGN}
  )

  if( NOT ARG_LIB_ROOT_DIR )
    set(ARG_LIB_ROOT_DIR  ".")
  endif()

  if( NOT ARG_LIB_DIR )
    set(ARG_LIB_DIR  "lib")
  endif()

  # Enumerate SOURCES* files
  set( source_list )
  foreach( l ${ARG_DIRS} )
    foreach( s "SOURCES" "SOURCES_${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}" )
      if( ARG_CLC_INTERNAL )
        file( TO_CMAKE_PATH ${ARG_LIB_ROOT_DIR}/${ARG_LIB_DIR}/${l}/${s} file_loc )
      else()
        file( TO_CMAKE_PATH ${ARG_LIB_ROOT_DIR}/${l}/${ARG_LIB_DIR}/${s} file_loc )
      endif()
      file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${file_loc} loc )
      # Prepend the location to give higher priority to
      # specialized implementation
      if( EXISTS ${loc} )
        set( source_list ${file_loc} ${source_list} )
      endif()
    endforeach()
  endforeach()

  ## Add the generated convert files here to prevent adding the ones listed in
  ## SOURCES
  set( rel_files ${${LIB_FILE_LIST}} ) # Source directory input files, relative to the root dir
  set( objects ${${LIB_FILE_LIST}} )   # A "set" of already-added input files

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

  set( ${LIB_FILE_LIST} ${rel_files} PARENT_SCOPE )
endfunction(libclc_configure_lib_source LIB_FILE_LIST)
