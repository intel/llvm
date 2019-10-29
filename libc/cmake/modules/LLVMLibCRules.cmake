
# A rule for self contained header file targets.
# This rule merely copies the header file from the current source directory to
# the current binary directory.
# Usage:
#     add_header(
#       <target name>
#       HDR <header file>
#     )
function(add_header target_name)
  cmake_parse_arguments(
    "ADD_HEADER"
    ""    # No optional arguments
    "HDR" # Single value arguments
    "DEPENDS"    # No multi value arguments
    ${ARGN}
  )
  if(NOT ADD_HEADER_HDR)
    message(FATAL_ERROR "'add_header' rules requires the HDR argument specifying a headef file.")
  endif()

  set(dest_file ${CMAKE_CURRENT_BINARY_DIR}/${ADD_HEADER_HDR})
  set(src_file ${CMAKE_CURRENT_SOURCE_DIR}/${ADD_HEADER_HDR})

  add_custom_command(
    OUTPUT ${dest_file}
    COMMAND cp ${src_file} ${dest_file}
    DEPENDS ${src_file}
  )

  add_custom_target(
    ${target_name}
    DEPENDS ${dest_file}
  )

  if(ADD_HEADER_DEPENDS)
  add_dependencies(
    ${target_name} ${ADD_HEADER_DEPENDS}
  )
  endif()
endfunction(add_header)

# A rule for generated header file targets.
# Usage:
#     add_gen_header(
#       <target name>
#       DEF_FILE <.h.def file>
#       GEN_HDR <generated header file name>
#       PARAMS <list of name=value pairs>
#       DATA_FILES <list input data files>
#     )
function(add_gen_header target_name)
  cmake_parse_arguments(
    "ADD_GEN_HDR"
    "" # No optional arguments
    "DEF_FILE;GEN_HDR" # Single value arguments
    "PARAMS;DATA_FILES"     # Multi value arguments
    ${ARGN}
  )
  if(NOT ADD_GEN_HDR_DEF_FILE)
    message(FATAL_ERROR "`add_gen_hdr` rule requires DEF_FILE to be specified.")
  endif()
  if(NOT ADD_GEN_HDR_GEN_HDR)
    message(FATAL_ERROR "`add_gen_hdr` rule requires GEN_HDR to be specified.")
  endif()

  set(out_file ${CMAKE_CURRENT_BINARY_DIR}/${ADD_GEN_HDR_GEN_HDR})
  set(in_file ${CMAKE_CURRENT_SOURCE_DIR}/${ADD_GEN_HDR_DEF_FILE})

  set(fq_data_files "")
  if(ADD_GEN_HDR_DATA_FILES)
    foreach(data_file IN LISTS ADD_GEN_HDR_DATA_FILES)
      list(APPEND fq_data_files "${CMAKE_CURRENT_SOURCE_DIR}/${data_file}")
    endforeach(data_file)
  endif()

  set(replacement_params "")
  if(ADD_GEN_HDR_PARAMS)
    list(APPEND replacement_params "-P" ${ADD_GEN_HDR_PARAMS})
  endif()

  set(gen_hdr_script "${LIBC_BUILD_SCRIPTS_DIR}/gen_hdr.py")

  add_custom_command(
    OUTPUT ${out_file}
    COMMAND ${gen_hdr_script} -o ${out_file} ${in_file} ${replacement_params}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${in_file} ${fq_data_files} ${gen_hdr_script}
  )

  add_custom_target(
    ${target_name}
    DEPENDS ${out_file}
  )
endfunction(add_gen_header)

set(ENTRYPOINT_OBJ_TARGET_TYPE "ENTRYPOINT_OBJ")

# A rule for entrypoint object targets.
# Usage:
#     add_entrypoint_object(
#       <target_name>
#       SRCS <list of .cpp files>
#       HDRS <list of .h files>
#       DEPENDS <list of dependencies>
#     )
function(add_entrypoint_object target_name)
  cmake_parse_arguments(
    "ADD_ENTRYPOINT_OBJ"
    "" # No optional arguments
    "" # No single value arguments
    "SRCS;HDRS;DEPENDS"  # Multi value arguments
    ${ARGN}
  )
  if(NOT ADD_ENTRYPOINT_OBJ_SRCS)
    message(FATAL_ERROR "`add_entrypoint_object` rule requires SRCS to be specified.")
  endif()
  if(NOT ADD_ENTRYPOINT_OBJ_HDRS)
    message(FATAL_ERROR "`add_entrypoint_object` rule requires HDRS to be specified.")
  endif()

  add_library(
    "${target_name}_objects"
    # We want an object library as the objects will eventually get packaged into
    # an archive (like libc.a).
    OBJECT
    ${ADD_ENTRYPOINT_OBJ_SRCS}
    ${ADD_ENTRYPOINT_OBJ_HDRS}
  )
  target_compile_options(
    ${target_name}_objects
    BEFORE
    PRIVATE
      -fpie -std=${LLVM_CXX_STD_default}
  )
  target_include_directories(
    ${target_name}_objects
    PRIVATE
      "${LIBC_BUILD_DIR}/include;${LIBC_SOURCE_DIR};${LIBC_BUILD_DIR}"
  )
  add_dependencies(
    ${target_name}_objects
    support_common_h
  )
  if(ADD_ENTRYPOINT_OBJ_DEPENDS)
    add_dependencies(
      ${target_name}_objects
      ${ADD_ENTRYPOINT_OBJ_DEPENDS}
    )
  endif()

  set(object_file_raw "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_raw.o")
  set(object_file "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.o")

  add_custom_command(
    OUTPUT ${object_file_raw}
    DEPENDS $<TARGET_OBJECTS:${target_name}_objects>
    COMMAND ${CMAKE_LINKER} -r $<TARGET_OBJECTS:${target_name}_objects> -o ${object_file_raw}
  )

  add_custom_command(
    OUTPUT ${object_file}
    DEPENDS ${object_file_raw}
    COMMAND ${CMAKE_OBJCOPY} --add-symbol "${target_name}=.llvm.libc.entrypoint.${target_name}:0,function,weak,global" ${object_file_raw} ${object_file}
  )

  add_custom_target(
    ${target_name}
    ALL
    DEPENDS ${object_file}
  )
  set_target_properties(
    ${target_name}
    PROPERTIES
      "TARGET_TYPE" ${ENTRYPOINT_OBJ_TARGET_TYPE}
      "OBJECT_FILE" ${object_file}
      "OBJECT_FILE_RAW" ${object_file_raw}
  )
endfunction(add_entrypoint_object)

# A rule to build a library from a collection of entrypoint objects.
# Usage:
#     add_entrypoint_library(
#       DEPENDS <list of add_entrypoint_object targets>
#     )
function(add_entrypoint_library target_name)
  cmake_parse_arguments(
    "ENTRYPOINT_LIBRARY"
    "" # No optional arguments
    "" # No single value arguments
    "DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT ENTRYPOINT_LIBRARY_DEPENDS)
    message(FATAL_ERROR "'add_entrypoint_library' target requires a DEPENDS list of 'add_entrypoint_object' targets.")
  endif()

  set(obj_list "")
  foreach(dep IN LISTS ENTRYPOINT_LIBRARY_DEPENDS)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    string(COMPARE EQUAL ${dep_type} ${ENTRYPOINT_OBJ_TARGET_TYPE} dep_is_entrypoint)
    if(NOT dep_is_entrypoint)
      message(FATAL_ERROR "Dependency '${dep}' of 'add_entrypoint_collection' is not an 'add_entrypoint_object' target.")
    endif()
    get_target_property(target_obj_file ${dep} "OBJECT_FILE")
    list(APPEND obj_list "${target_obj_file}")
  endforeach(dep)

  set(library_file "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${target_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_custom_command(
    OUTPUT ${library_file}
    COMMAND ${CMAKE_AR} -r ${library_file} ${obj_list}
    DEPENDS ${obj_list}
  )
  add_custom_target(
    ${target_name}
    ALL
    DEPENDS ${library_file}
  )
endfunction(add_entrypoint_library)

function(add_libc_unittest target_name)
  if(NOT LLVM_INCLUDE_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    "LIBC_UNITTEST"
    "" # No optional arguments
    "SUITE" # Single value arguments
    "SRCS;HDRS;DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT LIBC_UNITTEST_SRCS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a SRCS list of .cpp files.")
  endif()
  if(NOT LIBC_UNITTEST_DEPENDS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a DEPENDS list of 'add_entrypoint_object' targets.")
  endif()

  set(entrypoint_objects "")
  foreach(dep IN LISTS LIBC_UNITTEST_DEPENDS)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    string(COMPARE EQUAL ${dep_type} ${ENTRYPOINT_OBJ_TARGET_TYPE} dep_is_entrypoint)
    if(NOT dep_is_entrypoint)
      message(FATAL_ERROR "Dependency '${dep}' of 'add_entrypoint_unittest' is not an 'add_entrypoint_object' target.")
    endif()
    get_target_property(obj_file ${dep} "OBJECT_FILE_RAW")
    list(APPEND entrypoint_objects "${obj_file}")
  endforeach(dep)

  add_executable(
    ${target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_UNITTEST_SRCS}
    ${LIBC_UNITTEST_HDRS}
  )
  target_include_directories(
    ${target_name}
    PRIVATE
      ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include
      ${LLVM_MAIN_SRC_DIR}/utils/unittest/googlemock/include
      ${LIBC_SOURCE_DIR}
  )
  target_link_libraries(${target_name} PRIVATE ${entrypoint_objects} gtest_main gtest)
  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(
    ${target_name}
    ${LIBC_UNITTEST_DEPENDS}
    gtest
  )
  add_custom_command(
    TARGET ${target_name}
    POST_BUILD
    COMMAND $<TARGET_FILE:${target_name}>
  )
  if(LIBC_UNITTEST_SUITE)
    add_dependencies(
      ${LIBC_UNITTEST_SUITE}
      ${target_name}
    )
  endif()
endfunction(add_libc_unittest)
