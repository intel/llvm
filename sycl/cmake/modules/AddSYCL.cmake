function(add_sycl_library LIB_NAME TYPE)
  cmake_parse_arguments("ARG"
    ""
    "LINKER_SCRIPT"
    "SOURCES;INCLUDE_DIRS;LIBRARIES"
    ${ARGN}
  )
  add_library(${LIB_NAME} ${TYPE} ${ARG_SOURCES})
  target_include_directories(${LIB_NAME} PRIVATE ${ARG_INCLUDE_DIRS})
  target_link_libraries(${LIB_NAME} PRIVATE ${ARG_LIBRARIES})

  add_dependencies(sycl-runtime-libraries ${LIB_NAME})

  if (ARG_LINKER_SCRIPT AND UNIX AND NOT APPLE)
    target_link_libraries(${LIB_NAME} PRIVATE
      "-Wl,--version-script=${ARG_LINKER_SCRIPT}")
  endif()

  target_compile_definitions(${LIB_NAME} PRIVATE __SYCL_BUILD_SYCL_DLL)

  if (UNIX)
    target_compile_options(${LIB_NAME} PRIVATE -fvisibility=hidden)
  else()
    add_stripped_pdb(${LIB_NAME})
  endif()

  # TODO: Enabled for MSVC
  if (NOT MSVC AND SYCL_LIB_WITH_DEBUG_SYMBOLS)
    separate_arguments(CMAKE_CXX_FLAGS_DEBUG_SEPARATED UNIX_COMMAND "${CMAKE_CXX_FLAGS_DEBUG}")
    target_compile_options(${LIB_NAME} PRIVATE ${CMAKE_CXX_FLAGS_DEBUG_SEPARATED})
  endif()

  # TODO remove add_common_options
  add_common_options(${LIB_NAME})
endfunction()
