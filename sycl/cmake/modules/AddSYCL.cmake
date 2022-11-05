function(add_sycl_library LIB_NAME TYPE)
  cmake_parse_arguments("ARG"
    "TOOLCHAIN"
    "LINKER_SCRIPT"
    "SOURCES;INCLUDE_DIRS;LIBRARIES"
    ${ARGN}
  )
  add_library(${LIB_NAME} ${TYPE} ${ARG_SOURCES})
  target_include_directories(${LIB_NAME} PRIVATE ${ARG_INCLUDE_DIRS})
  target_link_libraries(${LIB_NAME} PRIVATE ${ARG_LIBRARIES})

  if (ARG_TOOLCHAIN)
    add_dependencies(sycl-toolchain ${LIB_NAME})
  endif()

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

  # TODO remove add_common_options
  add_common_options(${LIB_NAME})
endfunction()

function(add_sycl_plugin PLUGIN_NAME)
  cmake_parse_arguments("ARG"
    ""
    ""
    "SOURCES;INCLUDE_DIRS;LIBRARIES"
    ${ARGN}
  )

  add_sycl_library("pi_${PLUGIN_NAME}" SHARED
    TOOLCHAIN
    LINKER_SCRIPT "${PROJECT_SOURCE_DIR}/plugins/ld-version-script.txt"
    SOURCES ${ARG_SOURCES}
    INCLUDE_DIRS
      ${ARG_INCLUDE_DIRS}
      ${sycl_inc_dir}
    LIBRARIES
      ${ARG_LIBRARIES}
      OpenCL-Headers
  )

  install(TARGETS pi_${PLUGIN_NAME}
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_${PLUGIN_NAME}
    RUNTIME DESTINATION "bin" COMPONENT pi_${PLUGIN_NAME})

  set (manifest_file
    ${CMAKE_CURRENT_BINARY_DIR}/install_manifest_pi_${PLUGIN_NAME}.txt)
  add_custom_command(OUTPUT ${manifest_file}
    COMMAND "${CMAKE_COMMAND}"
    "-DCMAKE_INSTALL_COMPONENT=pi_${PLUGIN_NAME}"
    -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
    COMMENT "Deploying component pi_${PLUGIN_NAME}"
    USES_TERMINAL
  )
  add_custom_target(install-sycl-plugin-${PLUGIN_NAME}
    DEPENDS
      ${manifest_file} pi_${PLUGIN_NAME}
    )

  set_property(GLOBAL APPEND PROPERTY SYCL_TOOLCHAIN_INSTALL_COMPONENTS
    pi_${PLUGIN_NAME})
endfunction()
