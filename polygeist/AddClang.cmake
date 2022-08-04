
macro(set_clang_windows_version_resource_properties name)
  if(DEFINED windows_resource_file)
    set_windows_version_resource_properties(${name} ${windows_resource_file}
      VERSION_MAJOR ${CLANG_VERSION_MAJOR}
      VERSION_MINOR ${CLANG_VERSION_MINOR}
      VERSION_PATCHLEVEL ${CLANG_VERSION_PATCHLEVEL}
      VERSION_STRING "${CLANG_VERSION} (${BACKEND_PACKAGE_STRING})"
      PRODUCT_NAME "clang")
  endif()
endmacro()

macro(add_clang_executable name)
  add_llvm_executable( ${name} ${ARGN} )
  set_target_properties(${name} PROPERTIES FOLDER "Clang executables")
  set_clang_windows_version_resource_properties(${name})
endmacro(add_clang_executable)

macro(add_clang_tool name)
  if (NOT CLANG_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_clang_executable(${name} ${ARGN})
  # add_dependencies(${name} clang-resource-headers)

  if (CLANG_BUILD_TOOLS)
    get_target_export_arg(${name} Clang export_to_clangtargets)
    install(TARGETS ${name}
      ${export_to_clangtargets}
      RUNTIME DESTINATION bin
      COMPONENT ${name})

    if(NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
                               DEPENDS ${name}
                               COMPONENT ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY CLANG_EXPORTS ${name})
  endif()
endmacro()

