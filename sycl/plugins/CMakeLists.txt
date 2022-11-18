set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|IntelLLVM" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
endif()

foreach(plugin ${SYCL_ENABLE_PLUGINS})
  # The foreach doesn't seem to iterate the list in a stable order,
  # but we have to make sure that unified_runtime is added before
  # the level_zero since the later has a dependency on the earlier.
  if (NOT "${plugin}" IN_LIST SYCL_ENABLED_PLUGINS)

    if ("${plugin}" MATCHES "level_zero")
      if (NOT "unified_runtime" IN_LIST SYCL_ENABLED_PLUGINS)
        add_subdirectory("unified_runtime")
        list(APPEND SYCL_ENABLED_PLUGINS "unified_runtime")
      endif()
    endif()

    add_subdirectory(${plugin})
    list(APPEND SYCL_ENABLED_PLUGINS ${plugin})

  endif()
endforeach()

