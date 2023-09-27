if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|IntelLLVM" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
endif()

foreach(plugin ${SYCL_ENABLE_PLUGINS})
  add_subdirectory(${plugin})
endforeach()

# level_zero, cuda, hip, and native_cpu plugins depends today on unified_runtime plugin
# and unified_runtime plugin is not an independent plugin, adding it explicitly
if (NOT "unified_runtime" IN_LIST SYCL_ENABLE_PLUGINS)
  if ("level_zero" IN_LIST SYCL_ENABLE_PLUGINS OR
          "hip" IN_LIST SYCL_ENABLE_PLUGINS OR
          "cuda" IN_LIST SYCL_ENABLE_PLUGINS OR
          "native_cpu" IN_LIST SYCL_ENABLE_PLUGINS)
    add_subdirectory(unified_runtime)
  endif()
endif()
