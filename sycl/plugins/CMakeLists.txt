if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|IntelLLVM" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
endif()

foreach(plugin ${SYCL_ENABLE_PLUGINS})
  add_subdirectory(${plugin})
endforeach()

# level_zero and opencl plugins depend today on unified_runtime plugin
# and unified_runtime plugin is not an independent plugin, adding it explicitly
if ("level_zero" IN_LIST SYCL_ENABLE_PLUGINS OR "opencl" IN_LIST SYCL_ENABLE_PLUGINS)
  add_subdirectory(unified_runtime)
endif()
