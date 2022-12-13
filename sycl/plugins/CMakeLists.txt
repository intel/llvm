set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|IntelLLVM" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
endif()

foreach(plugin ${SYCL_ENABLE_PLUGINS})
  add_subdirectory(${plugin})
endforeach()

