set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
endif()

if(SYCL_BUILD_PI_CUDA)
  add_subdirectory(cuda)
endif()

if(SYCL_BUILD_PI_ROCM)
  add_subdirectory(rocm)
endif()

add_subdirectory(opencl)
add_subdirectory(level_zero)

# TODO : Remove 'if (NOT MSVC)' when CM_EMU supports Windows
# environment
if (NOT MSVC)
  if (SYCL_BUILD_PI_ESIMD_CPU)
    add_subdirectory(esimd_cpu)
  endif()
endif()
