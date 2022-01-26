set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|IntelLLVM" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
endif()

if(SYCL_BUILD_PI_CUDA)
  add_subdirectory(cuda)
endif()

if(SYCL_BUILD_PI_HIP)
  add_subdirectory(hip)
endif()

add_subdirectory(opencl)
add_subdirectory(level_zero)

if(SYCL_BUILD_PI_ESIMD_EMULATOR)
  # TODO : Remove 'if (NOT MSVC)' when CM_EMU supports Windows
  # environment
  if (NOT MSVC)
    add_subdirectory(esimd_emulator)
  endif()
endif()
