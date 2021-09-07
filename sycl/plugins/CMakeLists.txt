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

set( SYCL_PLUGIN_TARGETS
     pi_opencl
     pi_level_zero)

if(SYCL_BUILD_PI_ESIMD_CPU)
  list(APPEND SYCL_PLUGIN_TARGETS pi_esimd_cpu)
endif()

if(SYCL_BUILD_PI_CUDA)
  list(APPEND SYCL_PLUGIN_TARGETS pi_cuda)
endif()

foreach(pi_plugin ${SYCL_PLUGIN_TARGETS})
  if (MSVC)
    # by defining __SYCL_BUILD_SYCL_DLL, we can use __declspec(dllexport)
    # which are individually tagged for all pi* symbols in pi.h
    target_compile_definitions(${pi_plugin} PRIVATE __SYCL_BUILD_SYCL_DLL)
  else()
    # we set the visibility of all symbols 'hidden' by default.
    # In pi.h file, we set exported symbols with visibility==default individually
    target_compile_options(${pi_plugin} PUBLIC -fvisibility=hidden)

    # This script file is used to allow exporting pi* symbols only.
    # All other symbols are regarded as local (hidden)
    set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/../ld-version-script.txt")

    # Filter symbols based on the scope defined in the script file,
    # and export pi* function symbols in the library.
    target_link_libraries(${pi_plugin} PRIVATE "-Wl,--version-script=${linker_script}")
  endif()

endforeach()

