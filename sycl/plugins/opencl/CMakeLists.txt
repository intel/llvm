#TODO:
#1. Figure out why CMP0057 has to be set. Should have been taken care of earlier in the build
#2. Use AddLLVM to modify the build and access config options
#cmake_policy(SET CMP0057 NEW)
#include(AddLLVM)

# Plugin for OpenCL
# Create Shared library for libpi_opencl.so.
#TODO: remove dependency on pi.hpp in sycl project.
#TODO: Currently, the pi.hpp header is common between sycl and plugin library sources. 
#This can be changed by copying the pi.hpp file in the plugins project.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../source/version.rc.in
  ${CMAKE_CURRENT_BINARY_DIR}/version.rc
  @ONLY)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(pi_opencl SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "pi_opencl.cpp"
  ${CMAKE_CURRENT_BINARY_DIR}/version.rc
  )

add_dependencies(pi_opencl
  ocl-icd
  ocl-headers
)

add_dependencies(sycl-toolchain pi_opencl)

set_target_properties(pi_opencl PROPERTIES LINKER_LANGUAGE CXX)

#preprocessor definitions for compiling a target's sources. We do not need it for pi_opencl
target_include_directories(pi_opencl PRIVATE "${sycl_inc_dir}")

if (MSVC)
  target_compile_definitions(pi_opencl PRIVATE __SYCL_BUILD_SYCL_DLL)
  target_link_libraries(pi_opencl PRIVATE shlwapi)
  target_link_libraries( pi_opencl
    PRIVATE OpenCL::Headers
    PRIVATE ${OpenCL_LIBRARIES}
  )
else()
  # we set the visibility of all symbols 'hidden' by default.
  # In *.cpp files, we set exported symbols with visibility==default individually
  target_compile_options(pi_opencl PUBLIC -fvisibility=hidden)

  # This script file is used to allow exporting pi* symbols only.
  # All other symbols are regarded as local (hidden)
  set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/../ld-version-script.txt")

  #link pi_opencl with OpenCL headers and ICD Loader.
  target_link_libraries( pi_opencl
    PRIVATE OpenCL::Headers
    PRIVATE ${OpenCL_LIBRARIES}
    # Filter symbols based on the scope defined in the script file,
    # and export pi* function symbols in the library. 
    PRIVATE "-Wl,--version-script=${linker_script}"
  )
endif()

add_common_options(pi_opencl)

set_target_properties(pi_opencl PROPERTIES
                      VERSION ${SYCL_VERSION_STRING}
                      SOVERSION ${SYCL_MAJOR_VERSION})
install(TARGETS pi_opencl
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_opencl
    RUNTIME DESTINATION "bin" COMPONENT pi_opencl)
