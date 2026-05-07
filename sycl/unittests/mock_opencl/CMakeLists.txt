get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

# On Windows: mock provides libOpenCL.dll (always).
# On Linux with static OCL adapter: ocl_dynamic_lib.cpp dlopens libOpenCL.so.1,
# so the mock must be named OpenCL with VERSION 1 so CMake produces both
# libOpenCL.so.1 and libOpenCL.so symlinks.
# On Linux with dynamic OCL adapter: adapter links to libOpenCL.so.1 at runtime,
# so the mock must also be named OpenCL.
if(WIN32)
  set(LIBNAME OpenCL)
elseif(UR_STATIC_ADAPTER_OPENCL)
  set(LIBNAME OpenCL)
  set(MOCK_OCL_VERSION 1)
else()
  # On Linux, mock is always OpenCL (the ICD), regardless of adapter linking mode
  set(LIBNAME OpenCL)
  set(MOCK_OCL_VERSION 1)
endif()

add_library(mockOpenCL SHARED EXCLUDE_FROM_ALL mock_opencl.cpp)
set_target_properties(mockOpenCL PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${SYCL_BINARY_DIR}/unittests/lib
  LIBRARY_OUTPUT_NAME ${LIBNAME}
  RUNTIME_OUTPUT_DIRECTORY ${SYCL_BINARY_DIR}/unittests/lib
  RUNTIME_OUTPUT_NAME ${LIBNAME}
)
if(DEFINED MOCK_OCL_VERSION)
  set_target_properties(mockOpenCL PROPERTIES VERSION ${MOCK_OCL_VERSION})
endif()
