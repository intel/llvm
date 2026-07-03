get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

# On Windows: mock provides libOpenCL.dll (always).
# On Linux: mock is always OpenCL (the ICD), with VERSION 1 so CMake produces
# both libOpenCL.so.1 and libOpenCL.so symlinks. The static adapter dlopens
# libOpenCL.so.1, and the dynamic adapter links to it at build time.
if(WIN32)
  set(LIBNAME OpenCL)
else()
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
