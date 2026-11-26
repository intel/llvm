get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

# On Windows, the mock is named OpenCL.dll.
# On Linux, its name depends on the adapter mode:
# With a static adapter, it is named libOpenCL.so.1 (with a
# libOpenCL.so symlink), which the adapter dlopens.
# With a dynamic adapter, it is named libur_adapter_opencl.so so SYCL can
# resolve OpenCL symbols through the dynamic adapter.
if(WIN32)
  set(LIBNAME OpenCL)
elseif(UR_STATIC_ADAPTER_OPENCL)
  set(LIBNAME OpenCL)
  set(MOCK_OCL_VERSION 1)
else()
  set(LIBNAME ur_adapter_opencl)
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
