// REQUIRES: opencl, opencl_icd

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_INTERNAL_API %s -o %t.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <cassert>
#include <exception>
#include <sycl/sycl.hpp>
#include <vector>

#define CL_CHECK_ERRORS(ERR)                                                   \
  if (ERR != CL_SUCCESS) {                                                     \
    throw std::runtime_error("Error in OpenCL object creation.");              \
  }

cl_platform_id selectOpenCLPlatform() {
  cl_int err = 0;
  cl_uint num_of_platforms = 0;

  err = clGetPlatformIDs(0, NULL, &num_of_platforms);
  CL_CHECK_ERRORS(err);

  std::vector<cl_platform_id> platforms(num_of_platforms);

  err = clGetPlatformIDs(num_of_platforms, &platforms[0], 0);
  CL_CHECK_ERRORS(err);

  return platforms[0];
}

cl_device_id selectOpenCLDevice(cl_platform_id platform) {
  cl_int err = 0;
  cl_uint num_of_devices = 0;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, 0, &num_of_devices);
  CL_CHECK_ERRORS(err);

  std::vector<cl_device_id> devices(num_of_devices);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_of_devices,
                       &devices[0], 0);
  CL_CHECK_ERRORS(err);

  return devices[0];
}

cl_context createOpenCLContext(cl_platform_id platform, cl_device_id device) {
  cl_int err = 0;
  cl_context context;
  std::vector<cl_context_properties> context_props(3);

  context_props[0] = CL_CONTEXT_PLATFORM;
  context_props[1] = cl_context_properties(platform);
  context_props.back() = 0;

  context = clCreateContext(&context_props[0], 1, &device, 0, 0, &err);
  CL_CHECK_ERRORS(err);

  return context;
}

int main() {
  cl_platform_id ocl_platform = selectOpenCLPlatform();
  sycl::platform syclPlt(ocl_platform);
  assert(ocl_platform == syclPlt.get() &&
         "SYCL returns invalid OpenCL platform id");

  cl_device_id ocl_device = selectOpenCLDevice(ocl_platform);
  sycl::device syclDev(ocl_device);
  assert(ocl_device == syclDev.get() &&
         "SYCL returns invalid OpenCL device id");

  cl_context ocl_context = createOpenCLContext(ocl_platform, ocl_device);
  sycl::context syclContext(ocl_context);
  assert(ocl_context == syclContext.get() &&
         "SYCL returns invalid OpenCL context id");

  return 0;
}
