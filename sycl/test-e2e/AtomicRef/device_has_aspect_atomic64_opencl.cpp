// REQUIRES: opencl, opencl_icd

// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out

#include <CL/cl.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  queue Queue;
  device Dev = Queue.get_device();
  bool Result;
  // Get size for string of extensions
  size_t ExtSize;
  clGetDeviceInfo(get_native<backend::opencl>(Dev), CL_DEVICE_EXTENSIONS, 0,
                  nullptr, &ExtSize);
  std::string ExtStr(ExtSize, '\0');

  // Collect device extensions into string ExtStr
  clGetDeviceInfo(get_native<backend::opencl>(Dev), CL_DEVICE_EXTENSIONS,
                  ExtSize, &ExtStr.front(), nullptr);

  // Check that ExtStr has two extensions related to atomic64 support
  if (ExtStr.find("cl_khr_int64_base_atomics") == std::string::npos ||
      ExtStr.find("cl_khr_int64_extended_atomics") == std::string::npos)
    Result = false;
  else
    Result = true;
  assert(Dev.has(aspect::atomic64) == Result &&
         "The Result value differs from the implemented atomic64 check on "
         "the OpenCL backend.");
  return 0;
}
