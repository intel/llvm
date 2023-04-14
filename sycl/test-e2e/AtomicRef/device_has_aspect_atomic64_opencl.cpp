// REQUIRES: opencl, opencl_icd
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %opencl_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// XFAIL: hip
// Expected failure because hip does not have atomic64 check implementation

#include <CL/cl.h>
#include <CL/sycl.hpp>

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
