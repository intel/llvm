// REQUIRES: opencl, opencl_icd

// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out

// Checks that aspect::ext_oneapi_atomic16 on the OpenCL backend matches the
// device's cl_ext_float_atomics fp16 capabilities, so devices that report the
// extension but lack e.g. half atomic add (Gen12) do not get the aspect.

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

#include <cassert>
#include <string>

using namespace sycl;

int main() {
  queue Queue;
  device Dev = Queue.get_device();
  cl_device_id CLDev = get_native<backend::opencl>(Dev);

  size_t ExtSize = 0;
  clGetDeviceInfo(CLDev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ExtSize);
  std::string ExtStr(ExtSize, '\0');
  clGetDeviceInfo(CLDev, CL_DEVICE_EXTENSIONS, ExtSize, &ExtStr.front(),
                  nullptr);

  bool Result = false;
  if (ExtStr.find("cl_ext_float_atomics") != std::string::npos) {
    cl_device_fp_atomic_capabilities_ext Caps = 0;
    clGetDeviceInfo(CLDev, CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT,
                    sizeof(Caps), &Caps, nullptr);
    constexpr cl_device_fp_atomic_capabilities_ext Required =
        CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT |
        CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
        CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
        CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT |
        CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT |
        CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT;
    Result = (Caps & Required) == Required;
  }
  assert(Dev.has(aspect::ext_oneapi_atomic16) == Result &&
         "The Result value differs from the implemented atomic16 check on "
         "the OpenCL backend.");
  return 0;
}
