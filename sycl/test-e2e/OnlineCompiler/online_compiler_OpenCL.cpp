// REQUIRES: opencl, opencl_icd, cm-compiler
// XFAIL: gpu || cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16406
// RUN: %{build} -Wno-error=deprecated-declarations -DRUN_KERNELS %opencl_lib -o %t.out
// RUN: %{run} %t.out

// This test checks ext::intel feature class online_compiler for OpenCL.
// All OpenCL specific code is kept here and the common part that can be
// re-used by other backends is kept in online_compiler_common.hpp file.

#include <sycl/backend.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/experimental/online_compiler.hpp>

#include <vector>

using byte = unsigned char;

#ifdef RUN_KERNELS
std::tuple<unsigned long, unsigned long> GetOCLVersion(sycl::device Device) {
  cl_int Err;
  cl_device_id ClDevice = sycl::get_native<sycl::backend::opencl>(Device);

  size_t VersionSize = 0;
  Err = clGetDeviceInfo(ClDevice, CL_DEVICE_VERSION, 0, nullptr, &VersionSize);
  assert(Err == CL_SUCCESS);

  std::string Version(VersionSize, '\0');
  Err = clGetDeviceInfo(ClDevice, CL_DEVICE_VERSION, VersionSize,
                        Version.data(), nullptr);
  assert(Err == CL_SUCCESS);

  std::string_view Prefix = "OpenCL ";
  size_t VersionBegin = Version.find_first_of(" ");
  size_t VersionEnd = Version.find_first_of(" ", VersionBegin + 1);
  size_t VersionSeparator = Version.find_first_of(".", VersionBegin + 1);

  bool HaveOCLPrefix =
      std::equal(Prefix.begin(), Prefix.end(), Version.begin());

  assert(HaveOCLPrefix && VersionBegin != std::string::npos &&
         VersionEnd != std::string::npos &&
         VersionSeparator != std::string::npos);

  std::string VersionMajor{Version.begin() + VersionBegin + 1,
                           Version.begin() + VersionSeparator};
  std::string VersionMinor{Version.begin() + VersionSeparator + 1,
                           Version.begin() + VersionEnd};

  unsigned long OCLMajor = strtoul(VersionMajor.c_str(), nullptr, 10);
  unsigned long OCLMinor = strtoul(VersionMinor.c_str(), nullptr, 10);

  assert(OCLMajor > 0 && (OCLMajor > 2 || OCLMinor <= 2) &&
         OCLMajor != UINT_MAX && OCLMinor != UINT_MAX);

  return std::make_tuple(OCLMajor, OCLMinor);
}

bool testSupported(sycl::queue &Queue) {
  if (Queue.get_backend() != sycl::backend::opencl)
    return false;

  sycl::device Device = Queue.get_device();
  auto [OCLMajor, OCLMinor] = GetOCLVersion(Device);

  // Creating a program from IL is only supported on >=2.1 or if
  // cl_khr_il_program is supported on the device.
  return (OCLMajor == 2 && OCLMinor >= 1) || OCLMajor > 2 ||
         Device.has_extension("cl_khr_il_program");
}

sycl::kernel getSYCLKernelWithIL(sycl::queue &Queue,
                                 const std::vector<byte> &IL) {
  sycl::context Context = Queue.get_context();

  cl_int Err = 0;
  cl_program ClProgram = 0;

  sycl::device Device = Queue.get_device();
  auto [OCLMajor, OCLMinor] = GetOCLVersion(Device);
  if ((OCLMajor == 2 && OCLMinor >= 1) || OCLMajor > 2) {
    // clCreateProgramWithIL is supported if OCL version >=2.1.
    ClProgram =
        clCreateProgramWithIL(sycl::get_native<sycl::backend::opencl>(Context),
                              IL.data(), IL.size(), &Err);
  } else {
    // Fall back to using extension function for building IR.
    using ApiFuncT =
        cl_program(CL_API_CALL *)(cl_context, const void *, size_t, cl_int *);
    ApiFuncT FuncPtr =
        reinterpret_cast<ApiFuncT>(clGetExtensionFunctionAddressForPlatform(
            sycl::get_native<sycl::backend::opencl>(Context.get_platform()),
            "clCreateProgramWithILKHR"));

    assert(FuncPtr != nullptr);

    ClProgram = FuncPtr(sycl::get_native<sycl::backend::opencl>(Context),
                        IL.data(), IL.size(), &Err);
  }
  assert(Err == CL_SUCCESS);

  Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
  assert(Err == CL_SUCCESS);

  cl_kernel ClKernel = clCreateKernel(ClProgram, "my_kernel", &Err);
  assert(Err == CL_SUCCESS);

  return sycl::make_kernel<sycl::backend::opencl>(ClKernel, Context);
}
#endif // RUN_KERNELS

#include "online_compiler_common.hpp"
