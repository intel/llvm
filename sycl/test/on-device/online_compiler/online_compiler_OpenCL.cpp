// REQUIRES: opencl

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -DRUN_KERNELS -lOpenCL -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -lOpenCL -o %th.out
// RUN: %RUN_ON_HOST %th.out

// This test checks INTEL feature class online_compiler for OpenCL.
// All OpenCL specific code is kept here and the common part that can be
// re-used by other backends is kept in online_compiler_common.hpp file.

// TODO: The testing of CM compilation is temporarily turned OFF on Linux
// due to problems with dependencies on libclangFEWrapper.so which is not
// currently included into NEO package on Linux.
#ifdef _WIN32
#define COMPILE_CM_KERNEL 1
#endif

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/online_compiler.hpp>

#include <vector>

using byte = unsigned char;

#ifdef RUN_KERNELS
sycl::kernel getSYCLKernelWithIL(sycl::context &Context,
                                 const std::vector<byte> &IL) {
  cl_int Err;
  cl_program ClProgram =
      clCreateProgramWithIL(Context.get(), IL.data(), IL.size(), &Err);
  if (Err != CL_SUCCESS)
    throw sycl::runtime_error("clCreateProgramWithIL() failed", Err);

  Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
  if (Err != CL_SUCCESS)
    throw sycl::runtime_error("clBuildProgram() failed", Err);

  cl_kernel ClKernel = clCreateKernel(ClProgram, "my_kernel", &Err);
  if (Err != CL_SUCCESS)
    throw sycl::runtime_error("clCreateKernel() failed", Err);

  return sycl::kernel(ClKernel, Context);
}
#endif // RUN_KERNELS

#include "online_compiler_common.hpp"
