// REQUIRES: opencl, opencl_icd, cm-compiler

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -DRUN_KERNELS %opencl_lib -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s %opencl_lib -o %th.out
// RUN: %HOST_RUN_PLACEHOLDER %th.out

// This test checks ext::intel feature class online_compiler for OpenCL.
// All OpenCL specific code is kept here and the common part that can be
// re-used by other backends is kept in online_compiler_common.hpp file.

#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>
#include <sycl/ext/intel/online_compiler.hpp>

#include <vector>

using byte = unsigned char;

#ifdef RUN_KERNELS
sycl::kernel getSYCLKernelWithIL(sycl::queue &Queue,
                                 const std::vector<byte> &IL) {
  sycl::context Context = Queue.get_context();

  cl_int Err;
  cl_program ClProgram =
      clCreateProgramWithIL(sycl::get_native<sycl::backend::opencl>(Context),
                            IL.data(), IL.size(), &Err);
  if (Err != CL_SUCCESS)
    throw sycl::runtime_error("clCreateProgramWithIL() failed", Err);

  Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
  if (Err != CL_SUCCESS)
    throw sycl::runtime_error("clBuildProgram() failed", Err);

  cl_kernel ClKernel = clCreateKernel(ClProgram, "my_kernel", &Err);
  if (Err != CL_SUCCESS)
    throw sycl::runtime_error("clCreateKernel() failed", Err);

  return sycl::make_kernel<sycl::backend::opencl>(ClKernel, Context);
}
#endif // RUN_KERNELS

#include "online_compiler_common.hpp"
