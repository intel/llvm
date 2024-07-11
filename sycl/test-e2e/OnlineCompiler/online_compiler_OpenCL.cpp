// REQUIRES: opencl, opencl_icd, cm-compiler
// UNSUPPORTED: accelerator

// RUN: %{build} -DRUN_KERNELS %opencl_lib -o %t.out
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
sycl::kernel getSYCLKernelWithIL(sycl::queue &Queue,
                                 const std::vector<byte> &IL) {
  sycl::context Context = Queue.get_context();

  cl_int Err;
  cl_program ClProgram =
      clCreateProgramWithIL(sycl::get_native<sycl::backend::opencl>(Context),
                            IL.data(), IL.size(), &Err);
  assert(Err == CL_SUCCESS);

  Err = clBuildProgram(ClProgram, 0, nullptr, nullptr, nullptr, nullptr);
  assert(Err == CL_SUCCESS);

  cl_kernel ClKernel = clCreateKernel(ClProgram, "my_kernel", &Err);
  assert(Err == CL_SUCCESS);

  return sycl::make_kernel<sycl::backend::opencl>(ClKernel, Context);
}
#endif // RUN_KERNELS

#include "online_compiler_common.hpp"
