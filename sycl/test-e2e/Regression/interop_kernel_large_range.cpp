// REQUIRES: opencl, opencl_icd, gpu, aspect-usm_shared_allocations

// RUN: %{build} -o %t.out %opencl_lib
// RUN: %{run} %t.out

// Interop kernels have no device image and always use size_t
// id/range semantics, so the launch must succeed.

#include <CL/opencl.h>
#include <sycl/backend.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <limits>

using namespace sycl;

const char KernelSource[] = "__kernel void big_range(__global uchar *out) {\
        if (get_global_id(0) == 0) out[0] = 1;\
    }";
const size_t KernelSourceSize = sizeof(KernelSource);
const char *Sources[1] = {KernelSource};

int main() {
  queue Q{};
  context Ctx = Q.get_context();
  device Dev = Q.get_device();

  cl_int Err;
  cl_program Prog = clCreateProgramWithSource(
      get_native<backend::opencl>(Ctx), 1, Sources, &KernelSourceSize, &Err);
  assert(Err == CL_SUCCESS);
  cl_device_id CLDev = get_native<backend::opencl>(Dev);
  Err = clBuildProgram(Prog, 1, &CLDev, nullptr, nullptr, nullptr);
  assert(Err == CL_SUCCESS);
  cl_kernel CLKernel = clCreateKernel(Prog, "big_range", &Err);
  assert(Err == CL_SUCCESS);

  kernel Kernel = make_kernel<backend::opencl>(CLKernel, Ctx);

  uint8_t *Out = malloc_shared<uint8_t>(1, Q);

  auto Launch = [&](size_t Global, size_t Local) {
    *Out = 0;
    Q.submit([&](handler &CGH) {
       CGH.set_arg(0, Out);
       CGH.parallel_for(nd_range<1>{Global, Local}, Kernel);
     }).wait_and_throw();
    assert(*Out == 1 && "Kernel did not run");
  };

  constexpr size_t IntMax = std::numeric_limits<int>::max();
  Launch(1024, 16);          // sanity: below INT_MAX
  Launch(IntMax + 1ull, 16); // the regression: above INT_MAX

  free(Out, Q);
  clReleaseKernel(CLKernel);
  clReleaseProgram(Prog);
  return 0;
}
