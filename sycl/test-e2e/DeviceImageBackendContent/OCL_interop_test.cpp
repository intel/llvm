// REQUIRES: target-spir, opencl, opencl_icd, aspect-usm_shared_allocations
// RUN: %{build} %opencl_lib -fno-sycl-dead-args-optimization -o %t.out
// RUN: %{run} %t.out
// XFAIL: accelerator
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16914
//
#include <sycl/backend.hpp>
#include <sycl/detail/cl.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>
#include <vector>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::nd_range_kernel<1>)) void iota(int *ptr) {
  size_t id = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  ptr[id] = 42;
}

int main() {
  sycl::queue q;
  sycl::context ctxt = q.get_context();
  sycl::device d = ctxt.get_devices()[0];
  // First, run the kernel using the SYCL API.

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctxt);
  sycl::kernel_id iota_id = syclexp::get_kernel_id<iota>();
  sycl::kernel k_iota = bundle.get_kernel(iota_id);

  int *ptr = sycl::malloc_shared<int>(1, q);
  *ptr = 0;
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(ptr);
     cgh.parallel_for(sycl::nd_range{{1}, {1}}, k_iota);
   }).wait();
  // Now, run the kernel by first getting its image as an executable,
  // making an OCL kernel out of it and then making a SYCL kernel out of
  // the OCL kernel. Run this kernel on the SYCL API and verify
  // that it has the same result as the kernel that was run directly on SYCL
  // API. First, get a kernel bundle that contains the kernel "iota".
  auto exe_bndl = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctxt, {d},
      [&](const sycl::device_image<sycl::bundle_state::executable> &img) {
        return img.has_kernel(iota_id, d);
      });
  assert(!exe_bndl.empty());
  std::vector<std::byte> bytes;
  const sycl::device_image<sycl::bundle_state::executable> &img =
      *(exe_bndl.begin());
  bytes = img.ext_oneapi_get_backend_content();
  auto clContext = sycl::get_native<sycl::backend::opencl>(ctxt);
  auto clDevice = sycl::get_native<sycl::backend::opencl>(d);
  cl_int status;
  auto clProgram = clCreateProgramWithIL(
      clContext, reinterpret_cast<const void *>(bytes.data()), bytes.size(),
      &status);
  assert(status == CL_SUCCESS);
  status = clBuildProgram(clProgram, 1, &clDevice, "", nullptr, nullptr);
  assert(status == CL_SUCCESS);
  auto clKernel = clCreateKernel(clProgram, "__sycl_kernel_iota", &status);
  assert(status == CL_SUCCESS);
  sycl::kernel k_iota_twin =
      sycl::make_kernel<sycl::backend::opencl>(clKernel, ctxt);
  int *ptr_twin = sycl::malloc_shared<int>(1, q);
  *ptr_twin = 1;
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(ptr_twin);
     cgh.parallel_for(sycl::nd_range{{1}, {1}}, k_iota_twin);
   }).wait();
  assert(*ptr_twin == *ptr);
  sycl::free(ptr, q);
  sycl::free(ptr_twin, q);
}
