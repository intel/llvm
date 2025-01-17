// REQUIRES: level_zero, level_zero_dev_kit, aspect-usm_shared_allocations
// RUN: %{build} %level_zero_options -fno-sycl-dead-args-optimization -o %t.out
// RUN: %{run} %t.out
//
#include <level_zero/ze_api.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
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
  sycl::device d([](const sycl::device &d) {
    return d.get_backend() == sycl::backend::ext_oneapi_level_zero;
  });
  sycl::queue q{d};
  sycl::context ctxt = q.get_context();

  // The following ifndef is required due to a number of limitations of free
  // function kernels. See CMPLRLLVM-61498.
  // TODO: Remove it once these limitations are no longer there.
#ifndef __SYCL_DEVICE_ONLY__
  // First, run the kernel using the SYCL API.
  auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctxt);
  sycl::kernel_id iota_id = syclexp::get_kernel_id<iota>();
  sycl::kernel k_iota = Bundle.get_kernel(iota_id);
  int *ptr = sycl::malloc_shared<int>(1, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(ptr);
     cgh.parallel_for(sycl::nd_range{{1}, {1}}, k_iota);
   }).wait();

  // Now, run the kernel by first getting its image as an executable,
  // making an L0 kernel out of it and then making a SYCL kernel out of
  // the L0 kernel. Run this kernel on the SYCL API and verify
  // that it has the same result as the kernel that was run directly on SYCL
  // API. First, get a kernel bundle that contains the kernel "iota".
  auto exe_bndl = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctxt, {d},
      [&](const sycl::device_image<sycl::bundle_state::executable> &img) {
        return img.has_kernel(iota_id, d);
      });
  std::vector<std::byte> bytes;
  const sycl::device_image<sycl::bundle_state::executable> &img =
      *(exe_bndl.begin());
  bytes = img.ext_oneapi_get_backend_content();

  auto ZeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctxt);
  auto ZeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);

  ze_result_t status;
  ze_module_desc_t moduleDesc = {
      ZE_STRUCTURE_TYPE_MODULE_DESC,
      nullptr,
      ZE_MODULE_FORMAT_IL_SPIRV,
      bytes.size(),
      reinterpret_cast<unsigned char *>(bytes.data()),
      nullptr,
      nullptr};
  ze_module_handle_t ZeModule;
  status = zeModuleCreate(ZeContext, ZeDevice, &moduleDesc, &ZeModule, nullptr);
  assert(status == ZE_RESULT_SUCCESS);

  ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                 "__sycl_kernel_iota"};
  ze_kernel_handle_t ZeKernel;
  status = zeKernelCreate(ZeModule, &kernelDesc, &ZeKernel);
  assert(status == ZE_RESULT_SUCCESS);
  sycl::kernel k_iota_twin =
      sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
          {sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                    sycl::bundle_state::executable>({ZeModule},
                                                                    ctxt),
           ZeKernel},
          ctxt);
  int *ptr_twin = sycl::malloc_shared<int>(1, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(ptr_twin);
     cgh.parallel_for(sycl::nd_range{{1}, {1}}, k_iota_twin);
   }).wait();
  assert(*ptr_twin == *ptr);
  sycl::free(ptr, q);
  sycl::free(ptr_twin, q);
#endif
}
