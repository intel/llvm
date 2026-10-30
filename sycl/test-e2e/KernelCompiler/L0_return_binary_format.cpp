// REQUIRES: sycl-jit, level_zero, level_zero_dev_kit, aspect-usm_shared_allocations
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
//
// A bundle built from SYCL source via sycl_ext_oneapi_kernel_compiler must
// expose its image as native Level Zero code, as the backend_content spec
// says, not as the SPIR-V the JIT produced. Regression test for
// CMPLRLLVM-79017.

#include <level_zero/ze_api.h>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

const char *Source = R"(
#include <sycl/ext/oneapi/free_function_kernel_properties.hpp>
namespace syclexp = sycl::ext::oneapi::experimental;

extern "C" SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void store42(int *ptr) { *ptr = 42; }
)";

static bool isSPIRV(const std::vector<std::byte> &b) {
  return b.size() >= 4 && b[0] == std::byte{0x03} && b[1] == std::byte{0x02} &&
         b[2] == std::byte{0x23} && b[3] == std::byte{0x07};
}

int main() {
  // Needed when linked statically against the loader.
  ze_result_t status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  assert(status == ZE_RESULT_SUCCESS);

  sycl::queue q;
  sycl::context ctxt = q.get_context();
  sycl::device d = q.get_device();

  auto src = syclexp::create_kernel_bundle_from_source(
      ctxt, syclexp::source_language::sycl, std::string{Source});
  auto exe = syclexp::build(src, std::vector{d});
  assert(exe.ext_oneapi_has_kernel("store42"));

  // Run through SYCL first.
  sycl::kernel k = exe.ext_oneapi_get_kernel("store42");
  int *ptr = sycl::malloc_shared<int>(1, q);
  *ptr = 0;
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(ptr);
     cgh.single_task(k);
   }).wait();
  assert(*ptr == 42);

  // Now find the image, load its content as a native module and run the same
  // kernel through Level Zero interop.
  auto ZeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctxt);
  auto ZeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);

  ze_kernel_handle_t ZeKernel = nullptr;
  ze_module_handle_t ZeModule = nullptr;
  for (const auto &img : exe) {
    std::vector<std::byte> bytes = img.ext_oneapi_get_backend_content();
    assert(!bytes.empty());
    assert(!isSPIRV(bytes) && "backend content must be native, not SPIR-V");

    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        nullptr,
        ZE_MODULE_FORMAT_NATIVE,
        bytes.size(),
        reinterpret_cast<unsigned char *>(bytes.data()),
        nullptr,
        nullptr};
    ze_module_handle_t Module;
    status = zeModuleCreate(ZeContext, ZeDevice, &moduleDesc, &Module, nullptr);
    assert(status == ZE_RESULT_SUCCESS);

    uint32_t count = 0;
    status = zeModuleGetKernelNames(Module, &count, nullptr);
    assert(status == ZE_RESULT_SUCCESS);
    std::vector<const char *> names(count);
    status = zeModuleGetKernelNames(Module, &count, names.data());
    assert(status == ZE_RESULT_SUCCESS);
    for (const char *name : names) {
      if (std::string{name}.find("store42") == std::string::npos)
        continue;
      ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                     name};
      status = zeKernelCreate(Module, &kernelDesc, &ZeKernel);
      assert(status == ZE_RESULT_SUCCESS);
      ZeModule = Module;
      break;
    }
    if (ZeKernel)
      break;
    zeModuleDestroy(Module);
  }
  assert(ZeKernel && "kernel not found in any image's native content");

  sycl::kernel k_twin = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                sycl::bundle_state::executable>({ZeModule},
                                                                ctxt),
       ZeKernel},
      ctxt);
  int *ptr_twin = sycl::malloc_shared<int>(1, q);
  *ptr_twin = 1;
  q.submit([&](sycl::handler &cgh) {
     cgh.set_args(ptr_twin);
     cgh.single_task(k_twin);
   }).wait();
  assert(*ptr_twin == *ptr);

  sycl::free(ptr, q);
  sycl::free(ptr_twin, q);
  return 0;
}
