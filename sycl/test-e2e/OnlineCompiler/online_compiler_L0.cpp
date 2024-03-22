// REQUIRES: level_zero, level_zero_dev_kit, cm-compiler

// RUN: %{build} -DRUN_KERNELS %level_zero_options -o %t.out
// RUN: %{run} %t.out

// This test checks ext::intel feature class online_compiler for Level-Zero.
// All Level-Zero specific code is kept here and the common part that can be
// re-used by other backends is kept in online_compiler_common.hpp file.

#include <sycl/ext/intel/experimental/online_compiler.hpp>
#include <sycl/detail/core.hpp>

#include <vector>

// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using byte = unsigned char;

#ifdef RUN_KERNELS
sycl::kernel getSYCLKernelWithIL(sycl::queue &Queue,
                                 const std::vector<byte> &IL) {

  ze_module_desc_t ZeModuleDesc = {};
  ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  ZeModuleDesc.inputSize = IL.size();
  ZeModuleDesc.pInputModule = IL.data();
  ZeModuleDesc.pBuildFlags = "";
  ZeModuleDesc.pConstants = nullptr;

  sycl::context Context = Queue.get_context();
  sycl::device Device = Queue.get_device();
  auto ZeDevice =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Device);
  auto ZeContext =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Context);

  ze_module_build_log_handle_t ZeBuildLog;
  ze_module_handle_t ZeModule;
  ze_result_t ZeResult = zeModuleCreate(ZeContext, ZeDevice, &ZeModuleDesc,
                                        &ZeModule, &ZeBuildLog);
  if (ZeResult != ZE_RESULT_SUCCESS)
    throw sycl::runtime_error();

  ze_kernel_handle_t ZeKernel = nullptr;

  ze_kernel_desc_t ZeKernelDesc{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0,
                                "my_kernel"};
  ZeResult = zeKernelCreate(ZeModule, &ZeKernelDesc, &ZeKernel);
  if (ZeResult != ZE_RESULT_SUCCESS)
    throw sycl::runtime_error();
  sycl::kernel_bundle<sycl::bundle_state::executable> SyclKB =
      sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                               sycl::bundle_state::executable>(
          {ZeModule, sycl::ext::oneapi::level_zero::ownership::keep}, Context);

  auto Kernel = sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
      {SyclKB, ZeKernel, sycl::ext::oneapi::level_zero::ownership::keep},
      Context);

  // Should not throw an exception
  try {
    auto num_args = Kernel.get_info<sycl::info::kernel::num_args>();
    (void)num_args;
  } catch (sycl::exception &e) {
    assert(false && "Using \"info::kernel::num_args\" query for valid kernel "
                    "should not throw an exception.");
  }

  return Kernel;
}
#endif // RUN_KERNELS

#include "online_compiler_common.hpp"
