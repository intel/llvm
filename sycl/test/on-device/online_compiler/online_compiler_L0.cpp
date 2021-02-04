// REQUIRES: level_zero, level_zero_headers

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %level_zero_include_dir -DRUN_KERNELS -lze_loader %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %level_zero_include_dir -lze_loader %s -o %th.out
// RUN: %RUN_ON_HOST %th.out

// This test checks INTEL feature class online_compiler for Level-Zero.
// All Level-Zero specific code is kept here and the common part that can be
// re-used by other backends is kept in online_compiler_common.hpp file.

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/online_compiler.hpp>

#include <vector>

// TODO: The testing of CM compilation is temporarily turned OFF on Linux
// due to problems with dependencies on libclangFEWrapper.so which is not
// currently included into NEO package on Linux.
#ifdef _WIN32
#define COMPILE_CM_KERNEL 1
#endif

// clang-format off
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>
// clang-format on

using byte = unsigned char;

#ifdef RUN_KERNELS
sycl::kernel getSYCLKernelWithIL(sycl::context &Context,
                                 const std::vector<byte> &IL) {

  ze_module_desc_t ZeModuleDesc = {};
  ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  ZeModuleDesc.inputSize = IL.size();
  ZeModuleDesc.pInputModule = IL.data();
  ZeModuleDesc.pBuildFlags = "";
  ZeModuleDesc.pConstants = nullptr;

  assert(Context.get_devices().size() == 1 && "Expected to have only 1 device");
  sycl::device Device = Context.get_devices()[0];
  auto ZeDevice = Device.get_native<sycl::backend::level_zero>();
  auto ZeContext = Context.get_native<sycl::backend::level_zero>();

  ze_module_build_log_handle_t ZeBuildLog;
  ze_module_handle_t ZeModule;
  ze_result_t ZeResult = zeModuleCreate(ZeContext, ZeDevice, &ZeModuleDesc,
                                        &ZeModule, &ZeBuildLog);
  if (ZeResult != ZE_RESULT_SUCCESS)
    throw sycl::runtime_error();
  sycl::program SyclProgram =
      sycl::level_zero::make<sycl::program>(Context, ZeModule);
  return SyclProgram.get_kernel("my_kernel");
}
#endif // RUN_KERNELS

#include "online_compiler_common.hpp"
