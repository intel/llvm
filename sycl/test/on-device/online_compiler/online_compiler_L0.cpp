// REQUIRES: level_zero

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_source_dir -lze_loader %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/online_compiler.hpp>

#include <vector>

// clang-format off
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>
// clang-format on

using byte = unsigned char;

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
    throw sycl::INTEL::online_compile_error(std::string("ZeResult = ") +
                                            std::to_string(ZeResult));
  sycl::program SyclProgram =
      sycl::level_zero::make<sycl::program>(Context, ZeModule);
  return SyclProgram.get_kernel("my_kernel");
}

#include "online_compiler_common.hpp"
