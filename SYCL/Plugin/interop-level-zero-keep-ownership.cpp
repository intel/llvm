// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %level_zero_options
// RUN: env SYCL_BE=PI_LEVEL_ZERO %GPU_RUN_PLACEHOLDER %t.out

// Test for Level Zero interop API where SYCL RT doesn't take ownership

#include <CL/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>
// clang-format on

using namespace cl::sycl;

int main() {

  // Creat SYCL platform/device
  device Device(gpu_selector{});
  platform Platform = Device.get_info<info::device::platform>();

  // Create native Level-Zero context
  ze_context_handle_t ZeContext;
  ze_context_desc_t ZeContextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};
  auto ZeDriver = Platform.get_native<backend::level_zero>();
  auto ZeDevice = Device.get_native<backend::level_zero>();
  zeContextCreate(ZeDriver, &ZeContextDesc, &ZeContext);

  { // Scope in which SYCL interop context object is live
    vector_class<device> Devices{};
    Devices.push_back(Device);
    auto ContextInterop = level_zero::make<context>(
        Devices, ZeContext, level_zero::ownership::keep);
  }

  // Verifies that Level-Zero context is not destroyed by SYCL RT yet.
  zeContextDestroy(ZeContext);
  return 0;
}
