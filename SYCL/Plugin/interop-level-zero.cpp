// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_BE=PI_LEVEL_ZERO %GPU_RUN_PLACEHOLDER %t.out

// Test for Level Zero interop API

#include <CL/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>
// clang-format on

using namespace cl::sycl;

int main() {
  queue Queue{};
  auto Context = Queue.get_info<info::queue::context>();
  auto Device = Queue.get_info<info::queue::device>();
  auto Platform = Device.get_info<info::device::platform>();

  // Get native Level Zero handles
  auto ZePlatform = Platform.get_native<backend::level_zero>();
  auto ZeDevice = Device.get_native<backend::level_zero>();
  auto ZeContext = Context.get_native<backend::level_zero>();
  auto ZeQueue = Queue.get_native<backend::level_zero>();

  // Create native Level-Zero context.
  // It then will be owned/destroyed by SYCL RT.
  ze_context_handle_t ZeContextInterop{};
  ze_context_desc_t ZeContextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};
  zeContextCreate(ZePlatform, &ZeContextDesc, &ZeContextInterop);

  // Re-create SYCL objects from native Level Zero handles
  auto PlatformInterop = level_zero::make<platform>(ZePlatform);
  auto DeviceInterop = level_zero::make<device>(PlatformInterop, ZeDevice);
  auto ContextInterop = level_zero::make<context>(PlatformInterop.get_devices(),
                                                  ZeContextInterop);
  auto QueueInterop = level_zero::make<queue>(ContextInterop, ZeQueue);

  // Check native handles
  assert(ZePlatform == PlatformInterop.get_native<backend::level_zero>());
  assert(ZeDevice == DeviceInterop.get_native<backend::level_zero>());
  assert(ZeContextInterop == ContextInterop.get_native<backend::level_zero>());
  assert(ZeQueue == QueueInterop.get_native<backend::level_zero>());

  // Verify re-created objects
  int Arr[] = {2};
  {
    cl::sycl::buffer<int, 1> Buf(Arr, 1);
    QueueInterop.submit([&](cl::sycl::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel>([=]() { Acc[0] *= 3; });
    });
  }
  assert(Arr[0] == 6);

  return 0;
}
