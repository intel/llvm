// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_BE=PI_LEVEL_ZERO SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: ze_debug-1,ze_debug4

// Test for Level Zero interop API

#include <sycl/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using namespace cl::sycl;

int main() {
#ifdef SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO
  queue Queue{};

  auto Event = Queue.single_task([=]() {});
  auto Context = Queue.get_info<info::queue::context>();
  auto Device = Queue.get_info<info::queue::device>();
  auto Platform = Device.get_info<info::device::platform>();

  // Get native Level Zero handles
  auto ZePlatform = get_native<backend::ext_oneapi_level_zero>(Platform);
  auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);
  auto ZeContext = get_native<backend::ext_oneapi_level_zero>(Context);
  auto ZeQueue = get_native<backend::ext_oneapi_level_zero>(Queue);
  auto ZeEvent = get_native<backend::ext_oneapi_level_zero>(Event);

  // Create native Level-Zero context.
  // It then will be owned/destroyed by SYCL RT.
  ze_context_handle_t ZeContextInterop{};
  ze_context_desc_t ZeContextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};
  zeContextCreate(ZePlatform, &ZeContextDesc, &ZeContextInterop);

  // Re-create SYCL objects from native Level Zero handles
  auto PlatformInterop =
      make_platform<backend::ext_oneapi_level_zero>(ZePlatform);
  auto DeviceInterop = make_device<backend::ext_oneapi_level_zero>(ZeDevice);

  backend_input_t<backend::ext_oneapi_level_zero, context> ContextInteropInput =
      {ZeContextInterop, Context.get_devices()};
  auto ContextInterop =
      make_context<backend::ext_oneapi_level_zero>(ContextInteropInput);

  backend_input_t<backend::ext_oneapi_level_zero, queue> QueueInteropInput = {
      ZeQueue, Queue.get_device(), ext::oneapi::level_zero::ownership::keep};
  auto QueueInterop = make_queue<backend::ext_oneapi_level_zero>(
      QueueInteropInput, ContextInterop);

  backend_input_t<backend::ext_oneapi_level_zero, event> EventInteropInput = {
      ZeEvent};
  // ZeEvent isn't owning the resource (it's owned by Event object), we cannot \
  // transfer ownership that we don't have. As such, use "keep".
  EventInteropInput.Ownership =
      cl::sycl::ext::oneapi::level_zero::ownership::keep;
  auto EventInterop = make_event<backend::ext_oneapi_level_zero>(
      EventInteropInput, ContextInterop);

  // Check native handles
  assert(ZePlatform ==
         get_native<backend::ext_oneapi_level_zero>(PlatformInterop));
  assert(ZeDevice == get_native<backend::ext_oneapi_level_zero>(DeviceInterop));
  assert(ZeContextInterop ==
         get_native<backend::ext_oneapi_level_zero>(ContextInterop));
  assert(ZeQueue == get_native<backend::ext_oneapi_level_zero>(QueueInterop));
  assert(ZeEvent == get_native<backend::ext_oneapi_level_zero>(EventInterop));

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
#else
  std::cout << "Test skipped due to missing support for Level-Zero backend."
            << std::endl;
#endif
  return 0;
}
