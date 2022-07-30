// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %level_zero_options
// RUN: env SYCL_BE=PI_LEVEL_ZERO %GPU_RUN_PLACEHOLDER %t.out

// Test for Level Zero interop API where SYCL RT doesn't take ownership

#include <iostream>
#include <sycl/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using namespace sycl;

int main() {

  // Creat SYCL platform/device
  device Device(gpu_selector{});
  platform Platform = Device.get_info<info::device::platform>();

  // Create native Level-Zero context
  ze_context_handle_t ZeContext;
  ze_context_desc_t ZeContextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr,
                                     0};
  auto ZeDriver = get_native<backend::ext_oneapi_level_zero>(Platform);
  auto ZeDevice = get_native<backend::ext_oneapi_level_zero>(Device);
  zeContextCreate(ZeDriver, &ZeContextDesc, &ZeContext);

  { // Scope in which SYCL interop context object is live
    std::vector<device> Devices{};
    Devices.push_back(Device);
    auto Context = level_zero::make<context>(Devices, ZeContext,
                                             level_zero::ownership::keep);

    // Create L0 event pool
    ze_event_pool_handle_t ZeEventPool;
    ze_event_pool_desc_t ZeEventPoolDesc{};
    ZeEventPoolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    ZeEventPoolDesc.count = 1;
    ZeEventPoolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    zeEventPoolCreate(ZeContext, &ZeEventPoolDesc, 1, &ZeDevice, &ZeEventPool);

    // Create L0 event
    ze_event_handle_t ZeEvent;
    ze_event_desc_t ZeEventDesc{};
    ZeEventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    ZeEventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    ZeEventDesc.wait = 0;
    ZeEventDesc.index = 0;
    zeEventCreate(ZeEventPool, &ZeEventDesc, &ZeEvent);

    { // Scope in which SYCL interop event is alive
      int i = 0;
      event Event = level_zero::make<event>(Context, ZeEvent,
                                            level_zero::ownership::keep);

      info::event_command_status status;
      do {
        status = Event.get_info<info::event::command_execution_status>();
        printf("%d: %s\n", i,
               status == info::event_command_status::complete ? "complete"
                                                              : "!complete");
        if (++i == 5) {
          zeEventHostSignal(ZeEvent);
        }
      } while (status != info::event_command_status::complete);
    }
    zeEventDestroy(ZeEvent);
    zeEventPoolDestroy(ZeEventPool);
  }

  // Verifies that Level-Zero context is not destroyed by SYCL RT yet.
  zeContextDestroy(ZeContext);
  return 0;
}
