// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} -o %t.out %level_zero_options
// RUN: %{run} %t.out

// Test for Level Zero interop API where SYCL RT doesn't take ownership

#include <iostream>
#include <sycl/detail/core.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
// clang-format on

using namespace sycl;

int main() {

  // Creat SYCL platform/device
  device Device(gpu_selector_v);
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
    auto Context = make_context<backend::ext_oneapi_level_zero>(
        backend_input_t<backend::ext_oneapi_level_zero, context>{
            ZeContext, Devices, ext::oneapi::level_zero::ownership::keep});

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
      event Event = make_event<backend::ext_oneapi_level_zero>(
          backend_input_t<backend::ext_oneapi_level_zero, event>{
              ZeEvent, ext::oneapi::level_zero::ownership::keep},
          Context);

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
