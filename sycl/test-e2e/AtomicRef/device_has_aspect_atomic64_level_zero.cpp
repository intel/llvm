// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} -o %t.out %level_zero_options
// RUN: %{run} %t.out

#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  // Initializing Level Zero driver is required if this test is linked
  // statically with Level Zero loader, otherwise the driver will not be
  // initialized.
  ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  assert(result == ZE_RESULT_SUCCESS && "zeInit failed");

  queue Queue;
  device Dev = Queue.get_device();
  bool Result;
  ze_device_module_properties_t Properties{};
  Properties.stype = ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
  zeDeviceGetModuleProperties(get_native<backend::ext_oneapi_level_zero>(Dev),
                              &Properties);
  if (Properties.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS)
    Result = true;
  else
    Result = false;
  assert(Dev.has(aspect::atomic64) == Result &&
         "The Result value differs from the implemented atomic64 check on "
         "the L0 backend.");
  return 0;
}
