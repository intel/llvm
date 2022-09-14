// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %level_zero_options
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>

using namespace sycl;

int main() {
  queue Queue;
  device Dev = Queue.get_device();
  bool Result;
  ze_device_module_properties_t Properties{};
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
