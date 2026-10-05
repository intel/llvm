// REQUIRES: arch-intel_gpu_cri
// RUN: %{build} -o %t.out %level_zero_options
// RUN: %{run} %t.out

#include <level_zero/ze_api.h>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

using namespace sycl;

int main() {
  ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  assert(result == ZE_RESULT_SUCCESS && "zeInit failed");

  queue Queue;
  device Dev = Queue.get_device();
  auto Driver = get_native<backend::ext_oneapi_level_zero>(Dev.get_platform());
  uint32_t ExtensionCount = 0;
  result = zeDriverGetExtensionProperties(Driver, &ExtensionCount, nullptr);
  assert(result == ZE_RESULT_SUCCESS &&
         "zeDriverGetExtensionProperties failed");
  std::vector<ze_driver_extension_properties_t> Extensions(ExtensionCount);
  result = zeDriverGetExtensionProperties(Driver, &ExtensionCount,
                                          Extensions.data());
  assert(result == ZE_RESULT_SUCCESS &&
         "zeDriverGetExtensionProperties failed");
  Extensions.resize(ExtensionCount);

  bool Result;
  if (std::any_of(
          Extensions.begin(), Extensions.end(), [](const auto &Extension) {
            return std::strcmp(Extension.name, ZE_FLOAT_ATOMICS_EXT_NAME) == 0;
          }))
    Result = true;
  else
    Result = false;
  assert(Dev.has(aspect::ext_oneapi_atomic16) == Result &&
         "The Result value differs from the implemented atomic16 check on "
         "the L0 backend.");
  return 0;
}
