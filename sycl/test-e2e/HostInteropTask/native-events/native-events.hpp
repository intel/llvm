#include <sycl/sycl.hpp>

using manual_interop_sync =
    sycl::ext::codeplay::experimental::property::host_task::manual_interop_sync;

constexpr auto PropList = [](bool UseManualInteropSync) -> sycl::property_list {
  if (UseManualInteropSync)
    return {manual_interop_sync{}};
  return {};
};
