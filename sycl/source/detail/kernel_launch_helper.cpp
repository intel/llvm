#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/ext/intel/experimental/fp_control_kernel_properties.hpp>
#include <sycl/ext/intel/experimental/kernel_execution_properties.hpp>
#include <sycl/ext/oneapi/experimental/virtual_functions.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/kernel_handler.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/cluster_group_prop.hpp>
#include <sycl/ext/oneapi/experimental/use_root_sync_prop.hpp>

#include <detail/device_impl.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

    namespace syclex = sycl::ext::oneapi::experimental;
    void KernelLaunchProperties::verifyDeviceHasProgressGuarantee(
        device_impl &dev,
        syclex::forward_progress_guarantee guarantee,
        syclex::execution_scope threadScope,
        syclex::execution_scope coordinationScope) {
      using execution_scope = syclex::execution_scope;
      using forward_progress =
          syclex::forward_progress_guarantee;
      const bool supported = dev.supportsForwardProgress(
          guarantee, threadScope, coordinationScope);
      if (threadScope == execution_scope::work_group) {
        if (!supported) {
          throw sycl::exception(
              sycl::errc::feature_not_supported,
              "Required progress guarantee for work groups is not "
              "supported by this device.");
        }
        // If we are here, the device supports the guarantee required but there is a
        // caveat in that if the guarantee required is a concurrent guarantee, then
        // we most likely also need to enable cooperative launch of the kernel. That
        // is, although the device supports the required guarantee, some setup work
        // is needed to truly make the device provide that guarantee at runtime.
        // Otherwise, we will get the default guarantee which is weaker than
        // concurrent. Same reasoning applies for sub_group but not for work_item.
        // TODO: Further design work is probably needed to reflect this behavior in
        // Unified Runtime.
        if (guarantee == forward_progress::concurrent)
          MKernelIsCooperative = true;
      } else if (threadScope == execution_scope::sub_group) {
        if (!supported) {
          throw sycl::exception(sycl::errc::feature_not_supported,
                                "Required progress guarantee for sub groups is not "
                                "supported by this device.");
        }
        // Same reasoning as above.
        if (guarantee == forward_progress::concurrent)
          MKernelIsCooperative = true;
      } else { // threadScope is execution_scope::work_item otherwise undefined
              // behavior
        if (!supported) {
          throw sycl::exception(sycl::errc::feature_not_supported,
                                "Required progress guarantee for work items is not "
                                "supported by this device.");
        }
      }
  }

}
}
}