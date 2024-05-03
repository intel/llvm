#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;
using namespace sycl::ext::codeplay;

template <typename DataT> class IotaKernel;

template <typename DataT, typename T>
event iota(queue q, buffer<DataT> buff, T value) {
  return q.submit([&](handler &cgh) {
    accessor out(buff, cgh, write_only, no_init);
    auto offset = static_cast<DataT>(value);
    cgh.parallel_for<IotaKernel<DataT>>(buff.get_range(), [=](id<1> i) {
      out[i] = static_cast<DataT>(i) + offset;
    });
  });
}

inline void complete_fusion_with_check(experimental::fusion_wrapper fw,
                                       const property_list &properties = {}) {
  assert(fw.is_in_fusion_mode() && "Queue not in fusion mode");
  fw.complete_fusion(properties);
  assert(!fw.is_in_fusion_mode() && "Queue in fusion mode");
}
