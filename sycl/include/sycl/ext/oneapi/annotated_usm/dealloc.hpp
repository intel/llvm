#pragma once

#include <sycl/ext/oneapi/annotated_arg/annotated_ptr.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

////
//  Deallocation
////
template <typename T, typename propList>
void free(annotated_ptr<T, propList> &ptr, const context &syclContext) {
  sycl::free(ptr.get(), syclContext);
}

template <typename T, typename propList>
void free(annotated_ptr<T, propList> &ptr, const queue &syclQueue) {
  sycl::free(ptr.get(), syclQueue);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl