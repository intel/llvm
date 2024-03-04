#pragma once

#include "./sycl.hpp"

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

#ifdef __SYCL_DEVICE_ONLY__

template <typename ElementType, auto &Size, access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
multi_ptr<ElementType, access::address_space::private_space,
          DecorateAddress> private_alloca(kernel_handler &h);

#else

template <typename ElementType, auto &Size, access::decorated DecorateAddress>
multi_ptr<ElementType, access::address_space::private_space, DecorateAddress>
private_alloca(kernel_handler &h) {
  throw "sycl::ext::oneapi::experimental::private_alloca is not supported in "
        "the host";
}

#endif

} // namespace experimental
} // namesapce oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
