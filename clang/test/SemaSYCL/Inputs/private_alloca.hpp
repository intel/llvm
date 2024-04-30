#pragma once

#include "./sycl.hpp"

#include <stddef.h>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

template <typename ElementType, auto &Size, access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
multi_ptr<ElementType, access::address_space::private_space,
          DecorateAddress> private_alloca(kernel_handler &h);

template <typename ElementType, size_t Alignment, auto &Size,
          access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
multi_ptr<ElementType, access::address_space::private_space,
          DecorateAddress> aligned_private_alloca(kernel_handler &h);
} // namespace experimental
} // namesapce oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
