#pragma once

#include <cstddef>

namespace sycl {
inline namespace _V1 {

namespace ext::oneapi {
struct sub_group;
}

namespace ext::oneapi::experimental {
template <typename Group, std::size_t Extent> class group_with_scratchpad;
template <int Dimensions> class root_group;
}

} // namespace _V1
} // namespace sycl
