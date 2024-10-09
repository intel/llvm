#pragma once

#include <cstddef>

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {
template <typename DataT, int NumElements> class __SYCL_EBO vec;

template <typename DataT, std::size_t N> class marray;

namespace detail {
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
class SwizzleOp;
} // namespace detail

template <int Dimensions> class group;
struct sub_group;

namespace detail {
namespace half_impl {
class half;
}
} // namespace detail
using half = detail::half_impl::half;

namespace access {
enum class address_space;
enum class decorated;
} // namespace access

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr;
} // namespace _V1
} // namespace sycl
