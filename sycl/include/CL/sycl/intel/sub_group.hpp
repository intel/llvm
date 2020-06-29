//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/spirv.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/intel/functional.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/types.hpp>

#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <typename T, access::address_space Space> class multi_ptr;

namespace detail {

namespace sub_group {

// Selects 8, 16, 32, or 64-bit type depending on size of scalar type T.
template <typename T>
using SelectBlockT = select_cl_scalar_integral_unsigned_t<T>;

template <typename T, access::address_space Space>
using AcceptableForGlobalLoadStore =
    bool_constant<!std::is_same<void, SelectBlockT<T>>::value &&
                  Space == access::address_space::global_space>;

template <typename T, access::address_space Space>
using AcceptableForLocalLoadStore =
    bool_constant<!std::is_same<void, SelectBlockT<T>>::value &&
                  Space == access::address_space::local_space>;

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_SG_GENERATE_BODY_1ARG(name, SPIRVOperation)                     \
  template <typename T> T name(T x, id<1> local_id) {                          \
    using OCLT = sycl::detail::ConvertToOpenCLType_t<T>;                       \
    return __spirv_##SPIRVOperation(OCLT(x), local_id.get(0));                 \
  }

__SYCL_SG_GENERATE_BODY_1ARG(shuffle, SubgroupShuffleINTEL)
__SYCL_SG_GENERATE_BODY_1ARG(shuffle_xor, SubgroupShuffleXorINTEL)

#undef __SYCL_SG_GENERATE_BODY_1ARG

#define __SYCL_SG_GENERATE_BODY_2ARG(name, SPIRVOperation)                     \
  template <typename T> T name(T A, T B, uint32_t Delta) {                     \
    using OCLT = sycl::detail::ConvertToOpenCLType_t<T>;                       \
    return __spirv_##SPIRVOperation(OCLT(A), OCLT(B), Delta);                  \
  }

__SYCL_SG_GENERATE_BODY_2ARG(shuffle_down, SubgroupShuffleDownINTEL)
__SYCL_SG_GENERATE_BODY_2ARG(shuffle_up, SubgroupShuffleUpINTEL)

#undef __SYCL_SG_GENERATE_BODY_2ARG

template <typename T, access::address_space Space>
T load(const multi_ptr<T, Space> src) {
  using BlockT = SelectBlockT<T>;
  using PtrT =
      sycl::detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  BlockT Ret =
      __spirv_SubgroupBlockReadINTEL<BlockT>(reinterpret_cast<PtrT>(src.get()));

  return sycl::detail::bit_cast<T>(Ret);
}

template <int N, typename T, access::address_space Space>
vec<T, N> load(const multi_ptr<T, Space> src) {
  using BlockT = SelectBlockT<T>;
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  using PtrT =
      sycl::detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  VecT Ret =
      __spirv_SubgroupBlockReadINTEL<VecT>(reinterpret_cast<PtrT>(src.get()));

  return sycl::detail::bit_cast<typename vec<T, N>::vector_t>(Ret);
}

template <typename T, access::address_space Space>
void store(multi_ptr<T, Space> dst, const T &x) {
  using BlockT = SelectBlockT<T>;
  using PtrT = sycl::detail::ConvertToOpenCLType_t<multi_ptr<BlockT, Space>>;

  __spirv_SubgroupBlockWriteINTEL(reinterpret_cast<PtrT>(dst.get()),
                                  sycl::detail::bit_cast<BlockT>(x));
}

template <int N, typename T, access::address_space Space>
void store(multi_ptr<T, Space> dst, const vec<T, N> &x) {
  using BlockT = SelectBlockT<T>;
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  using PtrT =
      sycl::detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  __spirv_SubgroupBlockWriteINTEL(reinterpret_cast<PtrT>(dst.get()),
                                  sycl::detail::bit_cast<VecT>(x));
}
#endif // __SYCL_DEVICE_ONLY__

} // namespace sub_group

} // namespace detail

namespace intel {

struct sub_group {

  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = uint32_t;
  static constexpr int dimensions = 1;

  /* --- common interface members --- */

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupLocalInvocationId;
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupSize;
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  range_type get_max_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupMaxSize;
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupId;
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInNumSubgroups;
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T>
  using EnableIfIsScalarArithmetic =
      sycl::detail::enable_if_t<sycl::detail::is_scalar_arithmetic<T>::value,
                                T>;

  /* --- one-input shuffles --- */
  /* indices in [0 , sub_group size) */

  template <typename T> T shuffle(T x, id_type local_id) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::shuffle(x, local_id);
#else
    (void)x;
    (void)local_id;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T> T shuffle_down(T x, uint32_t delta) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::shuffle_down(x, x, delta);
#else
    (void)x;
    (void)delta;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T> T shuffle_up(T x, uint32_t delta) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::shuffle_up(x, x, delta);
#else
    (void)x;
    (void)delta;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T> T shuffle_xor(T x, id_type value) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::shuffle_xor(x, value);
#else
    (void)x;
    (void)value;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  /* --- two-input shuffles --- */
  /* indices in [0 , 2 * sub_group size) */

  template <typename T>
  __SYCL_DEPRECATED("Two-input sub-group shuffles are deprecated.")
  T shuffle(T x, T y, id_type local_id) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::shuffle_down(
        x, y, (local_id - get_local_id()).get(0));
#else
    (void)x;
    (void)y;
    (void)local_id;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T>
  __SYCL_DEPRECATED("Two-input sub-group shuffles are deprecated.")
  T shuffle_down(T current, T next, uint32_t delta) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::shuffle_down(current, next, delta);
#else
    (void)current;
    (void)next;
    (void)delta;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T>
  __SYCL_DEPRECATED("Two-input sub-group shuffles are deprecated.")
  T shuffle_up(T previous, T current, uint32_t delta) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::shuffle_up(previous, current, delta);
#else
    (void)previous;
    (void)current;
    (void)delta;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  /* --- sub_group load/stores --- */
  /* these can map to SIMD or block read/write hardware where available */

  template <typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value, T>
  load(const multi_ptr<T, Space> src) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::load(src);
#else
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value, T>
  load(const multi_ptr<T, Space> src) const {
#ifdef __SYCL_DEVICE_ONLY__
    return src.get()[get_local_id()[0]];
#else
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N != 1,
      vec<T, N>>
  load(const multi_ptr<T, Space> src) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::load<N, T>(src);
#else
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<T, Space> src) const {
#ifdef __SYCL_DEVICE_ONLY__
    vec<T, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = *(src.get() + i * get_max_local_range()[0] + get_local_id()[0]);
    }
    return res;
#else
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 1,
      vec<T, 1>>
  load(const multi_ptr<T, Space> src) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::sub_group::load(src);
#else
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space> dst, const T &x) const {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::sub_group::store(dst, x);
#else
    (void)dst;
    (void)x;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space> dst, const T &x) const {
#ifdef __SYCL_DEVICE_ONLY__
    dst.get()[get_local_id()[0]] = x;
#else
    (void)dst;
    (void)x;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 1>
  store(multi_ptr<T, Space> dst, const vec<T, 1> &x) const {
#ifdef __SYCL_DEVICE_ONLY__
    store<T, Space>(dst, x);
#else
    (void)dst;
    (void)x;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N != 1>
  store(multi_ptr<T, Space> dst, const vec<T, N> &x) const {
#ifdef __SYCL_DEVICE_ONLY__
    sycl::detail::sub_group::store(dst, x);
#else
    (void)dst;
    (void)x;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space> dst, const vec<T, N> &x) const {
#ifdef __SYCL_DEVICE_ONLY__
    for (int i = 0; i < N; ++i) {
      *(dst.get() + i * get_max_local_range()[0] + get_local_id()[0]) = x[i];
    }
#else
    (void)dst;
    (void)x;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  /* --- synchronization functions --- */
  void barrier() const {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_ControlBarrier(
        __spv::Scope::Subgroup, __spv::Scope::Subgroup,
        __spv::MemorySemanticsMask::AcquireRelease |
            __spv::MemorySemanticsMask::SubgroupMemory |
            __spv::MemorySemanticsMask::WorkgroupMemory |
            __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  __SYCL_DEPRECATED("Sub-group barrier accepting fence_space is deprecated."
                    "Use barrier() without a fence_space instead.")
  void barrier(access::fence_space accessSpace) const {
#ifdef __SYCL_DEVICE_ONLY__
    int32_t flags = sycl::detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                           flags);
#else
    (void)accessSpace;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

protected:
  template <int dimensions> friend class cl::sycl::nd_item;
  sub_group() = default;
};
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
