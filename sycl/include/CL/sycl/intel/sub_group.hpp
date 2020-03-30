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

#include <numeric> // std::bit_cast
#include <type_traits>

#ifdef __SYCL_DEVICE_ONLY__

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <typename T, access::address_space Space> class multi_ptr;

namespace detail {

namespace sub_group {

#define __SYCL_SG_GENERATE_BODY_1ARG(name, SPIRVOperation)                     \
  template <typename T> T name(T x, id<1> local_id) {                          \
    using OCLT = detail::ConvertToOpenCLType_t<T>;                             \
    return __spirv_##SPIRVOperation(OCLT(x), local_id.get(0));                 \
  }

__SYCL_SG_GENERATE_BODY_1ARG(shuffle, SubgroupShuffleINTEL)
__SYCL_SG_GENERATE_BODY_1ARG(shuffle_xor, SubgroupShuffleXorINTEL)

#undef __SYCL_SG_GENERATE_BODY_1ARG

#define __SYCL_SG_GENERATE_BODY_2ARG(name, SPIRVOperation)                     \
  template <typename T> T name(T A, T B, uint32_t Delta) {                     \
    using OCLT = detail::ConvertToOpenCLType_t<T>;                             \
    return __spirv_##SPIRVOperation(OCLT(A), OCLT(B), Delta);                  \
  }

__SYCL_SG_GENERATE_BODY_2ARG(shuffle_down, SubgroupShuffleDownINTEL)
__SYCL_SG_GENERATE_BODY_2ARG(shuffle_up, SubgroupShuffleUpINTEL)

#undef __SYCL_SG_GENERATE_BODY_2ARG

// Selects 8-bit, 16-bit or 32-bit type depending on size of T. If T doesn't
// maps to mentioned types, then void is returned
template <typename T>
using SelectBlockT =
    select_apply_cl_scalar_t<T, uint8_t, uint16_t, uint32_t, void>;

template <typename T, access::address_space Space>
using AcceptableForLoadStore =
    bool_constant<!std::is_same<void, SelectBlockT<T>>::value &&
                  Space == access::address_space::global_space>;

// TODO: move this to public cl::sycl::bit_cast as extension?
template <typename To, typename From> To bit_cast(const From &from) {
#if __cpp_lib_bit_cast
  return std::bit_cast<To>(from);
#else

#if __has_builtin(__builtin_bit_cast)
  return __builtin_bit_cast(To, from);
#else
  To to;
  detail::memcpy(&to, &from, sizeof(To));
  return to;
#endif // __has_builtin(__builtin_bit_cast)
#endif // __cpp_lib_bit_cast
}

template <typename T, access::address_space Space>
T load(const multi_ptr<T, Space> src) {
  using BlockT = SelectBlockT<T>;
  using PtrT = detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  BlockT Ret =
      __spirv_SubgroupBlockReadINTEL<BlockT>(reinterpret_cast<PtrT>(src.get()));

  return bit_cast<T>(Ret);
}

template <int N, typename T, access::address_space Space>
vec<T, N> load(const multi_ptr<T, Space> src) {
  using BlockT = SelectBlockT<T>;
  using VecT = detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  using PtrT = detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  VecT Ret =
      __spirv_SubgroupBlockReadINTEL<VecT>(reinterpret_cast<PtrT>(src.get()));

  return bit_cast<typename vec<T, N>::vector_t>(Ret);
}

template <typename T, access::address_space Space>
void store(multi_ptr<T, Space> dst, const T &x) {
  using BlockT = SelectBlockT<T>;
  using PtrT = detail::ConvertToOpenCLType_t<multi_ptr<BlockT, Space>>;

  __spirv_SubgroupBlockWriteINTEL(reinterpret_cast<PtrT>(dst.get()),
                                  bit_cast<BlockT>(x));
}

template <int N, typename T, access::address_space Space>
void store(multi_ptr<T, Space> dst, const vec<T, N> &x) {
  using BlockT = SelectBlockT<T>;
  using VecT = detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  using PtrT = detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  __spirv_SubgroupBlockWriteINTEL(reinterpret_cast<PtrT>(dst.get()),
                                  bit_cast<VecT>(x));
}

} // namespace sub_group

} // namespace detail

namespace intel {

struct sub_group {

  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = size_t;
  static constexpr int dimensions = 1;

  /* --- common interface members --- */

  id<1> get_local_id() const {
    return __spirv_BuiltInSubgroupLocalInvocationId;
  }
  range<1> get_local_range() const { return __spirv_BuiltInSubgroupSize; }

  range<1> get_max_local_range() const {
    return __spirv_BuiltInSubgroupMaxSize;
  }

  id<1> get_group_id() const { return __spirv_BuiltInSubgroupId; }

  unsigned int get_group_range() const { return __spirv_BuiltInNumSubgroups; }

  unsigned int get_uniform_group_range() const {
    return __spirv_BuiltInNumEnqueuedSubgroups;
  }

  /* --- vote / ballot functions --- */

  __SYCL_DEPRECATED__("Use sycl::intel::any_of instead.")
  bool any(bool predicate) const {
    return __spirv_GroupAny(__spv::Scope::Subgroup, predicate);
  }

  __SYCL_DEPRECATED__("Use sycl::intel::all_of instead.")
  bool all(bool predicate) const {
    return __spirv_GroupAll(__spv::Scope::Subgroup, predicate);
  }

  template <typename T>
  using EnableIfIsScalarArithmetic =
      detail::enable_if_t<detail::is_scalar_arithmetic<T>::value, T>;

  /* --- collectives --- */

  template <typename T>
  __SYCL_DEPRECATED__("Use sycl::intel::broadcast instead.")
  EnableIfIsScalarArithmetic<T> broadcast(T x, id<1> local_id) const {
    return detail::spirv::GroupBroadcast<__spv::Scope::Subgroup>(x, local_id);
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED__("Use sycl::intel::reduce instead.")
  EnableIfIsScalarArithmetic<T> reduce(T x, BinaryOperation op) const {
    return detail::calc<T, __spv::GroupOperation::Reduce,
                        __spv::Scope::Subgroup>(
        typename detail::GroupOpTag<T>::type(), x, op);
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED__("Use sycl::intel::reduce instead.")
  EnableIfIsScalarArithmetic<T> reduce(T x, T init, BinaryOperation op) const {
    return op(init, reduce(x, op));
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED__("Use sycl::intel::exclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> exclusive_scan(T x, BinaryOperation op) const {
    return detail::calc<T, __spv::GroupOperation::ExclusiveScan,
                        __spv::Scope::Subgroup>(
        typename detail::GroupOpTag<T>::type(), x, op);
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED__("Use sycl::intel::exclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> exclusive_scan(T x, T init,
                                               BinaryOperation op) const {
    if (get_local_id().get(0) == 0) {
      x = op(init, x);
    }
    T scan = exclusive_scan(x, op);
    if (get_local_id().get(0) == 0) {
      scan = init;
    }
    return scan;
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED__("Use sycl::intel::inclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> inclusive_scan(T x, BinaryOperation op) const {
    return detail::calc<T, __spv::GroupOperation::InclusiveScan,
                        __spv::Scope::Subgroup>(
        typename detail::GroupOpTag<T>::type(), x, op);
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED__("Use sycl::intel::inclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> inclusive_scan(T x, BinaryOperation op,
                                               T init) const {
    if (get_local_id().get(0) == 0) {
      x = op(init, x);
    }
    return inclusive_scan(x, op);
  }

  /* --- one-input shuffles --- */
  /* indices in [0 , sub_group size) */

  template <typename T> T shuffle(T x, id<1> local_id) const {
    return detail::sub_group::shuffle(x, local_id);
  }

  template <typename T> T shuffle_down(T x, uint32_t delta) const {
    return detail::sub_group::shuffle_down(x, x, delta);
  }

  template <typename T> T shuffle_up(T x, uint32_t delta) const {
    return detail::sub_group::shuffle_up(x, x, delta);
  }

  template <typename T> T shuffle_xor(T x, id<1> value) const {
    return detail::sub_group::shuffle_xor(x, value);
  }

  /* --- two-input shuffles --- */
  /* indices in [0 , 2 * sub_group size) */

  template <typename T> T shuffle(T x, T y, id<1> local_id) const {
    return detail::sub_group::shuffle_down(x, y,
                                           (local_id - get_local_id()).get(0));
  }

  template <typename T>
  T shuffle_down(T current, T next, uint32_t delta) const {
    return detail::sub_group::shuffle_down(current, next, delta);
  }

  template <typename T>
  T shuffle_up(T previous, T current, uint32_t delta) const {
    return detail::sub_group::shuffle_up(previous, current, delta);
  }

  /* --- sub_group load/stores --- */
  /* these can map to SIMD or block read/write hardware where available */

  template <typename T, access::address_space Space>
  detail::enable_if_t<
      detail::sub_group::AcceptableForLoadStore<T, Space>::value, T>
  load(const multi_ptr<T, Space> src) const {
    return detail::sub_group::load(src);
  }

  template <int N, typename T, access::address_space Space>
  detail::enable_if_t<
      detail::sub_group::AcceptableForLoadStore<T, Space>::value && N != 1,
      vec<T, N>>
  load(const multi_ptr<T, Space> src) const {
    return detail::sub_group::load<N, T>(src);
  }

  template <int N, typename T, access::address_space Space>
  detail::enable_if_t<
      detail::sub_group::AcceptableForLoadStore<T, Space>::value && N == 1,
      vec<T, 1>>
  load(const multi_ptr<T, Space> src) const {
    return detail::sub_group::load(src);
  }

  template <typename T, access::address_space Space>
  detail::enable_if_t<
      detail::sub_group::AcceptableForLoadStore<T, Space>::value>
  store(multi_ptr<T, Space> dst, const T &x) const {
    detail::sub_group::store(dst, x);
  }

  template <int N, typename T, access::address_space Space>
  detail::enable_if_t<
      detail::sub_group::AcceptableForLoadStore<T, Space>::value && N == 1>
  store(multi_ptr<T, Space> dst, const vec<T, 1> &x) const {
    store<T, Space>(dst, x);
  }

  template <int N, typename T, access::address_space Space>
  detail::enable_if_t<
      detail::sub_group::AcceptableForLoadStore<T, Space>::value && N != 1>
  store(multi_ptr<T, Space> dst, const vec<T, N> &x) const {
    detail::sub_group::store(dst, x);
  }

  /* --- synchronization functions --- */
  void barrier(access::fence_space accessSpace =
                   access::fence_space::global_and_local) const {
    uint32_t flags = detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                           flags);
  }

protected:
  template <int dimensions> friend class cl::sycl::nd_item;
  sub_group() = default;
};
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#else
#include <CL/sycl/intel/sub_group_host.hpp>
#endif
