//==--- detail/sub_group_load_store.hpp - SYCL sub_group load/store impl --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops.hpp>
#include <sycl/bit_cast.hpp>
#include <sycl/detail/sub_group_core.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

namespace sub_group {

template <int Size>
using subgroup_fixed_width_unsigned = std::conditional_t<
    Size == 1, uint8_t,
    std::conditional_t<Size == 2, uint16_t,
                       std::conditional_t<Size == 4, uint32_t, uint64_t>>>;

template <typename T>
using SelectBlockT = subgroup_fixed_width_unsigned<sizeof(T)>;

template <typename T, access::address_space Space>
using AcceptableForGlobalLoadStore =
    std::bool_constant<!std::is_same_v<void, SelectBlockT<T>> &&
                       Space == access::address_space::global_space>;

template <typename T, access::address_space Space>
using AcceptableForLocalLoadStore =
    std::bool_constant<!std::is_same_v<void, SelectBlockT<T>> &&
                       Space == access::address_space::local_space>;

template <typename T, access::address_space Space>
using AcceptableForLoadStore =
    std::bool_constant<AcceptableForGlobalLoadStore<T, Space>::value ||
                       AcceptableForLocalLoadStore<T, Space>::value>;

#ifdef __SYCL_DEVICE_ONLY__
inline constexpr bool
subgroup_dynamic_address_cast_is_possible(access::address_space Src,
                                          access::address_space Dst) {
  constexpr auto constant_space = access::address_space::constant_space;
  if (Src == constant_space || Dst == constant_space)
    return Src == Dst;

  constexpr auto generic_space = access::address_space::generic_space;
  return Src == Dst || Src == generic_space || Dst == generic_space;
}

template <access::address_space Space, typename ElementType>
auto subgroup_dynamic_address_cast(ElementType *Ptr) {
  constexpr auto global_space = access::address_space::global_space;
  constexpr auto local_space = access::address_space::local_space;
  constexpr auto SrcAS = deduce_AS<ElementType *>::value;
  using ResultTy = typename DecoratedType<
      std::remove_pointer_t<remove_decoration_t<ElementType *>>, Space>::type *;
  using RemoveCvT = std::remove_cv_t<ElementType>;

  if constexpr (!subgroup_dynamic_address_cast_is_possible(SrcAS, Space)) {
    return (ResultTy) nullptr;
  } else if constexpr (SrcAS == Space ||
                       Space == access::address_space::generic_space) {
    return (ResultTy)Ptr;
  } else if constexpr (Space == global_space) {
    return (ResultTy)__spirv_GenericCastToPtrExplicit_ToGlobal(
        const_cast<RemoveCvT *>(Ptr), __spv::StorageClass::CrossWorkgroup);
  } else if constexpr (Space == local_space) {
    return (ResultTy)__spirv_GenericCastToPtrExplicit_ToLocal(
        const_cast<RemoveCvT *>(Ptr), __spv::StorageClass::Workgroup);
  } else {
    return (ResultTy)Ptr;
  }
}
#else
template <access::address_space Space, typename ElementType>
auto subgroup_dynamic_address_cast(ElementType *Ptr) {
  return Ptr;
}
#endif

template <typename ElementType, int NumElements>
struct subgroup_block_vec_type {
#ifdef __SYCL_DEVICE_ONLY__
  using type =
      std::conditional_t<NumElements == 1, ElementType,
                         ElementType
                         __attribute__((ext_vector_type(NumElements)))>;
#else
  using type = vec<ElementType, NumElements>;
#endif
};

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
auto convertToBlockPtr(multi_ptr<ElementType, Space, DecorateAddress> Ptr) {
  using TargetElemTyNoCV = SelectBlockT<std::remove_const_t<ElementType>>;
  using TargetElemTy =
      std::conditional_t<std::is_const_v<ElementType>, const TargetElemTyNoCV,
                         TargetElemTyNoCV>;
  using ResultTy = typename DecoratedType<TargetElemTy, Space>::type *;
  return reinterpret_cast<ResultTy>(Ptr.get_decorated());
}

#ifdef __SYCL_DEVICE_ONLY__
template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
T load(const multi_ptr<T, Space, DecorateAddress> src) {
  using BlockT = SelectBlockT<T>;
  BlockT Ret = __spirv_SubgroupBlockReadINTEL<BlockT>(convertToBlockPtr(src));
  return sycl::bit_cast<T>(Ret);
}

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
vec<T, N> load(const multi_ptr<T, Space, DecorateAddress> src) {
  using BlockT = SelectBlockT<T>;
  using VecT = typename subgroup_block_vec_type<BlockT, N>::type;
  VecT Ret = __spirv_SubgroupBlockReadINTEL<VecT>(convertToBlockPtr(src));
  return sycl::bit_cast<vec<T, N>>(Ret);
}

template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
void store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) {
  using BlockT = SelectBlockT<T>;
  __spirv_SubgroupBlockWriteINTEL(convertToBlockPtr(dst),
                                  sycl::bit_cast<BlockT>(x));
}

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
void store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) {
  using BlockT = SelectBlockT<T>;
  using VecT = typename subgroup_block_vec_type<BlockT, N>::type;
  __spirv_SubgroupBlockWriteINTEL(convertToBlockPtr(dst),
                                  sycl::bit_cast<VecT>(x));
}
#endif // __SYCL_DEVICE_ONLY__

} // namespace sub_group

template <typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
inline multi_ptr<T, Space, IsDecorated>
GetUnqualMultiPtr(const multi_ptr<CVT, Space, IsDecorated> &Mptr) {
  if constexpr (IsDecorated == access::decorated::legacy) {
    return multi_ptr<T, Space, IsDecorated>{
        const_cast<typename multi_ptr<T, Space, IsDecorated>::pointer_t>(
            Mptr.get())};
  } else {
    return multi_ptr<T, Space, IsDecorated>{
        const_cast<typename multi_ptr<T, Space, IsDecorated>::pointer>(
            Mptr.get_decorated())};
  }
}

} // namespace detail

// Out-of-line definitions for the deprecated sub_group load/store members
// declared in detail/sub_group_core.hpp.

template <typename CVT, typename T> T sub_group::load(CVT *cv_src) const {
  T *src = const_cast<T *>(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (!std::is_same_v<remove_decoration_t<T>, T>) {
    return load(sycl::multi_ptr<remove_decoration_t<T>,
                                sycl::detail::deduce_AS<T>::value,
                                sycl::access::decorated::yes>(src));
  }

#if defined(__NVPTX__) || defined(__AMDGCN__)
  return src[get_local_id()[0]];
#else
  if (auto l = detail::sub_group::subgroup_dynamic_address_cast<
          access::address_space::local_space>(src))
    return load(multi_ptr<T, access::address_space::local_space,
                          access::decorated::yes>(l));

  if (auto g = detail::sub_group::subgroup_dynamic_address_cast<
          access::address_space::global_space>(src))
    return load(multi_ptr<T, access::address_space::global_space,
                          access::decorated::yes>(g));

  return {};
#endif
#else
  (void)src;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

template <typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
T sub_group::load(multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  static_assert(
      sycl::detail::sub_group::AcceptableForLoadStore<T, Space>::value,
      "Sub-group block load requires global or local address space.");
  multi_ptr<T, Space, IsDecorated> src =
      sycl::detail::GetUnqualMultiPtr(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (Space == access::address_space::local_space) {
    return src.get()[get_local_id()[0]];
  } else {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    return src.get()[get_local_id()[0]];
#else
    return sycl::detail::sub_group::load(src);
#endif
  }
#else
  (void)src;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

template <int N, typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
vec<T, N> sub_group::load(multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  static_assert(
      sycl::detail::sub_group::AcceptableForLoadStore<T, Space>::value,
      "Sub-group block load requires global or local address space.");
  multi_ptr<T, Space, IsDecorated> src =
      sycl::detail::GetUnqualMultiPtr(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (Space == access::address_space::local_space) {
    vec<T, N> res;
    for (int i = 0; i < N; ++i)
      res[i] = *(src.get() + i * get_max_local_range()[0] + get_local_id()[0]);
    return res;
  } else {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    vec<T, N> res;
    for (int i = 0; i < N; ++i)
      res[i] = *(src.get() + i * get_max_local_range()[0] + get_local_id()[0]);
    return res;
#else
    if constexpr (N == 16) {
      return {sycl::detail::sub_group::load<8, T>(src),
              sycl::detail::sub_group::load<8, T>(
                  src + 8 * get_max_local_range()[0])};
    } else if constexpr (N == 3) {
      return {
          sycl::detail::sub_group::load<1, T>(src),
          sycl::detail::sub_group::load<2, T>(src + get_max_local_range()[0])};
    } else if constexpr (N == 1) {
      return sycl::detail::sub_group::load(src);
    } else {
      return sycl::detail::sub_group::load<N, T>(src);
    }
#endif
  }
#else
  (void)src;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

template <typename T>
void sub_group::store(T *dst, const remove_decoration_t<T> &x) const {
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (!std::is_same_v<remove_decoration_t<T>, T>) {
    store(sycl::multi_ptr<remove_decoration_t<T>,
                          sycl::detail::deduce_AS<T>::value,
                          sycl::access::decorated::yes>(dst),
          x);
    return;
  }

#if defined(__NVPTX__) || defined(__AMDGCN__)
  dst[get_local_id()[0]] = x;
#else
  if (auto l = detail::sub_group::subgroup_dynamic_address_cast<
          access::address_space::local_space>(dst)) {
    store(multi_ptr<T, access::address_space::local_space,
                    access::decorated::yes>(l),
          x);
    return;
  }
  if (auto g = detail::sub_group::subgroup_dynamic_address_cast<
          access::address_space::global_space>(dst)) {
    store(multi_ptr<T, access::address_space::global_space,
                    access::decorated::yes>(g),
          x);
    return;
  }
  return;
#endif
#else
  (void)dst;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
void sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                      const T &x) const {
  static_assert(
      sycl::detail::sub_group::AcceptableForLoadStore<T, Space>::value,
      "Sub-group block store requires global or local address space.");
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (Space == access::address_space::local_space) {
    dst.get()[get_local_id()[0]] = x;
  } else {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    dst.get()[get_local_id()[0]] = x;
#else
    sycl::detail::sub_group::store(dst, x);
#endif
  }
#else
  (void)dst;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
void sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                      const vec<T, N> &x) const {
  static_assert(
      sycl::detail::sub_group::AcceptableForLoadStore<T, Space>::value,
      "Sub-group block store requires global or local address space.");
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (Space == access::address_space::local_space) {
    for (int i = 0; i < N; ++i)
      *(dst.get() + i * get_max_local_range()[0] + get_local_id()[0]) = x[i];
  } else {
#if defined(__NVPTX__) || defined(__AMDGCN__)
    for (int i = 0; i < N; ++i)
      *(dst.get() + i * get_max_local_range()[0] + get_local_id()[0]) = x[i];
#else
    if constexpr (N == 16) {
      store<8, T, Space, DecorateAddress>(dst, x.lo());
      store<8, T, Space, DecorateAddress>(dst + 8 * get_max_local_range()[0],
                                          x.hi());
    } else if constexpr (N == 3) {
      store<1, T, Space, DecorateAddress>(dst, x.s0());
      store<2, T, Space, DecorateAddress>(dst + get_max_local_range()[0],
                                          {x.s1(), x.s2()});
    } else {
      sycl::detail::sub_group::store(dst, x);
    }
#endif
  }
#else
  (void)dst;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

} // namespace _V1
} // namespace sycl
