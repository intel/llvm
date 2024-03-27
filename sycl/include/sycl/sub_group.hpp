//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>              // for address_space, decorated
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL_DEPRECATED
#include <sycl/detail/generic_type_traits.hpp> // for select_cl_scalar_inte...
#include <sycl/detail/pi.h>                    // for PI_ERROR_INVALID_DEVICE
#include <sycl/detail/type_traits.hpp>         // for is_scalar_arithmetic
#include <sycl/exception.hpp>                  // for exception, make_error...
#include <sycl/id.hpp>                         // for id
#include <sycl/memory_enums.hpp>               // for memory_scope
#include <sycl/multi_ptr.hpp>                  // for multi_ptr
#include <sycl/range.hpp>                      // for range
#include <sycl/types.hpp>                      // for vec

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/ext/oneapi/functional.hpp>
#endif

#include <stdint.h>    // for uint32_t
#include <tuple>       // for _Swallow_assign, ignore
#include <type_traits> // for enable_if_t, remove_cv_t

namespace sycl {
inline namespace _V1 {
template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr;

namespace detail {

namespace sub_group {

// Selects 8, 16, 32, or 64-bit type depending on size of scalar type T.
template <typename T>
using SelectBlockT = select_cl_scalar_integral_unsigned_t<T>;

template <typename MultiPtrTy> auto convertToBlockPtr(MultiPtrTy MultiPtr) {
  static_assert(is_multi_ptr_v<MultiPtrTy>);
  auto DecoratedPtr = convertToOpenCLType(MultiPtr);
  using DecoratedPtrTy = decltype(DecoratedPtr);
  using ElemTy = remove_decoration_t<std::remove_pointer_t<DecoratedPtrTy>>;

  using TargetElemTy = SelectBlockT<ElemTy>;
  // TODO: Handle cv qualifiers.
#ifdef __SYCL_DEVICE_ONLY__
  using ResultTy =
      typename DecoratedType<TargetElemTy,
                             deduce_AS<DecoratedPtrTy>::value>::type *;
#else
  using ResultTy = TargetElemTy *;
#endif
  return reinterpret_cast<ResultTy>(DecoratedPtr);
}

template <typename T, access::address_space Space>
using AcceptableForGlobalLoadStore =
    std::bool_constant<!std::is_same_v<void, SelectBlockT<T>> &&
                       Space == access::address_space::global_space>;

template <typename T, access::address_space Space>
using AcceptableForLocalLoadStore =
    std::bool_constant<!std::is_same_v<void, SelectBlockT<T>> &&
                       Space == access::address_space::local_space>;

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
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  VecT Ret = __spirv_SubgroupBlockReadINTEL<VecT>(convertToBlockPtr(src));

  return sycl::bit_cast<typename vec<T, N>::vector_t>(Ret);
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
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;

  __spirv_SubgroupBlockWriteINTEL(convertToBlockPtr(dst),
                                  sycl::bit_cast<VecT>(x));
}
#endif // __SYCL_DEVICE_ONLY__

} // namespace sub_group

// Helper for removing const and volatile qualifiers from the element type of
// a multi_ptr.
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

struct sub_group;
namespace ext::oneapi {
inline sycl::sub_group this_sub_group();
namespace experimental {
inline sycl::sub_group this_sub_group();
} // namespace experimental
} // namespace ext::oneapi

struct sub_group {

  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = uint32_t;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope =
      sycl::memory_scope::sub_group;

  /* --- common interface members --- */

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupLocalInvocationId();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupSize();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_max_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupMaxSize();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupId();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_NumSubgroups();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T>
  using EnableIfIsScalarArithmetic =
      std::enable_if_t<sycl::detail::is_scalar_arithmetic<T>::value, T>;

  /* --- one-input shuffles --- */
  /* indices in [0 , sub_group size) */
  template <typename T>
  __SYCL_DEPRECATED("Shuffles in the sub-group class are deprecated.")
  T shuffle(T x, id_type local_id) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::spirv::Shuffle(*this, x, local_id);
#else
    (void)x;
    (void)local_id;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T>
  __SYCL_DEPRECATED("Shuffles in the sub-group class are deprecated.")
  T shuffle_down(T x, uint32_t delta) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::spirv::ShuffleDown(*this, x, delta);
#else
    (void)x;
    (void)delta;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T>
  __SYCL_DEPRECATED("Shuffles in the sub-group class are deprecated.")
  T shuffle_up(T x, uint32_t delta) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::spirv::ShuffleUp(*this, x, delta);
#else
    (void)x;
    (void)delta;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T>
  __SYCL_DEPRECATED("Shuffles in the sub-group class are deprecated.")
  T shuffle_xor(T x, id_type value) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::spirv::ShuffleXor(*this, x, value);
#else
    (void)x;
    (void)value;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  /* --- sub_group load/stores --- */
  /* these can map to SIMD or block read/write hardware where available */
#ifdef __SYCL_DEVICE_ONLY__
  // Method for decorated pointer
  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<!std::is_same<remove_decoration_t<T>, T>::value, T>
  load(CVT *cv_src) const {
    T *src = const_cast<T *>(cv_src);
    return load(sycl::multi_ptr<remove_decoration_t<T>,
                                sycl::detail::deduce_AS<T>::value,
                                sycl::access::decorated::yes>(src));
  }

  // Method for raw pointer
  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<std::is_same<remove_decoration_t<T>, T>::value, T>
  load(CVT *cv_src) const {
    T *src = const_cast<T *>(cv_src);

#if defined(__NVPTX__) || defined(__AMDGCN__)
    return src[get_local_id()[0]];
#else  // __NVPTX__ || __AMDGCN__
    auto l = __SYCL_GenericCastToPtrExplicit_ToLocal<T>(src);
    if (l)
      return load(l);

    auto g = __SYCL_GenericCastToPtrExplicit_ToGlobal<T>(src);
    if (g)
      return load(g);

    assert(!"Sub-group load() is supported for local or global pointers only.");
    return {};
#endif // __NVPTX__ || __AMDGCN__
  }
#else  //__SYCL_DEVICE_ONLY__
  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  T load(CVT *src) const {
    (void)src;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
  }
#endif //__SYCL_DEVICE_ONLY__

  template <typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value, T>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
    multi_ptr<T, Space, IsDecorated> src =
        sycl::detail::GetUnqualMultiPtr(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
    return src.get()[get_local_id()[0]];
#else
    return sycl::detail::sub_group::load(src);
#endif // __NVPTX__ || __AMDGCN__
#else
    (void)src;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
  }

  template <typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value, T>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
    multi_ptr<T, Space, IsDecorated> src =
        sycl::detail::GetUnqualMultiPtr(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
    return src.get()[get_local_id()[0]];
#else
    (void)src;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
    multi_ptr<T, Space, IsDecorated> src =
        sycl::detail::GetUnqualMultiPtr(cv_src);
    vec<T, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = *(src.get() + i * get_max_local_range()[0] + get_local_id()[0]);
    }
    return res;
  }
#else  // __NVPTX__ || __AMDGCN__
  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N != 1 && N != 3 && N != 16,
      vec<T, N>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
    multi_ptr<T, Space, IsDecorated> src =
        sycl::detail::GetUnqualMultiPtr(cv_src);
    return sycl::detail::sub_group::load<N, T>(src);
  }

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 16,
      vec<T, 16>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
    multi_ptr<T, Space, IsDecorated> src =
        sycl::detail::GetUnqualMultiPtr(cv_src);
    return {sycl::detail::sub_group::load<8, T>(src),
            sycl::detail::sub_group::load<8, T>(src +
                                                8 * get_max_local_range()[0])};
  }

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 3,
      vec<T, 3>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
    multi_ptr<T, Space, IsDecorated> src =
        sycl::detail::GetUnqualMultiPtr(cv_src);
    return {
        sycl::detail::sub_group::load<1, T>(src),
        sycl::detail::sub_group::load<2, T>(src + get_max_local_range()[0])};
  }

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 1,
      vec<T, 1>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
    multi_ptr<T, Space, IsDecorated> src =
        sycl::detail::GetUnqualMultiPtr(cv_src);
    return sycl::detail::sub_group::load(src);
  }
#endif // ___NVPTX___
#else  // __SYCL_DEVICE_ONLY__
  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space, IsDecorated> src) const {
    (void)src;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
  }
#endif // __SYCL_DEVICE_ONLY__

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
    multi_ptr<T, Space, IsDecorated> src =
        sycl::detail::GetUnqualMultiPtr(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
    vec<T, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = *(src.get() + i * get_max_local_range()[0] + get_local_id()[0]);
    }
    return res;
#else
    (void)src;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

#ifdef __SYCL_DEVICE_ONLY__
  // Method for decorated pointer
  template <typename T>
  std::enable_if_t<!std::is_same<remove_decoration_t<T>, T>::value>
  store(T *dst, const remove_decoration_t<T> &x) const {
    store(sycl::multi_ptr<remove_decoration_t<T>,
                          sycl::detail::deduce_AS<T>::value,
                          sycl::access::decorated::yes>(dst),
          x);
  }

  // Method for raw pointer
  template <typename T>
  std::enable_if_t<std::is_same<remove_decoration_t<T>, T>::value>
  store(T *dst, const remove_decoration_t<T> &x) const {

#if defined(__NVPTX__) || defined(__AMDGCN__)
    dst[get_local_id()[0]] = x;
#else  // __NVPTX__ || __AMDGCN__
    auto l = __SYCL_GenericCastToPtrExplicit_ToLocal<T>(dst);
    if (l) {
      store(l, x);
      return;
    }

    auto g = __SYCL_GenericCastToPtrExplicit_ToGlobal<T>(dst);
    if (g) {
      store(g, x);
      return;
    }

    assert(
        !"Sub-group store() is supported for local or global pointers only.");
    return;
#endif // __NVPTX__ || __AMDGCN__
  }
#else  //__SYCL_DEVICE_ONLY__
  template <typename T> void store(T *dst, const T &x) const {
    (void)dst;
    (void)x;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
  }
#endif //__SYCL_DEVICE_ONLY__

  template <typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) const {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
    dst.get()[get_local_id()[0]] = x;
#else
    sycl::detail::sub_group::store(dst, x);
#endif // __NVPTX__ || __AMDGCN__
#else
    (void)dst;
    (void)x;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) const {
#ifdef __SYCL_DEVICE_ONLY__
    dst.get()[get_local_id()[0]] = x;
#else
    (void)dst;
    (void)x;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) const {
    for (int i = 0; i < N; ++i) {
      *(dst.get() + i * get_max_local_range()[0] + get_local_id()[0]) = x[i];
    }
  }
#else // __NVPTX__ || __AMDGCN__
  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N != 1 && N != 3 && N != 16>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) const {
    sycl::detail::sub_group::store(dst, x);
  }

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 1>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, 1> &x) const {
    sycl::detail::sub_group::store(dst, x);
  }

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 3>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, 3> &x) const {
    store<1, T, Space, DecorateAddress>(dst, x.s0());
    store<2, T, Space, DecorateAddress>(dst + get_max_local_range()[0],
                                        {x.s1(), x.s2()});
  }

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 16>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, 16> &x) const {
    store<8, T, Space, DecorateAddress>(dst, x.lo());
    store<8, T, Space, DecorateAddress>(dst + 8 * get_max_local_range()[0],
                                        x.hi());
  }

#endif // __NVPTX__ || __AMDGCN__
#else  // __SYCL_DEVICE_ONLY__
  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) const {
    (void)dst;
    (void)x;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
  }
#endif // __SYCL_DEVICE_ONLY__

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) const {
#ifdef __SYCL_DEVICE_ONLY__
    for (int i = 0; i < N; ++i) {
      *(dst.get() + i * get_max_local_range()[0] + get_local_id()[0]) = x[i];
    }
#else
    (void)dst;
    (void)x;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  /* --- synchronization functions --- */
  __SYCL_DEPRECATED("Sub-group barrier with no arguments is deprecated.")
  void barrier() const {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_ControlBarrier(
        __spv::Scope::Subgroup, __spv::Scope::Subgroup,
        __spv::MemorySemanticsMask::AcquireRelease |
            __spv::MemorySemanticsMask::SubgroupMemory |
            __spv::MemorySemanticsMask::WorkgroupMemory |
            __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
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
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES__
  /* --- deprecated collective functions --- */
  template <typename T>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::broadcast instead.")
  EnableIfIsScalarArithmetic<T> broadcast(T x, id<1> local_id) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::spirv::GroupBroadcast<sub_group>(x, local_id);
#else
    (void)x;
    (void)local_id;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::reduce instead.")
  EnableIfIsScalarArithmetic<T> reduce(T x, BinaryOperation op) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::calc<__spv::GroupOperation::Reduce>(
        typename sycl::detail::GroupOpTag<T>::type(), *this, x, op);
#else
    (void)x;
    (void)op;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::reduce instead.")
  EnableIfIsScalarArithmetic<T> reduce(T x, T init, BinaryOperation op) const {
#ifdef __SYCL_DEVICE_ONLY__
    return op(init, reduce(x, op));
#else
    (void)x;
    (void)init;
    (void)op;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::exclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> exclusive_scan(T x, BinaryOperation op) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::calc<__spv::GroupOperation::ExclusiveScan>(
        typename sycl::detail::GroupOpTag<T>::type(), *this, x, op);
#else
    (void)x;
    (void)op;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::exclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> exclusive_scan(T x, T init,
                                               BinaryOperation op) const {
#ifdef __SYCL_DEVICE_ONLY__
    if (get_local_id().get(0) == 0) {
      x = op(init, x);
    }
    T scan = exclusive_scan(x, op);
    if (get_local_id().get(0) == 0) {
      scan = init;
    }
    return scan;
#else
    (void)x;
    (void)init;
    (void)op;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::inclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> inclusive_scan(T x, BinaryOperation op) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::calc<__spv::GroupOperation::InclusiveScan>(
        typename sycl::detail::GroupOpTag<T>::type(), *this, x, op);
#else
    (void)x;
    (void)op;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::inclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> inclusive_scan(T x, BinaryOperation op,
                                               T init) const {
#ifdef __SYCL_DEVICE_ONLY__
    if (get_local_id().get(0) == 0) {
      x = op(init, x);
    }
    return inclusive_scan(x, op);
#else
    (void)x;
    (void)op;
    (void)init;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }
#endif // __INTEL_PREVIEW_BREAKING_CHANGES__

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    return get_local_linear_id() == 0;
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  // Common member functions for by-value semantics
  friend bool operator==(const sub_group &lhs, const sub_group &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    return lhs.get_group_id() == rhs.get_group_id();
#else
    std::ignore = lhs;
    std::ignore = rhs;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  friend bool operator!=(const sub_group &lhs, const sub_group &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    return !(lhs == rhs);
#else
    std::ignore = lhs;
    std::ignore = rhs;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

protected:
  template <int dimensions> friend class sycl::nd_item;
  friend sub_group ext::oneapi::this_sub_group();
  friend sub_group ext::oneapi::experimental::this_sub_group();
  sub_group() = default;
};

namespace ext::oneapi {
__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::experimental::this_sub_group() instead")
inline sycl::sub_group this_sub_group() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::sub_group();
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}
namespace experimental {
inline sycl::sub_group this_sub_group() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::sub_group();
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}
} // namespace experimental
} // namespace ext::oneapi

} // namespace _V1
} // namespace sycl
