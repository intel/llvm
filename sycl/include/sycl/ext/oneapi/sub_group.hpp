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
#include <CL/sycl/memory_enums.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/types.hpp>
#include <sycl/ext/oneapi/functional.hpp>

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
template <typename T, access::address_space Space>
T load(const multi_ptr<T, Space> src) {
  using BlockT = SelectBlockT<T>;
  using PtrT =
      sycl::detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  BlockT Ret =
      __spirv_SubgroupBlockReadINTEL<BlockT>(reinterpret_cast<PtrT>(src.get()));

  return sycl::bit_cast<T>(Ret);
}

template <int N, typename T, access::address_space Space>
vec<T, N> load(const multi_ptr<T, Space> src) {
  using BlockT = SelectBlockT<T>;
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  using PtrT =
      sycl::detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  VecT Ret =
      __spirv_SubgroupBlockReadINTEL<VecT>(reinterpret_cast<PtrT>(src.get()));

  return sycl::bit_cast<typename vec<T, N>::vector_t>(Ret);
}

template <typename T, access::address_space Space>
void store(multi_ptr<T, Space> dst, const T &x) {
  using BlockT = SelectBlockT<T>;
  using PtrT = sycl::detail::ConvertToOpenCLType_t<multi_ptr<BlockT, Space>>;

  __spirv_SubgroupBlockWriteINTEL(reinterpret_cast<PtrT>(dst.get()),
                                  sycl::bit_cast<BlockT>(x));
}

template <int N, typename T, access::address_space Space>
void store(multi_ptr<T, Space> dst, const vec<T, N> &x) {
  using BlockT = SelectBlockT<T>;
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  using PtrT =
      sycl::detail::ConvertToOpenCLType_t<const multi_ptr<BlockT, Space>>;

  __spirv_SubgroupBlockWriteINTEL(reinterpret_cast<PtrT>(dst.get()),
                                  sycl::bit_cast<VecT>(x));
}
#endif // __SYCL_DEVICE_ONLY__

} // namespace sub_group

} // namespace detail

namespace ext {
namespace oneapi {

struct sub_group;
namespace experimental {
inline sub_group this_sub_group();
}

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
    return __spirv_SubgroupSize();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  range_type get_max_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupMaxSize();
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_SubgroupId();
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
    return __spirv_NumSubgroups();
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
    return sycl::detail::spirv::SubgroupShuffle(x, local_id);
#else
    (void)x;
    (void)local_id;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T> T shuffle_down(T x, uint32_t delta) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::spirv::SubgroupShuffleDown(x, delta);
#else
    (void)x;
    (void)delta;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T> T shuffle_up(T x, uint32_t delta) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::spirv::SubgroupShuffleUp(x, delta);
#else
    (void)x;
    (void)delta;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T> T shuffle_xor(T x, id_type value) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::spirv::SubgroupShuffleXor(x, value);
#else
    (void)x;
    (void)value;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  /* --- sub_group load/stores --- */
  /* these can map to SIMD or block read/write hardware where available */
#ifdef __SYCL_DEVICE_ONLY__
  // Method for decorated pointer
  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  detail::enable_if_t<
      !std::is_same<typename detail::remove_AS<T>::type, T>::value, T>
  load(CVT *cv_src) const {
    T *src = const_cast<T *>(cv_src);
    return load(sycl::multi_ptr<typename detail::remove_AS<T>::type,
                                sycl::detail::deduce_AS<T>::value>(
        (typename detail::remove_AS<T>::type *)src));
  }

  // Method for raw pointer
  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  detail::enable_if_t<
      std::is_same<typename detail::remove_AS<T>::type, T>::value, T>
  load(CVT *cv_src) const {
    T *src = const_cast<T *>(cv_src);

#ifdef __NVPTX__
    return src[get_local_id()[0]];
#else  // __NVPTX__
    auto l = __SYCL_GenericCastToPtrExplicit_ToLocal<T>(src);
    if (l)
      return load(l);

    auto g = __SYCL_GenericCastToPtrExplicit_ToGlobal<T>(src);
    if (g)
      return load(g);

    assert(!"Sub-group load() is supported for local or global pointers only.");
    return {};
#endif // __NVPTX__
  }
#else  //__SYCL_DEVICE_ONLY__
  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  T load(CVT *src) const {
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
  }
#endif //__SYCL_DEVICE_ONLY__

  template <typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value, T>
  load(const multi_ptr<CVT, Space> cv_src) const {
    multi_ptr<T, Space> src = const_cast<T *>(static_cast<CVT *>(cv_src));
#ifdef __SYCL_DEVICE_ONLY__
#ifdef __NVPTX__
    return src.get()[get_local_id()[0]];
#else
    return sycl::detail::sub_group::load(src);
#endif // __NVPTX__
#else
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value, T>
  load(const multi_ptr<CVT, Space> cv_src) const {
    multi_ptr<T, Space> src = const_cast<T *>(static_cast<CVT *>(cv_src));
#ifdef __SYCL_DEVICE_ONLY__
    return src.get()[get_local_id()[0]];
#else
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }
#ifdef __SYCL_DEVICE_ONLY__
#ifdef __NVPTX__
  template <int N, typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space> cv_src) const {
    multi_ptr<T, Space> src = const_cast<T *>(static_cast<CVT *>(cv_src));
    vec<T, N> res;
    for (int i = 0; i < N; ++i) {
      res[i] = *(src.get() + i * get_max_local_range()[0] + get_local_id()[0]);
    }
    return res;
  }
#else  // __NVPTX__
  template <int N, typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N != 1 && N != 3 && N != 16,
      vec<T, N>>
  load(const multi_ptr<CVT, Space> cv_src) const {
    multi_ptr<T, Space> src = const_cast<T *>(static_cast<CVT *>(cv_src));
    return sycl::detail::sub_group::load<N, T>(src);
  }

  template <int N, typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 16,
      vec<T, 16>>
  load(const multi_ptr<CVT, Space> cv_src) const {
    multi_ptr<T, Space> src = const_cast<T *>(static_cast<CVT *>(cv_src));
    return {sycl::detail::sub_group::load<8, T>(src),
            sycl::detail::sub_group::load<8, T>(src +
                                                8 * get_max_local_range()[0])};
  }

  template <int N, typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 3,
      vec<T, 3>>
  load(const multi_ptr<CVT, Space> cv_src) const {
    multi_ptr<T, Space> src = const_cast<T *>(static_cast<CVT *>(cv_src));
    return {
        sycl::detail::sub_group::load<1, T>(src),
        sycl::detail::sub_group::load<2, T>(src + get_max_local_range()[0])};
  }

  template <int N, typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 1,
      vec<T, 1>>
  load(const multi_ptr<CVT, Space> cv_src) const {
    multi_ptr<T, Space> src = const_cast<T *>(static_cast<CVT *>(cv_src));
    return sycl::detail::sub_group::load(src);
  }
#endif // ___NVPTX___
#else  // __SYCL_DEVICE_ONLY__
  template <int N, typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space> src) const {
    (void)src;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
  }
#endif // __SYCL_DEVICE_ONLY__

  template <int N, typename CVT, access::address_space Space,
            typename T = std::remove_cv_t<CVT>>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space> cv_src) const {
    multi_ptr<T, Space> src = const_cast<T *>(static_cast<CVT *>(cv_src));
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

#ifdef __SYCL_DEVICE_ONLY__
  // Method for decorated pointer
  template <typename T>
  detail::enable_if_t<
      !std::is_same<typename detail::remove_AS<T>::type, T>::value>
  store(T *dst, const typename detail::remove_AS<T>::type &x) const {
    store(sycl::multi_ptr<typename detail::remove_AS<T>::type,
                          sycl::detail::deduce_AS<T>::value>(
              (typename detail::remove_AS<T>::type *)dst),
          x);
  }

  // Method for raw pointer
  template <typename T>
  detail::enable_if_t<
      std::is_same<typename detail::remove_AS<T>::type, T>::value>
  store(T *dst, const typename detail::remove_AS<T>::type &x) const {

#ifdef __NVPTX__
    dst[get_local_id()[0]] = x;
#else  // __NVPTX__
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
#endif // __NVPTX__
  }
#else  //__SYCL_DEVICE_ONLY__
  template <typename T> void store(T *dst, const T &x) const {
    (void)dst;
    (void)x;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
  }
#endif //__SYCL_DEVICE_ONLY__

  template <typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space> dst, const T &x) const {
#ifdef __SYCL_DEVICE_ONLY__
#ifdef __NVPTX__
    dst.get()[get_local_id()[0]] = x;
#else
    sycl::detail::sub_group::store(dst, x);
#endif // __NVPTX__
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

#ifdef __SYCL_DEVICE_ONLY__
#ifdef __NVPTX__
  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space> dst, const vec<T, N> &x) const {
    for (int i = 0; i < N; ++i) {
      *(dst.get() + i * get_max_local_range()[0] + get_local_id()[0]) = x[i];
    }
  }
#else // __NVPTX__
  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N != 1 && N != 3 && N != 16>
  store(multi_ptr<T, Space> dst, const vec<T, N> &x) const {
    sycl::detail::sub_group::store(dst, x);
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 1>
  store(multi_ptr<T, Space> dst, const vec<T, 1> &x) const {
    sycl::detail::sub_group::store(dst, x);
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 3>
  store(multi_ptr<T, Space> dst, const vec<T, 3> &x) const {
    store<1, T, Space>(dst, x.s0());
    store<2, T, Space>(dst + get_max_local_range()[0], {x.s1(), x.s2()});
  }

  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 16>
  store(multi_ptr<T, Space> dst, const vec<T, 16> &x) const {
    store<8, T, Space>(dst, x.lo());
    store<8, T, Space>(dst + 8 * get_max_local_range()[0], x.hi());
  }

#endif // __NVPTX__
#else  // __SYCL_DEVICE_ONLY__
  template <int N, typename T, access::address_space Space>
  sycl::detail::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space> dst, const vec<T, N> &x) const {
    (void)dst;
    (void)x;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
  }
#endif // __SYCL_DEVICE_ONLY__

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
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::reduce instead.")
  EnableIfIsScalarArithmetic<T> reduce(T x, BinaryOperation op) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::calc<T, __spv::GroupOperation::Reduce,
                              __spv::Scope::Subgroup>(
        typename sycl::detail::GroupOpTag<T>::type(), x, op);
#else
    (void)x;
    (void)op;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
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
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::exclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> exclusive_scan(T x, BinaryOperation op) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::calc<T, __spv::GroupOperation::ExclusiveScan,
                              __spv::Scope::Subgroup>(
        typename sycl::detail::GroupOpTag<T>::type(), x, op);
#else
    (void)x;
    (void)op;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
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
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  template <typename T, class BinaryOperation>
  __SYCL_DEPRECATED("Collectives in the sub-group class are deprecated. Use "
                    "sycl::ext::oneapi::inclusive_scan instead.")
  EnableIfIsScalarArithmetic<T> inclusive_scan(T x, BinaryOperation op) const {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::detail::calc<T, __spv::GroupOperation::InclusiveScan,
                              __spv::Scope::Subgroup>(
        typename sycl::detail::GroupOpTag<T>::type(), x, op);
#else
    (void)x;
    (void)op;
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
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
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    return get_local_linear_id() == 0;
#else
    throw runtime_error("Sub-groups are not supported on host device.",
                        PI_INVALID_DEVICE);
#endif
  }

protected:
  template <int dimensions> friend class cl::sycl::nd_item;
  friend sub_group this_sub_group();
  friend sub_group experimental::this_sub_group();
  sub_group() = default;
};

__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::experimental::this_sub_group() instead")
inline sub_group this_sub_group() {
#ifdef __SYCL_DEVICE_ONLY__
  return sub_group();
#else
  throw runtime_error("Sub-groups are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
