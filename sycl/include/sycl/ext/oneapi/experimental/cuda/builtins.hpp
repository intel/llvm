//==--- builtins.hpp - SYCL_ONEAPI_CUDA experimental builtins  -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#define SYCL_EXT_ONEAPI_CUDA_TEX_CACHE_READ 1

#include <sycl/types.hpp>

#if defined(_WIN32) || defined(_WIN64)
#define ATTRIBUTE_EXT_VEC_TYPE(N) __declspec(ext_vector_type(N))
#else
#define ATTRIBUTE_EXT_VEC_TYPE(N) __attribute__((ext_vector_type(N)))
#endif

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace cuda {

namespace detail {
using ldg_vector_types = sycl::detail::type_list<
    sycl::vec<char, 2>, sycl::vec<char, 3>, sycl::vec<char, 4>,
    sycl::vec<signed char, 2>, sycl::vec<signed char, 3>,
    sycl::vec<signed char, 4>, sycl::vec<short, 2>, sycl::vec<short, 3>,
    sycl::vec<short, 4>, sycl::vec<int, 2>, sycl::vec<int, 3>,
    sycl::vec<int, 4>, sycl::vec<long, 2>, sycl::vec<long, 3>,
    sycl::vec<long, 4>, sycl::vec<long long, 2>, sycl::vec<long long, 3>,
    sycl::vec<long long, 4>, sycl::vec<unsigned char, 2>,
    sycl::vec<unsigned char, 3>, sycl::vec<unsigned char, 4>,
    sycl::vec<unsigned short, 2>, sycl::vec<unsigned short, 3>,
    sycl::vec<unsigned short, 4>, sycl::vec<unsigned int, 2>,
    sycl::vec<unsigned int, 3>, sycl::vec<unsigned int, 4>,
    sycl::vec<unsigned long, 2>, sycl::vec<unsigned long, 3>,
    sycl::vec<unsigned long, 4>, sycl::vec<unsigned long long, 2>,
    sycl::vec<unsigned long long, 3>, sycl::vec<unsigned long long, 4>,
    sycl::vec<half, 2>, sycl::vec<half, 3>, sycl::vec<half, 4>,
    sycl::vec<float, 2>, sycl::vec<float, 3>, sycl::vec<float, 4>,
    sycl::vec<double, 2>, sycl::vec<double, 3>, sycl::vec<double, 4>>;

using ldg_types =
    sycl::detail::tl_append<ldg_vector_types,
                            sycl::detail::gtl::scalar_signed_basic_list,
                            sycl::detail::gtl::scalar_unsigned_basic_list>;
} // namespace detail

template <typename T>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    sycl::detail::is_contained<
        T, sycl::ext::oneapi::experimental::cuda::detail::ldg_types>::value,
    T>
ldg(const T *ptr) {
#if defined(__SYCL_DEVICE_ONLY__)
#if defined(__NVPTX__)
  if constexpr (std::is_same_v<T, char>) {
    return __nvvm_ldg_c(ptr);
  } else if constexpr (std::is_same_v<T, signed char>) {
    return __nvvm_ldg_sc(ptr);
  } else if constexpr (std::is_same_v<T, short>) {
    return __nvvm_ldg_s(ptr);
  } else if constexpr (std::is_same_v<T, int>) {
    return __nvvm_ldg_i(ptr);
  } else if constexpr (std::is_same_v<T, long>) {
    return __nvvm_ldg_l(ptr);
  } else if constexpr (std::is_same_v<T, long long>) {
    return __nvvm_ldg_ll(ptr);
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return __nvvm_ldg_uc(ptr);
  } else if constexpr (std::is_same_v<T, unsigned short>) {
    return __nvvm_ldg_us(ptr);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return __nvvm_ldg_ui(ptr);
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return __nvvm_ldg_ul(ptr);
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    return __nvvm_ldg_ull(ptr);
  } else if constexpr (std::is_same_v<T, half>) {
    auto native = reinterpret_cast<const __fp16 *>(ptr);
    return __nvvm_ldg_h(native);
  } else if constexpr (std::is_same_v<T, float>) {
    return __nvvm_ldg_f(ptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return __nvvm_ldg_d(ptr);
  } else if constexpr (std::is_same_v<T, sycl::vec<char, 2>>) {
    // We can assume that ptr is aligned at least to char2's alignment, but the
    // load will assume that ptr is aligned to char2's alignment.  This is only
    // safe if alignof(f2) <= alignof(char2).
    typedef char c2 ATTRIBUTE_EXT_VEC_TYPE(2);
    c2 rv = __nvvm_ldg_c2(reinterpret_cast<const c2 *>(ptr));
    sycl::vec<char, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<char, 3>>) {
    typedef char c2 ATTRIBUTE_EXT_VEC_TYPE(2);
    c2 rv_2 = __nvvm_ldg_c2(reinterpret_cast<const c2 *>(ptr));
    char rv = __nvvm_ldg_c(reinterpret_cast<const char *>(
        std::next(reinterpret_cast<const c2 *>(ptr))));
    sycl::vec<char, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<char, 4>>) {
    typedef char c4 ATTRIBUTE_EXT_VEC_TYPE(4);
    c4 rv = __nvvm_ldg_c4(reinterpret_cast<const c4 *>(ptr));
    sycl::vec<char, 4> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<signed char, 2>>) {
    typedef signed char sc2 ATTRIBUTE_EXT_VEC_TYPE(2);
    sc2 rv = __nvvm_ldg_sc2(reinterpret_cast<const sc2 *>(ptr));
    sycl::vec<signed char, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<signed char, 3>>) {
    typedef signed char sc2 ATTRIBUTE_EXT_VEC_TYPE(2);
    sc2 rv_2 = __nvvm_ldg_sc2(reinterpret_cast<const sc2 *>(ptr));
    signed char rv = __nvvm_ldg_sc(reinterpret_cast<const signed char *>(
        std::next(reinterpret_cast<const sc2 *>(ptr))));
    sycl::vec<signed char, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<signed char, 4>>) {
    typedef signed char sc4 ATTRIBUTE_EXT_VEC_TYPE(4);
    sc4 rv = __nvvm_ldg_sc4(reinterpret_cast<const sc4 *>(ptr));
    sycl::vec<signed char, 4> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<short, 2>>) {
    typedef short s2 ATTRIBUTE_EXT_VEC_TYPE(2);
    s2 rv = __nvvm_ldg_s2(reinterpret_cast<const s2 *>(ptr));
    sycl::vec<short, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<short, 3>>) {
    typedef short s2 ATTRIBUTE_EXT_VEC_TYPE(2);
    s2 rv_2 = __nvvm_ldg_s2(reinterpret_cast<const s2 *>(ptr));
    short rv = __nvvm_ldg_s(reinterpret_cast<const short *>(
        std::next(reinterpret_cast<const s2 *>(ptr))));
    sycl::vec<short, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<short, 4>>) {
    typedef short s4 ATTRIBUTE_EXT_VEC_TYPE(4);
    s4 rv = __nvvm_ldg_s4(reinterpret_cast<const s4 *>(ptr));
    sycl::vec<short, 4> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<int, 2>>) {
    typedef int i2 ATTRIBUTE_EXT_VEC_TYPE(2);
    i2 rv = __nvvm_ldg_i2(reinterpret_cast<const i2 *>(ptr));
    sycl::vec<int, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<int, 3>>) {
    typedef int i2 ATTRIBUTE_EXT_VEC_TYPE(2);
    i2 rv_2 = __nvvm_ldg_i2(reinterpret_cast<const i2 *>(ptr));
    int rv = __nvvm_ldg_i(reinterpret_cast<const int *>(
        std::next(reinterpret_cast<const i2 *>(ptr))));
    sycl::vec<int, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<int, 4>>) {
    typedef int i4 ATTRIBUTE_EXT_VEC_TYPE(4);
    i4 rv = __nvvm_ldg_i4(reinterpret_cast<const i4 *>(ptr));
    sycl::vec<int, 4> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<long, 2>>) {
    typedef long l2 ATTRIBUTE_EXT_VEC_TYPE(2);
    l2 rv = __nvvm_ldg_l2(reinterpret_cast<const l2 *>(ptr));
    sycl::vec<long, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<long, 3>>) {
    typedef long l2 ATTRIBUTE_EXT_VEC_TYPE(2);
    l2 rv_2 = __nvvm_ldg_l2(reinterpret_cast<const l2 *>(ptr));
    long rv = __nvvm_ldg_l(reinterpret_cast<const long *>(
        std::next(reinterpret_cast<const l2 *>(ptr))));
    sycl::vec<long, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<long, 4>>) {
    typedef long l2 ATTRIBUTE_EXT_VEC_TYPE(2);
    l2 rv1 = __nvvm_ldg_l2(reinterpret_cast<const l2 *>(ptr));
    l2 rv2 = __nvvm_ldg_l2(std::next(reinterpret_cast<const l2 *>(ptr)));
    sycl::vec<long, 4> ret;
    ret.x() = rv1[0];
    ret.y() = rv1[1];
    ret.z() = rv2[0];
    ret.w() = rv2[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<long long, 2>>) {
    typedef long long ll2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ll2 rv = __nvvm_ldg_ll2(reinterpret_cast<const ll2 *>(ptr));
    sycl::vec<long long, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<long long, 3>>) {
    typedef long long ll2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ll2 rv_2 = __nvvm_ldg_ll2(reinterpret_cast<const ll2 *>(ptr));
    long long rv = __nvvm_ldg_ll(reinterpret_cast<const long long *>(
        std::next(reinterpret_cast<const ll2 *>(ptr))));
    sycl::vec<long long, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<long long, 4>>) {
    typedef long long ll2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ll2 rv1 = __nvvm_ldg_ll2(reinterpret_cast<const ll2 *>(ptr));
    ll2 rv2 = __nvvm_ldg_ll2(std::next(reinterpret_cast<const ll2 *>(ptr)));
    sycl::vec<long long, 4> ret;
    ret.x() = rv1[0];
    ret.y() = rv1[1];
    ret.z() = rv2[0];
    ret.w() = rv2[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned char, 2>>) {
    typedef unsigned char uc2 ATTRIBUTE_EXT_VEC_TYPE(2);
    uc2 rv = __nvvm_ldg_uc2(reinterpret_cast<const uc2 *>(ptr));
    sycl::vec<unsigned char, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned char, 3>>) {
    typedef unsigned char uc2 ATTRIBUTE_EXT_VEC_TYPE(2);
    uc2 rv_2 = __nvvm_ldg_uc2(reinterpret_cast<const uc2 *>(ptr));
    unsigned char rv = __nvvm_ldg_uc(reinterpret_cast<const unsigned char *>(
        std::next(reinterpret_cast<const uc2 *>(ptr))));
    sycl::vec<unsigned char, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned char, 4>>) {
    typedef unsigned char uc4 ATTRIBUTE_EXT_VEC_TYPE(4);
    uc4 rv = __nvvm_ldg_uc4(reinterpret_cast<const uc4 *>(ptr));
    sycl::vec<unsigned char, 4> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned short, 2>>) {
    typedef unsigned short us2 ATTRIBUTE_EXT_VEC_TYPE(2);
    us2 rv = __nvvm_ldg_us2(reinterpret_cast<const us2 *>(ptr));
    sycl::vec<unsigned short, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned short, 3>>) {
    typedef unsigned short us2 ATTRIBUTE_EXT_VEC_TYPE(2);
    us2 rv_2 = __nvvm_ldg_us2(reinterpret_cast<const us2 *>(ptr));
    unsigned short rv = __nvvm_ldg_us(reinterpret_cast<const unsigned short *>(
        std::next(reinterpret_cast<const us2 *>(ptr))));
    sycl::vec<unsigned short, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned short, 4>>) {
    typedef unsigned short us4 ATTRIBUTE_EXT_VEC_TYPE(4);
    us4 rv = __nvvm_ldg_us4(reinterpret_cast<const us4 *>(ptr));
    sycl::vec<unsigned short, 4> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned int, 2>>) {
    typedef unsigned int ui2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ui2 rv = __nvvm_ldg_ui2(reinterpret_cast<const ui2 *>(ptr));
    sycl::vec<unsigned int, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned int, 3>>) {
    typedef unsigned int ui2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ui2 rv_2 = __nvvm_ldg_ui2(reinterpret_cast<const ui2 *>(ptr));
    unsigned int rv = __nvvm_ldg_ui(reinterpret_cast<const unsigned int *>(
        std::next(reinterpret_cast<const ui2 *>(ptr))));
    sycl::vec<unsigned int, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned int, 4>>) {
    typedef unsigned int ui4 ATTRIBUTE_EXT_VEC_TYPE(4);
    ui4 rv = __nvvm_ldg_ui4(reinterpret_cast<const ui4 *>(ptr));
    sycl::vec<unsigned int, 4> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned long, 2>>) {
    typedef unsigned long ul2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ul2 rv = __nvvm_ldg_ul2(reinterpret_cast<const ul2 *>(ptr));
    sycl::vec<unsigned long, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned long, 3>>) {
    typedef unsigned long ul2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ul2 rv_2 = __nvvm_ldg_ul2(reinterpret_cast<const ul2 *>(ptr));
    unsigned long rv = __nvvm_ldg_ul(reinterpret_cast<const unsigned long *>(
        std::next(reinterpret_cast<const ul2 *>(ptr))));
    sycl::vec<unsigned long, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned long, 4>>) {
    typedef unsigned long ul2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ul2 rv1 = __nvvm_ldg_ul2(reinterpret_cast<const ul2 *>(ptr));
    ul2 rv2 = __nvvm_ldg_ul2(std::next(reinterpret_cast<const ul2 *>(ptr)));
    sycl::vec<unsigned long, 4> ret;
    ret.x() = rv1[0];
    ret.y() = rv1[1];
    ret.z() = rv2[0];
    ret.w() = rv2[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned long long, 2>>) {
    typedef unsigned long long ull2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ull2 rv = __nvvm_ldg_ull2(reinterpret_cast<const ull2 *>(ptr));
    sycl::vec<unsigned long long, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned long long, 3>>) {
    typedef unsigned long long ull2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ull2 rv_2 = __nvvm_ldg_ull2(reinterpret_cast<const ull2 *>(ptr));
    unsigned long long rv =
        __nvvm_ldg_ull(reinterpret_cast<const unsigned long long *>(
            std::next(reinterpret_cast<const ull2 *>(ptr))));
    sycl::vec<unsigned long long, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<unsigned long long, 4>>) {
    typedef unsigned long long ull2 ATTRIBUTE_EXT_VEC_TYPE(2);
    ull2 rv1 = __nvvm_ldg_ull2(reinterpret_cast<const ull2 *>(ptr));
    ull2 rv2 = __nvvm_ldg_ull2(std::next(reinterpret_cast<const ull2 *>(ptr)));
    sycl::vec<unsigned long long, 4> ret;
    ret.x() = rv1[0];
    ret.y() = rv1[1];
    ret.z() = rv2[0];
    ret.w() = rv2[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<half, 2>>) {
    typedef __fp16 h2 ATTRIBUTE_EXT_VEC_TYPE(2);
    auto rv = __nvvm_ldg_h2(reinterpret_cast<const h2 *>(ptr));
    sycl::vec<half, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<half, 3>>) {
    typedef __fp16 h2 ATTRIBUTE_EXT_VEC_TYPE(2);
    h2 rv_2 = __nvvm_ldg_h2(reinterpret_cast<const h2 *>(ptr));
    auto rv = __nvvm_ldg_h(reinterpret_cast<const __fp16 *>(
        std::next(reinterpret_cast<const h2 *>(ptr))));
    sycl::vec<half, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<half, 4>>) {
    typedef __fp16 h2 ATTRIBUTE_EXT_VEC_TYPE(2);
    auto rv1 = __nvvm_ldg_h2(reinterpret_cast<const h2 *>(ptr));
    auto rv2 = __nvvm_ldg_h2(std::next(reinterpret_cast<const h2 *>(ptr)));
    sycl::vec<half, 4> ret;
    ret.x() = rv1[0];
    ret.y() = rv1[1];
    ret.z() = rv2[0];
    ret.w() = rv2[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<float, 2>>) {
    typedef float f2 ATTRIBUTE_EXT_VEC_TYPE(2);
    f2 rv = __nvvm_ldg_f2(reinterpret_cast<const f2 *>(ptr));
    sycl::vec<float, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<float, 3>>) {
    typedef float f2 ATTRIBUTE_EXT_VEC_TYPE(2);
    f2 rv_2 = __nvvm_ldg_f2(reinterpret_cast<const f2 *>(ptr));
    float rv = __nvvm_ldg_f(reinterpret_cast<const float *>(
        std::next(reinterpret_cast<const f2 *>(ptr))));
    sycl::vec<float, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<float, 4>>) {
    typedef float f4 ATTRIBUTE_EXT_VEC_TYPE(4);
    f4 rv = __nvvm_ldg_f4(reinterpret_cast<const f4 *>(ptr));
    sycl::vec<float, 4> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    ret.z() = rv[2];
    ret.w() = rv[3];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<double, 2>>) {
    typedef double d2 ATTRIBUTE_EXT_VEC_TYPE(2);
    d2 rv = __nvvm_ldg_d2(reinterpret_cast<const d2 *>(ptr));
    sycl::vec<double, 2> ret;
    ret.x() = rv[0];
    ret.y() = rv[1];
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<double, 3>>) {
    typedef double d2 ATTRIBUTE_EXT_VEC_TYPE(2);
    d2 rv_2 = __nvvm_ldg_d2(reinterpret_cast<const d2 *>(ptr));
    double rv = __nvvm_ldg_d(reinterpret_cast<const double *>(
        std::next(reinterpret_cast<const d2 *>(ptr))));
    sycl::vec<double, 3> ret;
    ret.x() = rv_2[0];
    ret.y() = rv_2[1];
    ret.z() = rv;
    return ret;
  } else if constexpr (std::is_same_v<T, sycl::vec<double, 4>>) {
    typedef double d2 ATTRIBUTE_EXT_VEC_TYPE(2);
    d2 rv1 = __nvvm_ldg_d2(reinterpret_cast<const d2 *>(ptr));
    d2 rv2 = __nvvm_ldg_d2(std::next(reinterpret_cast<const d2 *>(ptr)));
    sycl::vec<double, 4> ret;
    ret.x() = rv1[0];
    ret.y() = rv1[1];
    ret.z() = rv2[0];
    ret.w() = rv2[1];
    return ret;
  }
#else
  return *ptr;
#endif
#else
  throw exception(make_error_code(errc::runtime),
                  "ldg is not supported on host.");
#endif
}

#undef ATTRIBUTE_EXT_VEC_TYPE

} // namespace cuda
} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
