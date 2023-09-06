//==----------- builtins_helper.hpp - SYCL built-in helper  ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/exception.hpp>
#include <sycl/pointers.hpp>
#include <sycl/types.hpp>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

#define __MAKE_1V(Fun, Call, N, Ret, Arg1)                                     \
  __SYCL_EXPORT sycl::vec<Ret, N> Fun __NOEXC(sycl::vec<Arg1, N> x) {          \
    sycl::vec<Ret, N> r;                                                       \
    detail::helper<N - 1>().run_1v(                                            \
        r, [](Arg1 x) { return __host_std::Call(x); }, x);                     \
    return r;                                                                  \
  }

#define __MAKE_1V_2V(Fun, Call, N, Ret, Arg1, Arg2)                            \
  __SYCL_EXPORT sycl::vec<Ret, N> Fun __NOEXC(sycl::vec<Arg1, N> x,            \
                                              sycl::vec<Arg2, N> y) {          \
    sycl::vec<Ret, N> r;                                                       \
    detail::helper<N - 1>().run_1v_2v(                                         \
        r, [](Arg1 x, Arg2 y) { return __host_std::Call(x, y); }, x, y);       \
    return r;                                                                  \
  }

#define __MAKE_1V_2V_RS(Fun, Call, N, Ret, Arg1, Arg2)                         \
  __SYCL_EXPORT Ret Fun __NOEXC(sycl::vec<Arg1, N> x, sycl::vec<Arg2, N> y) {  \
    Ret r = Ret();                                                             \
    detail::helper<N - 1>().run_1v_2v_rs(                                      \
        r, [](Ret &r, Arg1 x, Arg2 y) { return __host_std::Call(r, x, y); },   \
        x, y);                                                                 \
    return r;                                                                  \
  }

#define __MAKE_1V_RS(Fun, Call, N, Ret, Arg1)                                  \
  __SYCL_EXPORT Ret Fun __NOEXC(sycl::vec<Arg1, N> x) {                        \
    Ret r = Ret();                                                             \
    detail::helper<N - 1>().run_1v_rs(                                         \
        r, [](Ret &r, Arg1 x) { return __host_std::Call(r, x); }, x);          \
    return r;                                                                  \
  }

#define __MAKE_1V_2V_3V(Fun, Call, N, Ret, Arg1, Arg2, Arg3)                   \
  __SYCL_EXPORT sycl::vec<Ret, N> Fun __NOEXC(                                 \
      sycl::vec<Arg1, N> x, sycl::vec<Arg2, N> y, sycl::vec<Arg3, N> z) {      \
    sycl::vec<Ret, N> r;                                                       \
    detail::helper<N - 1>().run_1v_2v_3v(                                      \
        r, [](Arg1 x, Arg2 y, Arg3 z) { return __host_std::Call(x, y, z); },   \
        x, y, z);                                                              \
    return r;                                                                  \
  }

#define __MAKE_1V_2S_3S(Fun, N, Ret, Arg1, Arg2, Arg3)                         \
  __SYCL_EXPORT sycl::vec<Ret, N> Fun __NOEXC(sycl::vec<Arg1, N> x, Arg2 y,    \
                                              Arg3 z) {                        \
    sycl::vec<Ret, N> r;                                                       \
    detail::helper<N - 1>().run_1v_2s_3s(                                      \
        r, [](Arg1 x, Arg2 y, Arg3 z) { return __host_std::Fun(x, y, z); }, x, \
        y, z);                                                                 \
    return r;                                                                  \
  }

#define __MAKE_1V_2S(Fun, N, Ret, Arg1, Arg2)                                  \
  __SYCL_EXPORT sycl::vec<Ret, N> Fun __NOEXC(sycl::vec<Arg1, N> x, Arg2 y) {  \
    sycl::vec<Ret, N> r;                                                       \
    detail::helper<N - 1>().run_1v_2s(                                         \
        r, [](Arg1 x, Arg2 y) { return __host_std::Fun(x, y); }, x, y);        \
    return r;                                                                  \
  }

#define __MAKE_SR_1V_AND(Fun, Call, N, Ret, Arg1)                              \
  __SYCL_EXPORT Ret Fun __NOEXC(sycl::vec<Arg1, N> x) {                        \
    Ret r;                                                                     \
    detail::helper<N - 1>().run_1v_sr_and(                                     \
        r, [](Arg1 x) { return __host_std::Call(x); }, x);                     \
    return r;                                                                  \
  }

#define __MAKE_SR_1V_OR(Fun, Call, N, Ret, Arg1)                               \
  __SYCL_EXPORT Ret Fun __NOEXC(sycl::vec<Arg1, N> x) {                        \
    Ret r;                                                                     \
    detail::helper<N - 1>().run_1v_sr_or(                                      \
        r, [](Arg1 x) { return __host_std::Call(x); }, x);                     \
    return r;                                                                  \
  }

#define __MAKE_1V_2P(Fun, N, Ret, Arg1, Arg2)                                  \
  __SYCL_EXPORT sycl::vec<Ret, N> Fun __NOEXC(sycl::vec<Arg1, N> x,            \
                                              sycl::vec<Arg2, N> *y) {         \
    sycl::vec<Ret, N> r;                                                       \
    detail::helper<N - 1>().run_1v_2p(                                         \
        r, [](Arg1 x, Arg2 *y) { return __host_std::Fun(x, y); }, x, y);       \
    return r;                                                                  \
  }

#define __MAKE_1V_2V_3P(Fun, N, Ret, Arg1, Arg2, Arg3)                         \
  __SYCL_EXPORT sycl::vec<Ret, N> Fun __NOEXC(                                 \
      sycl::vec<Arg1, N> x, sycl::vec<Arg2, N> y, sycl::vec<Arg3, N> *z) {     \
    sycl::vec<Ret, N> r;                                                       \
    detail::helper<N - 1>().run_1v_2v_3p(                                      \
        r, [](Arg1 x, Arg2 y, Arg3 *z) { return __host_std::Fun(x, y, z); },   \
        x, y, z);                                                              \
    return r;                                                                  \
  }

#define MAKE_1V(Fun, Ret, Arg1) MAKE_1V_FUNC(Fun, Fun, Ret, Arg1)

#define MAKE_1V_FUNC(Fun, Call, Ret, Arg1)                                     \
  __MAKE_1V(Fun, Call, 1, Ret, Arg1)                                           \
  __MAKE_1V(Fun, Call, 2, Ret, Arg1)                                           \
  __MAKE_1V(Fun, Call, 3, Ret, Arg1)                                           \
  __MAKE_1V(Fun, Call, 4, Ret, Arg1)                                           \
  __MAKE_1V(Fun, Call, 8, Ret, Arg1)                                           \
  __MAKE_1V(Fun, Call, 16, Ret, Arg1)

#define MAKE_1V_2V(Fun, Ret, Arg1, Arg2)                                       \
  MAKE_1V_2V_FUNC(Fun, Fun, Ret, Arg1, Arg2)

#define MAKE_1V_2V_FUNC(Fun, Call, Ret, Arg1, Arg2)                            \
  __MAKE_1V_2V(Fun, Call, 1, Ret, Arg1, Arg2)                                  \
  __MAKE_1V_2V(Fun, Call, 2, Ret, Arg1, Arg2)                                  \
  __MAKE_1V_2V(Fun, Call, 3, Ret, Arg1, Arg2)                                  \
  __MAKE_1V_2V(Fun, Call, 4, Ret, Arg1, Arg2)                                  \
  __MAKE_1V_2V(Fun, Call, 8, Ret, Arg1, Arg2)                                  \
  __MAKE_1V_2V(Fun, Call, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2V_3V(Fun, Ret, Arg1, Arg2, Arg3)                              \
  MAKE_1V_2V_3V_FUNC(Fun, Fun, Ret, Arg1, Arg2, Arg3)

#define MAKE_1V_2V_3V_FUNC(Fun, Call, Ret, Arg1, Arg2, Arg3)                   \
  __MAKE_1V_2V_3V(Fun, Call, 1, Ret, Arg1, Arg2, Arg3)                         \
  __MAKE_1V_2V_3V(Fun, Call, 2, Ret, Arg1, Arg2, Arg3)                         \
  __MAKE_1V_2V_3V(Fun, Call, 3, Ret, Arg1, Arg2, Arg3)                         \
  __MAKE_1V_2V_3V(Fun, Call, 4, Ret, Arg1, Arg2, Arg3)                         \
  __MAKE_1V_2V_3V(Fun, Call, 8, Ret, Arg1, Arg2, Arg3)                         \
  __MAKE_1V_2V_3V(Fun, Call, 16, Ret, Arg1, Arg2, Arg3)

#define MAKE_SC_1V_2V_3V(Fun, Ret, Arg1, Arg2, Arg3)                           \
  MAKE_SC_3ARG(Fun, Ret, Arg1, Arg2, Arg3)                                     \
  MAKE_1V_2V_3V_FUNC(Fun, Fun, Ret, Arg1, Arg2, Arg3)

#define MAKE_SC_FSC_1V_2V_3V_FV(FunSc, FunV, Ret, Arg1, Arg2, Arg3)            \
  MAKE_SC_3ARG(FunSc, Ret, Arg1, Arg2, Arg3)                                   \
  MAKE_1V_2V_3V_FUNC(FunSc, FunV, Ret, Arg1, Arg2, Arg3)

#define MAKE_SC_3ARG(Fun, Ret, Arg1, Arg2, Arg3)                               \
  __SYCL_EXPORT Ret Fun __NOEXC(Arg1 x, Arg2 y, Arg3 z) {                      \
    return (Ret)__##Fun(x, y, z);                                              \
  }

#define MAKE_1V_2S(Fun, Ret, Arg1, Arg2)                                       \
  __MAKE_1V_2S(Fun, 1, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 2, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 3, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 4, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 8, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2S_3S(Fun, Ret, Arg1, Arg2, Arg3)                              \
  __MAKE_1V_2S_3S(Fun, 1, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2S_3S(Fun, 2, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2S_3S(Fun, 3, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2S_3S(Fun, 4, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2S_3S(Fun, 8, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2S_3S(Fun, 16, Ret, Arg1, Arg2, Arg3)

#define MAKE_SR_1V_AND(Fun, Call, Ret, Arg1)                                   \
  __MAKE_SR_1V_AND(Fun, Call, 1, Ret, Arg1)                                    \
  __MAKE_SR_1V_AND(Fun, Call, 2, Ret, Arg1)                                    \
  __MAKE_SR_1V_AND(Fun, Call, 3, Ret, Arg1)                                    \
  __MAKE_SR_1V_AND(Fun, Call, 4, Ret, Arg1)                                    \
  __MAKE_SR_1V_AND(Fun, Call, 8, Ret, Arg1)                                    \
  __MAKE_SR_1V_AND(Fun, Call, 16, Ret, Arg1)

#define MAKE_SR_1V_OR(Fun, Call, Ret, Arg1)                                    \
  __MAKE_SR_1V_OR(Fun, Call, 1, Ret, Arg1)                                     \
  __MAKE_SR_1V_OR(Fun, Call, 2, Ret, Arg1)                                     \
  __MAKE_SR_1V_OR(Fun, Call, 3, Ret, Arg1)                                     \
  __MAKE_SR_1V_OR(Fun, Call, 4, Ret, Arg1)                                     \
  __MAKE_SR_1V_OR(Fun, Call, 8, Ret, Arg1)                                     \
  __MAKE_SR_1V_OR(Fun, Call, 16, Ret, Arg1)

#define MAKE_1V_2P(Fun, Ret, Arg1, Arg2)                                       \
  __MAKE_1V_2P(Fun, 1, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 2, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 3, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 4, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 8, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 16, Ret, Arg1, Arg2)

#define MAKE_GEO_1V_2V_RS(Fun, Call, Ret, Arg1, Arg2)                          \
  __MAKE_1V_2V_RS(Fun, Call, 1, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 2, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 3, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 4, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 8, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2V_3P(Fun, Ret, Arg1, Arg2, Arg3)                              \
  __MAKE_1V_2V_3P(Fun, 1, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 2, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 3, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 4, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 8, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 16, Ret, Arg1, Arg2, Arg3)

namespace __host_std {
namespace detail {

template <int N> struct helper {
  template <typename Res, typename Op, typename T1>
  inline void run_1v(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v(r, op, x);
    r.template swizzle<N>() = op(x.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2v(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2v(r, op, x, y);
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2s(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2s(r, op, x, y);
    r.template swizzle<N>() = op(x.template swizzle<N>(), y);
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2s_3s(Res &r, Op op, T1 x, T2 y, T3 z) {
    helper<N - 1>().run_1v_2s_3s(r, op, x, y, z);
    r.template swizzle<N>() = op(x.template swizzle<N>(), y, z);
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2v_rs(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2v_rs(r, op, x, y);
    op(r, x.template swizzle<N>(), y.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_rs(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v_rs(r, op, x);
    op(r, x.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2p(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2p(r, op, x, y);
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T2>::type::element_type temp{};
    r.template swizzle<N>() = op(x.template swizzle<N>(), &temp);
    y->template swizzle<N>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2v_3p(Res &r, Op op, T1 x, T2 y, T3 z) {
    helper<N - 1>().run_1v_2v_3p(r, op, x, y, z);
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T3>::type::element_type temp{};
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>(), &temp);
    z->template swizzle<N>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2v_3v(Res &r, Op op, T1 x, T2 y, T3 z) {
    helper<N - 1>().run_1v_2v_3v(r, op, x, y, z);
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>(),
           z.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_or(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v_sr_or(r, op, x);
    r = (op(x.template swizzle<N>()) || r);
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_and(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v_sr_and(r, op, x);
    r = (op(x.template swizzle<N>()) && r);
  }
};

template <> struct helper<0> {
  template <typename Res, typename Op, typename T1>
  inline void run_1v(Res &r, Op op, T1 x) {
    r.template swizzle<0>() = op(x.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2v(Res &r, Op op, T1 x, T2 y) {
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2s(Res &r, Op op, T1 x, T2 y) {
    r.template swizzle<0>() = op(x.template swizzle<0>(), y);
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2s_3s(Res &r, Op op, T1 x, T2 y, T3 z) {
    r.template swizzle<0>() = op(x.template swizzle<0>(), y, z);
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2v_rs(Res &r, Op op, T1 x, T2 y) {
    op(r, x.template swizzle<0>(), y.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_rs(Res &r, Op op, T1 x) {
    op(r, x.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2p(Res &r, Op op, T1 x, T2 y) {
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T2>::type::element_type temp{};
    r.template swizzle<0>() = op(x.template swizzle<0>(), &temp);
    y->template swizzle<0>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2v_3p(Res &r, Op op, T1 x, T2 y, T3 z) {
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T3>::type::element_type temp{};
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>(), &temp);
    z->template swizzle<0>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2v_3v(Res &r, Op op, T1 x, T2 y, T3 z) {
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>(),
           z.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_or(Res &r, Op op, T1 x) {
    r = op(x.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_and(Res &r, Op op, T1 x) {
    r = op(x.template swizzle<0>());
  }
};

} // namespace detail
} // namespace __host_std
