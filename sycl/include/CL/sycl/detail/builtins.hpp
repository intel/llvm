//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>

#include <type_traits>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

#ifdef __SYCL_DEVICE_ONLY__
#define __FUNC_PREFIX_OCL __spirv_ocl_
#define __FUNC_PREFIX_CORE __spirv_
#define __SYCL_EXTERN_IT1(Ret, prefix, call, Arg1)
#define __SYCL_EXTERN_IT2(Ret, prefix, call, Arg1, Arg2)
#define __SYCL_EXTERN_IT2_SAME(Ret, prefix, call, Arg)
#define __SYCL_EXTERN_IT3(Ret, prefix, call, Arg1, Arg2, Arg3)
#else
#define __FUNC_PREFIX_OCL
#define __FUNC_PREFIX_CORE
#define __SYCL_EXTERN_IT1(Ret, prefix, call, Arg)                              \
  extern Ret __SYCL_PPCAT(prefix, call)(Arg)
#define __SYCL_EXTERN_IT2_SAME(Ret, prefix, call, Arg)                         \
  extern Ret __SYCL_PPCAT(prefix, call)(Arg, Arg)
#define __SYCL_EXTERN_IT2(Ret, prefix, call, Arg1, Arg2)                       \
  extern Ret __SYCL_PPCAT(prefix, call)(Arg1, Arg2)
#define __SYCL_EXTERN_IT3(Ret, prefix, call, Arg1, Arg2, Arg3)                 \
  extern Ret __SYCL_PPCAT(prefix, call)(Arg1, Arg2, Arg3)
#endif

#define __SYCL_PPCAT_NX(A, B) A##B
#define __SYCL_PPCAT(A, B) __SYCL_PPCAT_NX(A, B)

#define __SYCL_MAKE_CALL_ARG1(call, prefix)                                    \
  template <typename R, typename T1>                                           \
  inline __SYCL_ALWAYS_INLINE R __invoke_##call(T1 t1) __NOEXC {               \
    using Ret = cl::sycl::detail::ConvertToOpenCLType_t<R>;                    \
    using Arg1 = cl::sycl::detail::ConvertToOpenCLType_t<T1>;                  \
    __SYCL_EXTERN_IT1(Ret, prefix, call, Arg1);                                \
    Arg1 arg1 = cl::sycl::detail::convertDataToType<T1, Arg1>(t1);             \
    Ret ret = __SYCL_PPCAT(prefix, call)(arg1);                                \
    return cl::sycl::detail::convertDataToType<Ret, R>(ret);                   \
  }

#define __SYCL_MAKE_CALL_ARG2(call, prefix)                                    \
  template <typename R, typename T1, typename T2>                              \
  inline __SYCL_ALWAYS_INLINE R __invoke_##call(T1 t1, T2 t2) __NOEXC {        \
    using Ret = cl::sycl::detail::ConvertToOpenCLType_t<R>;                    \
    using Arg1 = cl::sycl::detail::ConvertToOpenCLType_t<T1>;                  \
    using Arg2 = cl::sycl::detail::ConvertToOpenCLType_t<T2>;                  \
    __SYCL_EXTERN_IT2(Ret, prefix, call, Arg1, Arg2);                          \
    Arg1 arg1 = cl::sycl::detail::convertDataToType<T1, Arg1>(t1);             \
    Arg2 arg2 = cl::sycl::detail::convertDataToType<T2, Arg2>(t2);             \
    Ret ret = __SYCL_PPCAT(prefix, call)(arg1, arg2);                          \
    return cl::sycl::detail::convertDataToType<Ret, R>(ret);                   \
  }

#define __SYCL_MAKE_CALL_ARG2_SAME(call, prefix)                               \
  template <typename R, typename T>                                            \
  inline __SYCL_ALWAYS_INLINE R __invoke_##call(T t1, T t2) __NOEXC {          \
    using Ret = cl::sycl::detail::ConvertToOpenCLType_t<R>;                    \
    using Arg = cl::sycl::detail::ConvertToOpenCLType_t<T>;                    \
    __SYCL_EXTERN_IT2_SAME(Ret, prefix, call, Arg);                            \
    Arg arg1 = cl::sycl::detail::convertDataToType<T, Arg>(t1);                \
    Arg arg2 = cl::sycl::detail::convertDataToType<T, Arg>(t2);                \
    Ret ret = __SYCL_PPCAT(prefix, call)(arg1, arg2);                          \
    return cl::sycl::detail::convertDataToType<Ret, R>(ret);                   \
  }

#define __SYCL_MAKE_CALL_ARG2_SAME_RESULT(call, prefix)                        \
  template <typename T>                                                        \
  inline __SYCL_ALWAYS_INLINE T __invoke_##call(T v1, T v2) __NOEXC {          \
    using Type = cl::sycl::detail::ConvertToOpenCLType_t<T>;                   \
    __SYCL_EXTERN_IT2_SAME(Type, prefix, call, Type);                          \
    Type arg1 = cl::sycl::detail::convertDataToType<T, Type>(v1);              \
    Type arg2 = cl::sycl::detail::convertDataToType<T, Type>(v2);              \
    Type ret = __SYCL_PPCAT(prefix, call)(arg1, arg2);                         \
    return cl::sycl::detail::convertDataToType<Type, T>(ret);                  \
  }

#define __SYCL_MAKE_CALL_ARG3(call, prefix)                                    \
  template <typename R, typename T1, typename T2, typename T3>                 \
  inline __SYCL_ALWAYS_INLINE R __invoke_##call(T1 t1, T2 t2, T3 t3) __NOEXC { \
    using Ret = cl::sycl::detail::ConvertToOpenCLType_t<R>;                    \
    using Arg1 = cl::sycl::detail::ConvertToOpenCLType_t<T1>;                  \
    using Arg2 = cl::sycl::detail::ConvertToOpenCLType_t<T2>;                  \
    using Arg3 = cl::sycl::detail::ConvertToOpenCLType_t<T3>;                  \
    __SYCL_EXTERN_IT3(Ret, prefix, call, Arg1, Arg2, Arg3);                    \
    Arg1 arg1 = cl::sycl::detail::convertDataToType<T1, Arg1>(t1);             \
    Arg2 arg2 = cl::sycl::detail::convertDataToType<T2, Arg2>(t2);             \
    Arg3 arg3 = cl::sycl::detail::convertDataToType<T3, Arg3>(t3);             \
    Ret ret = __SYCL_PPCAT(prefix, call)(arg1, arg2, arg3);                    \
    return cl::sycl::detail::convertDataToType<Ret, R>(ret);                   \
  }

#ifndef __SYCL_DEVICE_ONLY__
__SYCL_INLINE_NAMESPACE(cl) {
namespace __host_std {
#endif // __SYCL_DEVICE_ONLY__
/* ----------------- 4.13.3 Math functions. ---------------------------------*/
__SYCL_MAKE_CALL_ARG1(acos, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(acosh, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(acospi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(asin, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(asinh, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(asinpi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(atan, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(atan2, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(atanh, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(atanpi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(atan2pi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(cbrt, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(ceil, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(copysign, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(cos, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(cosh, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(cospi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(erfc, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(erf, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(exp, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(exp2, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(exp10, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(expm1, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(fabs, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(fdim, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(floor, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(fma, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(fmax, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(fmin, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(fmod, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(fract, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(frexp, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(hypot, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(ilogb, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(ldexp, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(lgamma, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(lgamma_r, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(log, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(log2, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(log10, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(log1p, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(logb, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(mad, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(maxmag, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(minmag, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(modf, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(nan, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(nextafter, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(pow, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(pown, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(powr, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(remainder, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(remquo, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(rint, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(rootn, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(round, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(rsqrt, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(sin, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(sincos, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(sinh, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(sinpi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(sqrt, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(tan, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(tanh, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(tanpi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(tgamma, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(trunc, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_cos, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(native_divide, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_exp, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_exp2, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_exp10, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_log, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_log2, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_log10, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(native_powr, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_recip, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_rsqrt, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_sin, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_sqrt, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(native_tan, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_cos, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(half_divide, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_exp, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_exp2, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_exp10, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_log, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_log2, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_log10, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(half_powr, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_recip, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_rsqrt, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_sin, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_sqrt, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(half_tan, __FUNC_PREFIX_OCL)
/* --------------- 4.13.4 Integer functions. --------------------------------*/
__SYCL_MAKE_CALL_ARG1(s_abs, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(u_abs, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_abs_diff, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_abs_diff, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_add_sat, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_add_sat, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_hadd, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_hadd, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_rhadd, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_rhadd, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(s_clamp, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(u_clamp, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(clz, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(ctz, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(s_mad_hi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(u_mad_hi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(u_mad_sat, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(s_mad_sat, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_max, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_max, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_min, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_min, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_mul_hi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_mul_hi, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(rotate, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_sub_sat, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_sub_sat, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_upsample, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_upsample, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(popcount, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(s_mad24, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(u_mad24, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(s_mul24, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(u_mul24, __FUNC_PREFIX_OCL)
/* --------------- 4.13.5 Common functions. ---------------------------------*/
__SYCL_MAKE_CALL_ARG3(fclamp, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(degrees, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(fmax_common, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(fmin_common, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(mix, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(radians, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(step, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(smoothstep, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(sign, __FUNC_PREFIX_OCL)
/* --------------- 4.13.6 Geometric Functions. ------------------------------*/
__SYCL_MAKE_CALL_ARG2(cross, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2_SAME(Dot, __FUNC_PREFIX_CORE)         // dot
__SYCL_MAKE_CALL_ARG2_SAME_RESULT(FMul, __FUNC_PREFIX_CORE) // dot
__SYCL_MAKE_CALL_ARG2(distance, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(length, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(normalize, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG2(fast_distance, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(fast_length, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG1(fast_normalize, __FUNC_PREFIX_OCL)
/* --------------- 4.13.7 Relational functions. -----------------------------*/
__SYCL_MAKE_CALL_ARG2_SAME(FOrdEqual, __FUNC_PREFIX_CORE)       // isequal
__SYCL_MAKE_CALL_ARG2_SAME(FUnordNotEqual, __FUNC_PREFIX_CORE)  // isnotequal
__SYCL_MAKE_CALL_ARG2_SAME(FOrdGreaterThan, __FUNC_PREFIX_CORE) // isgreater
__SYCL_MAKE_CALL_ARG2_SAME(FOrdGreaterThanEqual,
                           __FUNC_PREFIX_CORE)               // isgreaterequal
__SYCL_MAKE_CALL_ARG2_SAME(FOrdLessThan, __FUNC_PREFIX_CORE) // isless
__SYCL_MAKE_CALL_ARG2_SAME(FOrdLessThanEqual, __FUNC_PREFIX_CORE) // islessequal
__SYCL_MAKE_CALL_ARG2_SAME(LessOrGreater, __FUNC_PREFIX_CORE) // islessgreater
__SYCL_MAKE_CALL_ARG1(IsFinite, __FUNC_PREFIX_CORE)           // isfinite
__SYCL_MAKE_CALL_ARG1(IsInf, __FUNC_PREFIX_CORE)              // isinf
__SYCL_MAKE_CALL_ARG1(IsNan, __FUNC_PREFIX_CORE)              // isnan
__SYCL_MAKE_CALL_ARG1(IsNormal, __FUNC_PREFIX_CORE)           // isnormal
__SYCL_MAKE_CALL_ARG2_SAME(Ordered, __FUNC_PREFIX_CORE)       // isordered
__SYCL_MAKE_CALL_ARG2_SAME(Unordered, __FUNC_PREFIX_CORE)     // isunordered
__SYCL_MAKE_CALL_ARG1(SignBitSet, __FUNC_PREFIX_CORE)         // signbit
__SYCL_MAKE_CALL_ARG1(Any, __FUNC_PREFIX_CORE)                // any
__SYCL_MAKE_CALL_ARG1(All, __FUNC_PREFIX_CORE)                // all
__SYCL_MAKE_CALL_ARG3(bitselect, __FUNC_PREFIX_OCL)
__SYCL_MAKE_CALL_ARG3(select, __FUNC_PREFIX_OCL) // select
#ifndef __SYCL_DEVICE_ONLY__
} // namespace __host_std
} // __SYCL_INLINE_NAMESPACE(cl)
#endif

#undef __NOEXC
#undef __SYCL_MAKE_CALL_ARG1
#undef __SYCL_MAKE_CALL_ARG2
#undef __SYCL_MAKE_CALL_ARG2_SAME
#undef __SYCL_MAKE_CALL_ARG2_SAME_RESULT
#undef __SYCL_MAKE_CALL_ARG3
#undef __SYCL_PPCAT_NX
#undef __SYCL_PPCAT
#undef __FUNC_PREFIX_OCL
#undef __FUNC_PREFIX_CORE
#undef __SYCL_EXTERN_IT1
#undef __SYCL_EXTERN_IT2
#undef __SYCL_EXTERN_IT2_SAME
#undef __SYCL_EXTERN_IT3
