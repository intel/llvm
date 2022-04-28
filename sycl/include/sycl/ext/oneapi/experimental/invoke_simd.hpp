//==------ invoke_simd.hpp - SYCL invoke_simd extension --*- C++ -*---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// Implemenation of the sycl_ext_oneapi_invoke_simd extension.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc
// ===--------------------------------------------------------------------=== //

#pragma once

// SYCL extension macro definition as required by the SYCL specification.
// 1 - Initial extension version. Base features are supported.
#define SYCL_EXT_ONEAPI_INVOKE_SIMD 1

#include <sycl/ext/oneapi/experimental/uniform.hpp>

#include <CL/sycl/sub_group.hpp>
#include <std/experimental/simd.hpp>
#include <sycl/detail/boost/mp11.hpp>

#include <functional>

// TODOs:
// * (a) TODO bool translation in spmd2simd.
// * (b) TODO enforce constness of a functor/lambda's () operator
// * (c) TODO support lambdas and functors in BE

/// Middle End - to - Back End interface to invoke explicit SIMD functions from
/// SPMD SYCL context. Must not be used by user code. BEs are expected to
/// recognize this intrinsic and transform the intrinsic call with a direct call
/// to the SIMD target, as well as process SPMD arguments in the way described
/// in the specification for `invoke_simd`.
/// @tparam SpmdRet the return type. Can be `uniform<T>`.
/// @tparam SimdCallee the type of the SIMD callee function (the "target"). Must
///  be a function type (not lambda or functor).
/// @tparam SpmdArgs The original SPMD arguments passed to the invoke_simd.
template <bool IsFunc, class SpmdRet, class SimdCallee, class... SpmdArgs,
          class = std::enable_if_t<!IsFunc>>
SYCL_EXTERNAL __regcall SpmdRet
__builtin_invoke_simd(SimdCallee target, const void *obj, SpmdArgs... args)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  // __builtin_invoke_simd is not supported on the host device yet
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "__builtin_invoke_simd is not supported on host");
}
#endif // __SYCL_DEVICE_ONLY__

template <bool IsFunc, class SpmdRet, class SimdCallee, class... SpmdArgs,
          class = std::enable_if_t<IsFunc>>
SYCL_EXTERNAL __regcall SpmdRet __builtin_invoke_simd(SimdCallee target,
                                                      SpmdArgs... args)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  // __builtin_invoke_simd is not supported on the host device yet
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "__builtin_invoke_simd is not supported on host");
}
#endif // __SYCL_DEVICE_ONLY__

namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

// --- Basic definitions prescribed by the spec.
namespace simd_abi {
// "Fixed-size simd width of N" ABI based on clang vectors - used as the ABI for
// SIMD objects this implementation of invoke_simd spec is based on.
template <class T, int N>
using native_fixed_size = typename std::experimental::__simd_abi<
    std::experimental::_StorageKind::_VecExt, N>;
} // namespace simd_abi

// The SIMD object type, which is the generic std::experimental::simd type with
// the native fixed size ABI.
template <class T, int N>
using simd = std::experimental::simd<T, simd_abi::native_fixed_size<T, N>>;

// The SIMD mask object type.
template <class T, int N>
using simd_mask =
    std::experimental::simd_mask<T, simd_abi::native_fixed_size<T, N>>;

// --- Helpers
namespace detail {

namespace __MP11_NS = sycl::detail::boost::mp11;

// This structure performs the SPMD-to-SIMD parameter type conversion as defined
// by the spec.
template <class T, int N, class = void> struct spmd2simd;
// * `uniform<T>` converts to `T`
template <class T, int N> struct spmd2simd<uniform<T>, N> {
  using type = T;
};
// * tuple of types converts to tuple of converted tuple element types.
template <class... T, int N> struct spmd2simd<std::tuple<T...>, N> {
  using type = std::tuple<typename spmd2simd<T, N>::type...>;
};
// * arithmetic type converts to a simd vector with this element type and the
//   width equal to caller's subgroup size and passed as the `N` template
//   argument.
template <class T, int N>
struct spmd2simd<T, N, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = simd<T, N>;
};

// This structure performs the SIMD-to-SPMD return type conversion as defined
// by the spec.
template <class, class = void> struct simd2spmd;
// * `uniform<T>` stays the same
template <class T> struct simd2spmd<uniform<T>> {
  using type = uniform<T>;
};
// * `simd<T, N>` converts to T
template <class T, int N> struct simd2spmd<simd<T, N>> {
  using type = T;
};
// * tuple of types converts to tuple of converted tuple element types.
template <class... T> struct simd2spmd<std::tuple<T...>> {
  using type = std::tuple<typename simd2spmd<T>::type...>;
};
// * arithmetic type T converts to `uniform<T>`
template <class T>
struct simd2spmd<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = uniform<T>;
};

// Check if given type is uniform.
template <class T> struct is_uniform_type : std::false_type {};
template <class T> struct is_uniform_type<uniform<T>> : std::true_type {
  using type = T;
};

// Check if given type is simd or simd_mask.
template <class T> struct is_simd_or_mask_type : std::false_type {};
template <class T, int N>
struct is_simd_or_mask_type<simd<T, N>> : std::true_type {};
template <class T, int N>
struct is_simd_or_mask_type<simd_mask<T, N>> : std::true_type {};

// Checks if the return value type and the types of arguments of given
// SimdCallable are all uniform.
template <class SimdCallable, class... SpmdArgs> struct has_uniform_signature {
  constexpr operator bool() {
    using ArgTypeList = __MP11_NS::mp_list<SpmdArgs...>;

    if constexpr (__MP11_NS::mp_all_of<ArgTypeList, is_uniform_type>::value) {
      using SimdRet = std::invoke_result_t<SimdCallable, SpmdArgs...>;
      return is_uniform_type<SimdRet>::value ||
             !is_simd_or_mask_type<SimdRet>::value;
    } else {
      return false;
    }
  }
};

// "Unwraps" a value of the `uniform` type (used before passing to SPMD
//  arguments to the __builtin_invoke_simd):
// - the case when there is nothing to unwrap
template <typename T> struct unwrap_uniform {
  static auto impl(T val) { return val; }
};

// - the real unwrapping case
template <typename T> struct unwrap_uniform<uniform<T>> {
  static T impl(uniform<T> val) { return val; }
};

// Deduces subgroup size of the caller based on given SIMD callable and
// corresponding SPMD arguments it is being invoke with via invoke_simd.
// Basically, for each supported subgroup size, this meta-function finds out if
// the callable can be invoked by C++ rules given the SPMD arguments transformed
// as prescribed by the spec assuming this subgroup size. One and only one
// subgroup size should conform.
template <class SimdCallable, class... SpmdArgs> struct sg_size {
  template <class N>
  using IsInvocableSgSize = __MP11_NS::mp_bool<std::is_invocable_v<
      SimdCallable, typename spmd2simd<SpmdArgs, N::value>::type...>>;

  constexpr operator int() {
    using SupportedSgSizes = __MP11_NS::mp_list_c<int, 1, 2, 4, 8, 16, 32>;
    using InvocableSgSizes =
        __MP11_NS::mp_copy_if<SupportedSgSizes, IsInvocableSgSize>;
    static_assert((__MP11_NS::mp_size<InvocableSgSizes>::value == 1) &&
                  "no or multiple invoke_simd targets found");
    return __MP11_NS::mp_front<InvocableSgSizes>::value;
  }
};

// Determine the return type of a SIMD callable.
template <int N, class SimdCallable, class... SpmdArgs>
using SimdRetType =
    std::invoke_result_t<SimdCallable,
                         typename spmd2simd<SpmdArgs, N>::type...>;
// Determine the return type of an invoke_simd based on the return type of a
// SIMD callable.
template <int N, class SimdCallable, class... SpmdArgs>
using SpmdRetType =
    typename simd2spmd<SimdRetType<N, SimdCallable, SpmdArgs...>>::type;

template <class SimdCallable, class... SpmdArgs>
static constexpr int get_sg_size() {
  if constexpr (has_uniform_signature<SimdCallable, SpmdArgs...>()) {
    return 0; // subgroup size does not matter then
  } else {
    return sg_size<SimdCallable, SpmdArgs...>();
  }
}

// This function is a wrapper around a call to a functor with field or a lambda
// with captures. Note __regcall - this is needed for efficient argument
// forwarding.
template <int N, class Callable, class... T>
__regcall detail::SimdRetType<N, Callable, T...>
simd_call_helper(const void *obj_ptr,
                 typename detail::spmd2simd<T, N>::type... simd_args) {
  auto f =
      *reinterpret_cast<const std::remove_reference_t<Callable> *>(obj_ptr);
  return f(simd_args...);
};

#ifdef _GLIBCXX_RELEASE
#if _GLIBCXX_RELEASE < 10
#define __INVOKE_SIMD_USE_STD_IS_FUNCTION_WA
#endif // _GLIBCXX_RELEASE < 10
#endif // _GLIBCXX_RELEASE

#ifdef __INVOKE_SIMD_USE_STD_IS_FUNCTION_WA
// TODO This is a workaround for libstdc++ version 9 buggy behavior which
// returns false in the code below. Version 10 works fine. Once required
// minimum libstdc++ version is bumped to 10, this w/a should be removed.
//   template <class F> bool foo(F &&f) {
//     return std::is_function_v<std::remove_reference_t<F>>;
//   }
// where F is a function type with __regcall.
template <class F> struct is_regcall_function_ptr_or_ref : std::false_type {};

template <class Ret, class... Args>
struct is_regcall_function_ptr_or_ref<Ret(__regcall &)(Args...)>
    : std::true_type {};

template <class Ret, class... Args>
struct is_regcall_function_ptr_or_ref<Ret(__regcall *)(Args...)>
    : std::true_type {};

template <class Ret, class... Args>
struct is_regcall_function_ptr_or_ref<Ret(__regcall *&)(Args...)>
    : std::true_type {};

template <class F>
static constexpr bool is_regcall_function_ptr_or_ref_v =
    is_regcall_function_ptr_or_ref<F>::value;
#endif // __INVOKE_SIMD_USE_STD_IS_FUNCTION_WA

template <class Callable>
static constexpr bool is_function_ptr_or_ref_v =
    std::is_function_v<std::remove_pointer_t<std::remove_reference_t<Callable>>>
#ifdef __INVOKE_SIMD_USE_STD_IS_FUNCTION_WA
    || is_regcall_function_ptr_or_ref_v<Callable>
#endif // __INVOKE_SIMD_USE_STD_IS_FUNCTION_WA
    ;
} // namespace detail

// --- The main API

/// The invoke_simd free function invokes a SIMD function using all work-items
/// in a sub_group. The invoke_simd interface marshals data between the SPMD
/// context of the calling kernel and the SIMD context of the callee, converting
/// arguments and return values between scalar and SIMD types as appropriate.
///
/// @param sg the subgroup simd function is invoked from
/// @param f represents the invoked simd function.
///   Must be a C++ callable that can be invoked with the same number of
///   arguments specified in the args parameter pack. Callable may be a function
///   object, a lambda, or a function pointer (if the device supports
///   SPV_INTEL_function_pointers). Callable must be an immutable callable with
///   the same type and state for all work-items in the sub-group, otherwise
///   behavior is undefined.
/// @param args SPMD parameters to the invoked function, which undergo
///   transformation before actual passing to the simd function, as described in
///   the specification.
// TODO works only for functions now, enable for other callables.
template <class Callable, class... T>
__attribute__((always_inline)) auto invoke_simd(sycl::sub_group sg,
                                                Callable &&f, T... args) {
  // If the invoke_simd call site is fully uniform, then it does not matter
  // what the subgroup size is and arguments don't need widening and return
  // value does not need shrinking by this library or SPMD compiler, so 0
  // is fine in this case.
  constexpr int N = detail::get_sg_size<Callable, T...>();
  using RetSpmd = detail::SpmdRetType<N, Callable, T...>;
  constexpr bool is_function = detail::is_function_ptr_or_ref_v<Callable>;

  if constexpr (is_function) {
    return __builtin_invoke_simd<is_function, RetSpmd>(
        f, detail::unwrap_uniform<T>::impl(args)...);
  } else {
    // TODO support functors and lambdas which are handled in this branch.
    // The limiting factor for now is that the LLVMIR data flow analysis
    // implemented in LowerInvokeSimd.cpp which, finds actual invoke_simd
    // target function, can't handle this case yet.
    return __builtin_invoke_simd<is_function, RetSpmd>(
        detail::simd_call_helper<N, Callable, T...>, &f,
        detail::unwrap_uniform<T>::impl(args)...);
  }
// TODO Temporary macro and assert to enable API compilation testing.
// LowerInvokeSimd.cpp does not support this case yet.
#ifndef __INVOKE_SIMD_ENABLE_ALL_CALLABLES
  static_assert(is_function &&
                "invoke_simd does not support functors or lambdas yet");
#endif // __INVOKE_SIMD_ENABLE_ALL_CALLABLES
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
