//==-------- free_function_traits.hpp - SYCL free function queries --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/detail/kernel_desc.hpp>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

template <auto *Func, int Dims> struct is_nd_range_kernel {
  static constexpr bool value = false;
};

template <auto *Func> struct is_single_task_kernel {
  static constexpr bool value = false;
};

template <auto *Func, int Dims>
inline constexpr bool is_nd_range_kernel_v =
    is_nd_range_kernel<Func, Dims>::value;

template <auto *Func>
inline constexpr bool is_single_task_kernel_v =
    is_single_task_kernel<Func>::value;

template <auto *Func> struct is_kernel {
// During device compilation mode the compiler does not yet know
// what the kernels are named because that is exactly what its trying to
// figure out during this phase. Therefore, we set the is_kernel trait to true
// by default during device compilation in order to not get missing functions
// errors.
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr bool value = true;
#else
  static constexpr bool value = false;
#endif
};

template <auto *Func>
inline constexpr bool is_kernel_v = is_kernel<Func>::value;

namespace detail {
// A struct with special type is a struct type that contains special types
// passed as a paremeter to a free function kernel. It is decomposed into its
// consituents by the frontend which puts the relevant informaton about each of
// them into the struct below, namely offset, size and parameter kind for each
// one of them. The runtime then calls the addArg function to add each one of
// them as kernel arguments. The value bool is used to distinguish these structs
// from ordinary e.g standard layout structs.
template <typename T> struct is_struct_with_special_type {
  static constexpr bool value = false;
  static constexpr int offsets[] = {-1};
  static constexpr int sizes[] = {-1};
  static constexpr sycl::detail::kernel_param_kind_t kinds[] = {
      sycl::detail::kernel_param_kind_t::kind_invalid};
};

} // namespace detail

template <auto *Func> struct kernel_function_s {};

template <auto *Func> inline constexpr kernel_function_s<Func> kernel_function;
} // namespace ext::oneapi::experimental

template <typename T> struct is_device_copyable;

} // namespace _V1
} // namespace sycl

// CUDA-`<<<>>>`-style bare-name launch. SYCL_EXT_ONEAPI_KERNEL_FUNCTION(NAME,
// args...) lets a free function kernel be launched by naming it directly, with
// the compiler deducing its template arguments / resolving the overload from
// the launch arguments, while keeping the existing enqueue-function API
// unchanged:
//
//   nd_launch(q, range, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(iota, 3.14f, ptr));
//   single_task(q, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(store42, ptr));
//
// The macro expands into two comma-separated arguments of the enclosing enqueue
// call: the kernel_function<Func> non-type-template selector followed by the
// same launch arguments. __builtin_sycl_launch_kernel resolves NAME against the
// argument types (overload resolution / template argument deduction) and
// evaluates to a pointer to the chosen specialization, which fills the
// `auto *Func` template parameter. The launch arguments are then forwarded
// normally, so the emitted SPIR-V kernel is the real user function (no wrapper)
// and the launch path is exactly kernel_function<Func>.
//
// Non-deducible template parameters (those appearing in no function parameter)
// must still be spelled explicitly, e.g.
// SYCL_EXT_ONEAPI_KERNEL_FUNCTION((kern<float, 32>), args); this is a
// fundamental C++ limitation, matching CUDA.
//
// The leading unary '+' on the builtin call is load-bearing, not a typo. When
// SYCL_EXT_ONEAPI_KERNEL_FUNCTION is used inside a template with dependent
// arguments the builtin call is deferred with a BuiltinFn placeholder callee;
// deducing this `auto` non-type template argument classifies the argument
// expression. Wrapping the call in unary '+' makes that argument a
// UnaryOperator (a plain prvalue that is not classified through
// CallExpr::getCallReturnType) instead of the dependent CallExpr, which would
// otherwise assert in the front end. Unary '+' on a function pointer is the
// identity, so the deduced Func value is unchanged; this keeps the whole
// feature free of any compiler-side change for the dependent case.
//
// SYCL_EXT_ONEAPI_KERNEL_FUNCTION is the only part of the free function kernel
// API that needs compiler support: it relies on the
// __builtin_sycl_launch_kernel front-end builtin, which is provided only by the
// SYCL device compiler (Intel oneAPI DPC++ / clang -fsycl). Everything else in
// this extension (SYCL_EXT_ONEAPI_FUNCTION_PROPERTY, kernel_function<Func>,
// nd_launch, single_task) is header-only and works with any host compiler
// (MSVC, GCC,
// ...). The macro is therefore gated on __has_builtin:
//   * When the builtin is available, SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED
//     is defined to 1 and the macro performs the deduction.
//   * Otherwise the macro expands to a reference to an undeclared,
//     descriptively-named identifier so that *using* it is a clear compile
//     error at the call site (portable across GCC / Clang / MSVC), while merely
//     including this header remains valid everywhere.
#ifndef __has_builtin
// Older MSVC (< VS2019 16.1) lacks __has_builtin; treat as "not available".
#define __has_builtin(x) 0
#endif

#if __has_builtin(__builtin_sycl_launch_kernel)

#define SYCL_EXT_ONEAPI_KERNEL_FUNCTION_SUPPORTED 1

#define SYCL_EXT_ONEAPI_KERNEL_FUNCTION(NAME, ...)                             \
  ::sycl::ext::oneapi::experimental::kernel_function<                          \
      +__builtin_sycl_launch_kernel(NAME, ##__VA_ARGS__)>,                     \
      ##__VA_ARGS__

#else

// No compiler support (e.g. a plain MSVC/GCC host compile without the SYCL
// device compiler). Expand to an undeclared identifier whose name IS the
// diagnostic, so any conforming compiler reports a clear "use of undeclared
// identifier" error pointing at the call site. Including the header is fine;
// only using the macro fails.
#define SYCL_EXT_ONEAPI_KERNEL_FUNCTION(NAME, ...)                             \
  SYCL_EXT_ONEAPI_KERNEL_FUNCTION_requires_dpcpp_as_the_compiler

#endif
