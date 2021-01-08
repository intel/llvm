//==----------------------- kernel_desc.hpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This header file must not include any standard C++ header files.

#include <CL/sycl/detail/defines_elementary.hpp>
#include <CL/sycl/detail/export.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#ifndef __SYCL_DEVICE_ONLY__
#define _Bool bool
#endif

// kernel parameter kinds
enum class kernel_param_kind_t {
  kind_accessor,
  kind_std_layout, // standard layout object parameters
  kind_sampler,
  kind_pointer
};

// describes a kernel parameter
struct kernel_param_desc_t {
  // parameter kind
  kernel_param_kind_t kind;
  // kind == kind_std_layout
  //   parameter size in bytes (includes padding for structs)
  // kind == kind_accessor
  //   access target; possible access targets are defined in access/access.hpp
  int info;
  // offset of the captured value of the parameter in the lambda or function
  // object
  int offset;
};

// Translates specialization constant type to its name.
template <class Name> struct SpecConstantInfo {
  static constexpr const char *getName() { return ""; }
};

#ifndef __SYCL_UNNAMED_LAMBDA__
template <class KernelNameType> struct KernelInfo {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return ""; }
  static constexpr bool isESIMD() { return 0; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};
#else
template <char...> struct KernelInfoData {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int Idx) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return ""; }
  static constexpr bool isESIMD() { return 0; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

// C++14 like index_sequence and make_index_sequence
// not needed C++14 members (value_type, size) not implemented
template <class T, T...> struct integer_sequence {};
template <unsigned long long... I>
using index_sequence = integer_sequence<unsigned long long, I...>;
template <unsigned long long N>
using make_index_sequence =
    __make_integer_seq<integer_sequence, unsigned long long, N>;

template <typename T> struct KernelInfoImpl {
private:
  static constexpr auto n = __builtin_unique_stable_name(T);
  template <unsigned long long... I>
  static KernelInfoData<n[I]...> impl(index_sequence<I...>) {
    return {};
  }

public:
  using type = decltype(impl(make_index_sequence<__builtin_strlen(n)>{}));
};
template <typename T> using KernelInfo = typename KernelInfoImpl<T>::type;
#endif //__SYCL_UNNAMED_LAMBDA__

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
