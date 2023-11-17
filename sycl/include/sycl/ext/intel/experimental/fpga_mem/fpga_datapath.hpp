//==----------- fpga_datapath.hpp - SYCL fpga_datapath extension -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp> // for address_space
#include <sycl/exception.hpp>     // for make_error_code

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

template <typename T>
class
// Annotation when object is instantiated in global scope
#ifdef __SYCL_DEVICE_ONLY__
    [[__sycl_detail__::add_ir_attributes_global_variable("sycl-datapath", "")]]
#endif
    fpga_datapath {
protected:
  T val
#ifdef __SYCL_DEVICE_ONLY__
      // Annotation when object is instantiated in function scope
      [[__sycl_detail__::add_ir_annotations_member("sycl-datapath", "")]]
#endif
      ;

public:
  // All the initialization
  // constexpr is used as a hint to the compiler to try and evaluate the
  // constructor at compile-time
  template <typename... S> constexpr fpga_datapath(S... args) : val{args...} {}

  fpga_datapath() = default;

  fpga_datapath(const fpga_datapath &) = default;
  fpga_datapath(fpga_datapath &&) = default;
  fpga_datapath &operator=(const fpga_datapath &) = default;
  fpga_datapath &operator=(fpga_datapath &&) = default;

  T &get() noexcept { return val; }

  constexpr const T &get() const noexcept { return val; }

  // Allows for implicit conversion from this to T
  operator T &() noexcept { return get(); }

  // Allows for implicit conversion from this to T
  constexpr operator const T &() const noexcept { return get(); }

  fpga_datapath &operator=(const T &newValue) noexcept {
    val = newValue;
    return *this;
  }

  // Note that there is no need for "fpga_datapath" to define member functions
  // for operators like "++", "[]", "->", comparison, etc. Instead, the type
  // "T" need only define these operators as non-member functions. Because
  // there is an implicit conversion from "fpga_datapath" to "T&".
};

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
