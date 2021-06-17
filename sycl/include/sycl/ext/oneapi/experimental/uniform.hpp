//==------ uniform.hpp - SYCL uniform extension --------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// Implemenation of the SYCL_EXT_ONEAPI_UNIFORM extension.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/Uniform/Uniform.asciidoc
// ===--------------------------------------------------------------------=== //

#pragma once

namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {

// TODO:
// An implementation must issue a diagnostic if the
// sycl::ext::oneapi::experimental::uniform class template is instantiated with
// any of the following types:
// sycl::item, sycl::nd_item, sycl::h_item, sycl::group, sycl::sub_group and
// sycl::nd_range.
template <class T> class uniform {
public:
  explicit uniform(T x) noexcept : Val(x) {}
  operator T() const { return Val; }

private:
  T Val;
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
