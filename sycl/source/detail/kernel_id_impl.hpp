//==------- kernel_id_impl.hpp - SYCL kernel_id_impl -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// Used for sorting vector of kernel_id's
struct LessByNameComp {
  bool operator()(const sycl::kernel_id &LHS,
                  const sycl::kernel_id &RHS) const {
    return std::strcmp(LHS.get_name(), RHS.get_name()) < 0;
  }
};

struct EqualByNameComp {
  bool operator()(const sycl::kernel_id &LHS,
                  const sycl::kernel_id &RHS) const {
    return strcmp(LHS.get_name(), RHS.get_name()) == 0;
  }
};

// The class is impl counterpart for sycl::kernel_id which represent a kernel
// identificator
class kernel_id_impl {
public:
  kernel_id_impl(std::string Name) : MName(std::move(Name)) {}
  kernel_id_impl(){};
  const char *get_name() { return MName.data(); }

  const std::string &get_name_string() { return MName; }

private:
  std::string MName;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
