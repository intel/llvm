// UNSUPPORTED: preview-mode
// RUN: %{build} -o %t.out

//==---------- reference_count.cpp - Deprecated info descriptors test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

using namespace sycl;

template <typename SyclObject, typename Param>
using GetInfoMemberT = uint32_t (SyclObject::*)() const;

template <typename SyclObject, typename Param>
GetInfoMemberT<SyclObject, Param> check_get_info() {
  static_assert(std::is_same_v<typename Param::return_type, uint32_t>);
  return &SyclObject::template get_info<Param>;
}

int main() {
  auto ContextRefCount =
      check_get_info<context, info::context::reference_count>();
  auto DeviceRefCount = check_get_info<device, info::device::reference_count>();
  auto EventRefCount = check_get_info<event, info::event::reference_count>();
  auto KernelRefCount = check_get_info<kernel, info::kernel::reference_count>();
  auto QueueRefCount = check_get_info<queue, info::queue::reference_count>();

  (void)ContextRefCount;
  (void)DeviceRefCount;
  (void)EventRefCount;
  (void)KernelRefCount;
  (void)QueueRefCount;
  return 0;
}
