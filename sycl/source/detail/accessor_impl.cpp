//==---------------- accessor_impl.cpp - SYCL standard source file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {

AccessorImplHost::~AccessorImplHost() {
  if (MBlockedCmd)
    detail::Scheduler::getInstance().releaseHostAccessor(this);
}
}
}
}

