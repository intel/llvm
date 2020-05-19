//==------------ interop_handle.cpp --- SYCL interop handle ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/interop_handle.hpp>

#include <algorithm>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

cl_mem interop_handle::getMemImpl(detail::Requirement *Req) const {
  auto Iter = std::find_if(std::begin(MMemObjs), std::end(MMemObjs),
                           [=](ReqToMem Elem) { return (Elem.first == Req); });

  if (Iter == std::end(MMemObjs))
    throw("Invalid memory object used inside interop");

  return detail::pi::cast<cl_mem>(Iter->second);
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
