//==-- spec_constant_impl.cpp - SYCL RT model for specialization constants -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/spec_constant_impl.hpp>

#include <CL/sycl/detail/pi.h>
#include <CL/sycl/exception.hpp>

#include <cstring>
#include <iostream>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

void spec_constant_impl::set(size_t Size, const void *Val) {
  if ((Size > sizeof(Bytes)) || (Size == 0))
    throw sycl::runtime_error("invalid spec constant size", PI_INVALID_VALUE);
  this->Size = Size;
  std::memcpy(Bytes, Val, Size);
}

std::ostream &operator<<(std::ostream &Out, const spec_constant_impl &V) {
  Out << "spec_constant_impl" << V.getID() << "{ Size=" << V.getSize()
      << " IsSet=" << V.isSet() << " Val=[";
  std::ios_base::fmtflags FlagsSav = Out.flags();
  Out << std::hex;
  for (unsigned I = 0; I < V.getSize(); ++I) {
    Out << (I == 0 ? "" : " ") << static_cast<int>(*(V.getValuePtr() + I));
  }
  Out << "]" << FlagsSav;
  return Out;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
