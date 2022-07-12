//==-- spec_constant_impl.cpp - SYCL RT model for specialization constants -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/spec_constant_impl.hpp>

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/pi.h>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/exception.hpp>

#include <cstdio>
#include <cstring>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

void spec_constant_impl::set(size_t Size, const void *Val) {
  if (0 == Size)
    throw sycl::runtime_error("invalid spec constant size",
                              PI_ERROR_INVALID_VALUE);
  auto *BytePtr = reinterpret_cast<const char *>(Val);
  this->Bytes.assign(BytePtr, BytePtr + Size);
}

void stableSerializeSpecConstRegistry(const SpecConstRegistryT &Reg,
                                      SerializedObj &Dst) {
  for (const auto &E : Reg) {
    Dst.insert(Dst.end(), E.first.begin(), E.first.end());
    const spec_constant_impl &SC = E.second;
    Dst.insert(Dst.end(), SC.getValuePtr(), SC.getValuePtr() + SC.getSize());
  }
}
spec_constant_impl::operator std::string() const {
  std::string Out;
  Out += "spec_constant_impl";
  Out += " { Size=";
  Out += std::to_string(this->getSize()) +
         " IsSet=" + (this->isSet() ? "true" : "false") + " Val=[";

  char tempString[50];
  for (unsigned I = 0; I < this->getSize(); ++I) {
    Out += (I == 0 ? "" : " ");
    snprintf(tempString, 50, "%x",
             static_cast<int>(*(this->getValuePtr() + I)));
    Out += tempString;
  }
  Out += "]";
  return Out;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
