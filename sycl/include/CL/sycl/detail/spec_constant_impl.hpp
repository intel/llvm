//==-- spec_constant_impl.hpp - SYCL RT model for specialization constants -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

#include <iostream>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// Represents a specialization constant in SYCL runtime.
class spec_constant_impl {
public:
  spec_constant_impl(unsigned int ID) : ID(ID), Size(0), Bytes{0} {}

  spec_constant_impl(unsigned int ID, size_t Size, const void *Val) : ID(ID) {
    set(Size, Val);
  }

  void set(size_t Size, const void *Val);

  unsigned int getID() const { return ID; }
  size_t getSize() const { return Size; }
  const unsigned char *getValuePtr() const { return Bytes; }
  bool isSet() const { return Size != 0; }

private:
  unsigned int ID; // specialization constant's ID (equals to SPIRV ID)
  size_t Size;     // size of its value
  // TODO invent more flexible approach to support values of arbitrary type:
  unsigned char Bytes[8]; // memory to hold the value bytes
};

std::ostream &operator<<(std::ostream &Out, const spec_constant_impl &V);

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
