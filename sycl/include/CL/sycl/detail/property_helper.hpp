//==--------- property_helper.hpp --- SYCL property helper -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

namespace detail {

// All properties are split here to simple and complex. A simple property is one
// which has no data stored in it. A complex property is one which has data
// stored in it and usually provides and access to it.
// For simple property we just store a bool which indicates if a property is
// set or not. For complex properties we store a pointer to the base class
// because we do not know the size of such properties beforehand.

// List of all simple properties' IDs
enum SimplePropKind {
  BufferUseHostPtr = 0,
  ImageUseHostPtr,
  QueueEnableProfiling,
  InOrder,
  NoInit,
  BufferUsePinnedHostMemory,
  UsePrimaryContext,
  SimplePropKindSize
};

// List of all complex properties' IDs
enum ComplexPropKind {
  BufferUseMutex = 0,
  BufferContextBound,
  ImageUseMutex,
  ImageContextBound,
  ComplexPropKindSize
};

// Base class for simple properties, needed to check that the type of an object
// passed to the property_list is a property.
class SimplePropertyBase {};

// Helper class for the simple properties. Every simple property is supposed to
// inherit from it. The ID template parameter should be one from SimplePropKind.
template <int ID> class SimpleProperty : SimplePropertyBase {
public:
  static constexpr int getKind() { return ID; }
};

// Base class for complex properties, needed to check that the type of an object
// passed to the property_list is a property and for checking if two complex
// properties are of the same type.
class ComplexPropertyBase {
public:
  ComplexPropertyBase(int ID) : MID(ID) {}
  bool isSame(int ID) const { return ID == MID; }
  virtual ~ComplexPropertyBase() = default;

private:
  int MID = -1;
};

// Helper class for the complex properties. Every complex property is supposed
// to inherit from it. The ID template parameter should be one from
// ComplexPropKind.
template <int ID> class ComplexProperty : public ComplexPropertyBase {
public:
  ComplexProperty() : ComplexPropertyBase(ID) {}
  static int getKind() { return ID; }
};

} // namespace detail

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
