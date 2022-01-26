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

// All properties are split here to dataless properties and properties with
// data. A dataless property is one which has no data stored in it. A property
// with data is one which has data stored in it and usually provides and access
// to it. For dataless property we just store a bool which indicates if a
// property is set or not. For properties with data we store a pointer to the
// base class because we do not know the size of such properties beforehand.

// List of all dataless properties' IDs
enum DataLessPropKind {
  BufferUseHostPtr = 0,
  ImageUseHostPtr = 1,
  QueueEnableProfiling = 2,
  InOrder = 3,
  NoInit = 4,
  BufferUsePinnedHostMemory = 5,
  UsePrimaryContext = 6,
  InitializeToIdentity = 7,
  UseDefaultStream = 8,
  DiscardEvents = 9,
  // Indicates the last known dataless property.
  LastKnownDataLessPropKind = 9,
  // Exceeding 32 may cause ABI breaking change on some of OSes.
  DataLessPropKindSize = 32
};

// List of all properties with data IDs
enum PropWithDataKind {
  BufferUseMutex = 0,
  BufferContextBound = 1,
  ImageUseMutex = 2,
  ImageContextBound = 3,
  BufferMemChannel = 4,
  PropWithDataKindSize = 5
};

// Base class for dataless properties, needed to check that the type of an
// object passed to the property_list is a property.
class DataLessPropertyBase {};

// Helper class for the dataless properties. Every such property is supposed
// to inherit from it. The ID template parameter should be one from
// DataLessPropKind.
template <int ID> class DataLessProperty : DataLessPropertyBase {
public:
  static constexpr int getKind() { return ID; }
};

// Base class for properties with data, needed to check that the type of an
// object passed to the property_list is a property and for checking if two
// properties with data are of the same type.
class PropertyWithDataBase {
public:
  PropertyWithDataBase(int ID) : MID(ID) {}
  bool isSame(int ID) const { return ID == MID; }
  virtual ~PropertyWithDataBase() = default;

private:
  int MID = -1;
};

// Helper class for the properties with data. Every such property is supposed
// to inherit from it. The ID template parameter should be one from
// PropWithDataKind.
template <int ID> class PropertyWithData : public PropertyWithDataBase {
public:
  PropertyWithData() : PropertyWithDataBase(ID) {}
  static int getKind() { return ID; }
};

} // namespace detail

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
