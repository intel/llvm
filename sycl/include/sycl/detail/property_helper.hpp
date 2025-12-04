//==--------- property_helper.hpp --- SYCL property helper -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
inline namespace _V1 {

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
  DeviceReadOnly = 10,
  QueuePriorityNormal = 11,
  QueuePriorityLow = 12,
  QueuePriorityHigh = 13,
  GraphNoCycleCheck = 14,
  QueueSubmissionBatched = 15,
  QueueSubmissionImmediate = 16,
  GraphAssumeDataOutlivesBuffer = 17,
  GraphAssumeBufferOutlivesGraph = 18,
  GraphDependOnAllLeaves = 19,
  GraphUpdatable = 20,
  GraphEnableProfiling = 21,
  // Indicates the last known dataless property.
  LastKnownDataLessPropKind = 22,
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
  AccPropBufferLocation = 5,
  QueueComputeIndex = 6,
  GraphNodeDependencies = 7,
  PropWithDataKindSize = 8
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
  int getKind() const { return MID; }

private:
  int MID = -1;
};

// Helper class for the properties with data. Every such property is supposed
// to inherit from it. The ID template parameter should be one from
// PropWithDataKind.
template <int ID> class PropertyWithData : public PropertyWithDataBase {
public:
  PropertyWithData() : PropertyWithDataBase(ID) {}
  static constexpr int getKind() { return ID; }
};

} // namespace detail

} // namespace _V1
} // namespace sycl
