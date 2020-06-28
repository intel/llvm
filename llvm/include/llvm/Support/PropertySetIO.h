//==-- PropertySetIO.h -- models a sequence of property sets and their I/O -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Models a sequence of property sets and their input and output operations.
// TODO use Yaml as I/O engine.
// PropertyValue set format:
//   '['<PropertyValue set name>']'
//   <property name>=<property type>'|'<property value>
//   <property name>=<property type>'|'<property value>
//   ...
//   '['<PropertyValue set name>']'
//   <property name>=<property type>'|'<property value>
// where
//   <PropertyValue set name>, <property name> are strings
//   <property type> - string representation of the property type
//   <property value> - string representation of the property value.
//
// For example:
// [Staff/Ages]
// person1=1|20
// person2=1|25
// [Staff/Experience]
// person1=1|1
// person2=1|2
// person3=1|12
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PROPERTYSETIO_H
#define LLVM_SUPPORT_PROPERTYSETIO_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <istream>
#include <map>
#include <memory>
#include <string>

namespace llvm {
namespace util {

// Represents a property value. PropertyValue name is stored in the encompassing
// container.
class PropertyValue {
public:
  // Defines supported property types
  enum Type { first = 0, NONE = first, UINT32, last = UINT32 };

  // Translates C++ type to the corresponding type tag.
  template <typename T> static Type getTypeTag();

  // Casts from int value to a type tag.
  static Expected<Type> getTypeTag(int T) {
    if (T < first || T > last)
      return createStringError(std::error_code(), "bad property type " + T);
    return static_cast<Type>(T);
  }

  PropertyValue() = default;
  PropertyValue(Type T) : Ty(T) {}

  PropertyValue(uint32_t Val) : Ty(UINT32), Val({Val}) {}
  PropertyValue(const PropertyValue &P) = default;
  PropertyValue(PropertyValue &&P) = default;

  PropertyValue &operator=(PropertyValue &&P) = default;

  PropertyValue &operator=(const PropertyValue &P) = default;

  // get property value as unsigned 32-bit integer
  uint32_t asUint32() const {
    assert(Ty == UINT32);
    return Val.UInt32Val;
  }

  bool isValid() const { return getType() != NONE; }

  // set property value; the 'T' type must be convertible to a property type tag
  template <typename T> void set(T V) {
    assert(getTypeTag<T>() == Ty);
    getValueRef<T>() = V;
  }

  Type getType() const { return Ty; }

  size_t size() const {
    switch (Ty) {
    case UINT32:
      return sizeof(Val.UInt32Val);
    default:
      llvm_unreachable_internal("unsupported property type");
    }
  }

private:
  template <typename T> T &getValueRef();

  Type Ty = NONE;
  union {
    uint32_t UInt32Val;
  } Val;
};

std::ostream &operator<<(std::ostream &Out, const PropertyValue &V);

// A property set. Preserves insertion order when iterating elements.
using PropertySet = MapVector<StringRef, PropertyValue>;

// A "registry" of multiple property sets. Maps a property set name to its
// contents. Can be read/written.
class PropertySetRegistry {
public:
  using MapTy = MapVector<StringRef, PropertySet>;

  // Function for bulk addition of an entire property set under given category
  // (property set name).
  template <typename T>
  void add(StringRef Category, const std::map<StringRef, T> &Props) {
    assert(PropSetMap.find(Category) == PropSetMap.end() &&
           "category already added");
    auto &PropSet = PropSetMap[Category];

    for (const auto &Prop : Props)
      PropSet.insert(std::make_pair(Prop.first, PropertyValue(Prop.second)));
  }

  // Parses and creates a property set registry.
  static Expected<std::unique_ptr<PropertySetRegistry>>
  read(const MemoryBuffer *Buf);

  // Dumps a property set registry to a stream.
  void write(raw_ostream &Out) const;

  // Start iterator of all preperty sets in the registry.
  MapTy::const_iterator begin() const { return PropSetMap.begin(); }
  // End iterator of all preperty sets in the registry.
  MapTy::const_iterator end() const { return PropSetMap.end(); }

  // Retrieves a property set with given name.
  PropertySet &operator[](StringRef Name) { return PropSetMap[Name]; }
  // Constant access to the underlying map.
  const MapTy &getPropSets() const { return PropSetMap; }

private:
  MapTy PropSetMap;
};

} // namespace util
} // namespace llvm

#endif // #define LLVM_SUPPORT_PROPERTYSETIO_H
