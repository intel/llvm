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
#include <type_traits>

namespace llvm {
namespace util {

// Represents a property value. PropertyValue name is stored in the encompassing
// container.
class PropertyValue {
public:
  // Type of the size of the value. Value size gets serialized along with the
  // value data in some cases for later reading at runtime, so size_t is not
  // suitable as its size varies.
  using SizeTy = uint64_t;
  using byte = uint8_t;

  // Defines supported property types
  enum Type { first = 0, NONE = first, UINT32, BYTE_ARRAY, last = BYTE_ARRAY };

  // Translates C++ type to the corresponding type tag.
  template <typename T> static Type getTypeTag();

  // Casts from int value to a type tag.
  static Expected<Type> getTypeTag(int T) {
    if (T < first || T > last)
      return createStringError(std::error_code(), "bad property type ", T);
    return static_cast<Type>(T);
  }

  ~PropertyValue() {
    if ((getType() == BYTE_ARRAY) && Val.ByteArrayVal)
      delete[] Val.ByteArrayVal;
  }

  PropertyValue() = default;
  PropertyValue(Type T) : Ty(T) {}

  PropertyValue(uint32_t Val) : Ty(UINT32), Val({Val}) {}
  PropertyValue(const byte *Data, SizeTy DataBitSize);
  template <typename T>
  PropertyValue(const std::vector<T> &Data)
      : PropertyValue(reinterpret_cast<const byte *>(Data.data()),
                      Data.size() * sizeof(T) * /* bits in one byte */ 8) {}
  PropertyValue(const PropertyValue &P);
  PropertyValue(PropertyValue &&P);

  PropertyValue &operator=(PropertyValue &&P);

  PropertyValue &operator=(const PropertyValue &P);

  // get property value as unsigned 32-bit integer
  uint32_t asUint32() const {
    if (Ty != UINT32)
      llvm_unreachable("must be UINT32 value");
    return Val.UInt32Val;
  }

  // Get raw data size in bits.
  SizeTy getByteArraySizeInBits() const {
    if (Ty != BYTE_ARRAY)
      llvm_unreachable("must be BYTE_ARRAY value");
    SizeTy Res = 0;

    for (size_t I = 0; I < sizeof(SizeTy); ++I)
      Res |= (SizeTy)Val.ByteArrayVal[I] << (8 * I);
    return Res;
  }

  // Get byte array data size in bytes.
  SizeTy getByteArraySize() const {
    SizeTy SizeInBits = getByteArraySizeInBits();
    constexpr unsigned int MASK = 0x7;
    return ((SizeInBits + MASK) & ~MASK) / 8;
  }

  // Get byte array data size in bytes, including the leading bytes encoding the
  // size.
  SizeTy getRawByteArraySize() const {
    return getByteArraySize() + sizeof(SizeTy);
  }

  // Get byte array data including the leading bytes encoding the size.
  const byte *asRawByteArray() const {
    if (Ty != BYTE_ARRAY)
      llvm_unreachable("must be BYTE_ARRAY value");
    return Val.ByteArrayVal;
  }

  // Get byte array data excluding the leading bytes encoding the size.
  const byte *asByteArray() const {
    if (Ty != BYTE_ARRAY)
      llvm_unreachable("must be BYTE_ARRAY value");
    return Val.ByteArrayVal + sizeof(SizeTy);
  }

  bool isValid() const { return getType() != NONE; }

  // set property value; the 'T' type must be convertible to a property type tag
  template <typename T> void set(T V) {
    if (getTypeTag<T>() != Ty)
      llvm_unreachable("invalid type tag for this operation");
    getValueRef<T>() = V;
  }

  Type getType() const { return Ty; }

  SizeTy size() const {
    switch (Ty) {
    case UINT32:
      return sizeof(Val.UInt32Val);
    case BYTE_ARRAY:
      return getRawByteArraySize();
    default:
      llvm_unreachable_internal("unsupported property type");
    }
  }

private:
  template <typename T> T &getValueRef();
  void copy(const PropertyValue &P);

  Type Ty = NONE;
  // TODO: replace this union with std::variant when uplifting to C++17
  union {
    uint32_t UInt32Val;
    // Holds first sizeof(size_t) bytes of size followed by actual raw data.
    byte *ByteArrayVal;
  } Val;
};

// A property set. Preserves insertion order when iterating elements.
using PropertySet = MapVector<StringRef, PropertyValue>;

// A "registry" of multiple property sets. Maps a property set name to its
// contents. Can be read/written.
class PropertySetRegistry {
public:
  using MapTy = MapVector<StringRef, PropertySet>;

  // Specific property category names used by tools.
  static constexpr char SYCL_SPECIALIZATION_CONSTANTS[] =
      "SYCL/specialization constants";
  static constexpr char SYCL_SPEC_CONSTANTS_DEFAULT_VALUES[] =
      "SYCL/specialization constants default values";
  static constexpr char SYCL_DEVICELIB_REQ_MASK[] = "SYCL/devicelib req mask";
  static constexpr char SYCL_KERNEL_PARAM_OPT_INFO[] = "SYCL/kernel param opt";
  static constexpr char SYCL_MISC_PROP[] = "SYCL/misc properties";
  static constexpr char SYCL_ASSERT_USED[] = "SYCL/assert used";

  // Function for bulk addition of an entire property set under given category
  // (property set name).
  template <typename MapTy> void add(StringRef Category, const MapTy &Props) {
    using KeyTy = typename MapTy::value_type::first_type;
    static_assert(std::is_same<typename std::remove_const<KeyTy>::type,
                               llvm::StringRef>::value,
                  "wrong key type");

    assert(PropSetMap.find(Category) == PropSetMap.end() &&
           "category already added");
    auto &PropSet = PropSetMap[Category];

    for (const auto &Prop : Props)
      PropSet.insert({Prop.first, PropertyValue(Prop.second)});
  }

  // Function to add a property to a given category (property set name).
  template <typename T>
  void add(StringRef Category, StringRef PropName, const T &PropVal) {
    auto &PropSet = PropSetMap[Category];
    PropSet.insert({PropName, PropertyValue(PropVal)});
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
