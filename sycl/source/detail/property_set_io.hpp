//==-- PropertySetIO.h -- models a sequence of property sets and their I/O -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Adjusted copy of llvm/include/llvm/Support/PropertySetIO.h.
// TODO: Remove once we can consistently link the SYCL runtime library with
// LLVMSupport.

#pragma once

#include "detail/base64.hpp"
#include "sycl/exception.hpp"

#include <unordered_map>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Helper function for converting a string_view to an integer. Allows only
// integer values and the empty string (interpreted as 0).
template <typename IntT>
static IntT stringViewToInt(const std::string_view &SV) {
  static_assert(std::is_integral_v<IntT>);

  IntT Result = static_cast<IntT>(0);
  if (SV.empty())
    return Result;

  bool Negate = std::is_signed_v<IntT> && SV[0] == '-';

  for (size_t I = static_cast<size_t>(Negate); I < SV.size(); ++I) {
    const char CurrentC = SV[I];
    if (CurrentC < '0' || CurrentC > '9')
      throw sycl::exception(make_error_code(errc::invalid),
                            "Invalid integer numeral: " +
                                std::string{CurrentC});
    Result *= static_cast<IntT>(10);
    Result += static_cast<IntT>(CurrentC - '0');
  }
  return Negate ? -Result : Result;
}

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
  template <typename T> static Type getTypeTag() {
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, byte *>);
    if constexpr (std::is_same_v<T, uint32_t>) {
      return UINT32;
    } else {
      return BYTE_ARRAY;
    }
  }

  // Casts from int value to a type tag.
  static Type getTypeTag(int T) {
    if (T < first || T > last)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Bad property type.");
    return static_cast<Type>(T);
  }

  ~PropertyValue() {
    if ((getType() == BYTE_ARRAY) && Val.ByteArrayVal)
      delete[] Val.ByteArrayVal;
  }

  PropertyValue() = default;
  PropertyValue(Type T) : Ty(T) {}

  PropertyValue(uint32_t Val) : Ty(UINT32), Val({Val}) {}
  PropertyValue(const byte *Data, SizeTy DataBitSize) {
    constexpr int ByteSizeInBits = 8;
    Ty = BYTE_ARRAY;
    SizeTy DataSize = (DataBitSize + (ByteSizeInBits - 1)) / ByteSizeInBits;
    constexpr size_t SizeFieldSize = sizeof(SizeTy);

    // Allocate space for size and data.
    Val.ByteArrayVal = new byte[SizeFieldSize + DataSize];

    // Write the size into first bytes.
    for (size_t I = 0; I < SizeFieldSize; ++I) {
      Val.ByteArrayVal[I] = (byte)DataBitSize;
      DataBitSize >>= ByteSizeInBits;
    }
    // Append data.
    std::memcpy(Val.ByteArrayVal + SizeFieldSize, Data, DataSize);
  }
  template <typename C, typename T = typename C::value_type>
  PropertyValue(const C &Data)
      : PropertyValue(reinterpret_cast<const byte *>(Data.data()),
                      Data.size() * sizeof(T) * /* bits in one byte */ 8) {}
  PropertyValue(const std::string_view Str)
      : PropertyValue(reinterpret_cast<const byte *>(Str.data()),
                      Str.size() * sizeof(char) * /* bits in one byte */ 8) {}
  PropertyValue(const PropertyValue &P) { *this = P; }
  PropertyValue(PropertyValue &&P) { *this = std::move(P); }

  PropertyValue &operator=(PropertyValue &&P) {
    copy(P);

    if (P.getType() == BYTE_ARRAY)
      P.Val.ByteArrayVal = nullptr;
    P.Ty = NONE;
    return *this;
  }

  PropertyValue &operator=(const PropertyValue &P) {
    if (P.getType() == BYTE_ARRAY)
      *this = PropertyValue(P.asByteArray(), P.getByteArraySizeInBits());
    else
      copy(P);
    return *this;
  }

  // get property value as unsigned 32-bit integer
  uint32_t asUint32() const {
    if (Ty != UINT32)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Must be UINT32 value.");
    return Val.UInt32Val;
  }

  // Get raw data size in bits.
  SizeTy getByteArraySizeInBits() const {
    if (Ty != BYTE_ARRAY)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Must be BYTE_ARRAY value.");
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
      throw sycl::exception(make_error_code(errc::invalid),
                            "Must be BYTE_ARRAY value.");
    return Val.ByteArrayVal;
  }

  // Get byte array data excluding the leading bytes encoding the size.
  const byte *asByteArray() const {
    if (Ty != BYTE_ARRAY)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Must be BYTE_ARRAY value.");
    return Val.ByteArrayVal + sizeof(SizeTy);
  }

  bool isValid() const { return getType() != NONE; }

  // set property value; the 'T' type must be convertible to a property type tag
  template <typename T> void set(T V) {
    if (getTypeTag<T>() != Ty)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Invalid type tag for this operation.");
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
      throw sycl::exception(make_error_code(errc::invalid),
                            "Unsupported property type.");
    }
  }

  const char *data() const {
    switch (Ty) {
    case UINT32:
      return reinterpret_cast<const char *>(&Val.UInt32Val);
    case BYTE_ARRAY:
      return reinterpret_cast<const char *>(Val.ByteArrayVal);
    default:
      throw sycl::exception(make_error_code(errc::invalid),
                            "Unsupported property type.");
    }
  }

private:
  template <typename T> T &getValueRef() {
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, byte *>);
    if constexpr (std::is_same_v<T, uint32_t>) {
      return Val.UInt32Val;
    } else {
      return Val.ByteArrayVal;
    }
  }

  void copy(const PropertyValue &P) {
    Ty = P.Ty;
    Val = P.Val;
  }

  Type Ty = NONE;
  // TODO: replace this union with std::variant when uplifting to C++17
  union {
    uint32_t UInt32Val;
    // Holds first sizeof(size_t) bytes of size followed by actual raw data.
    byte *ByteArrayVal;
  } Val;
};

using PropertySet = std::unordered_map<std::string, PropertyValue>;

/// A registry of property sets. Maps a property set name to its
/// content.
///
/// The order of keys is preserved and corresponds to the order of insertion.
class PropertySetRegistry {
public:
  using MapTy = std::unordered_map<std::string, PropertySet>;

  // SYCLBIN specific property sets.
  static constexpr char SYCLBIN_GLOBAL_METADATA[] = "SYCLBIN/global metadata";
  static constexpr char SYCLBIN_IR_MODULE_METADATA[] =
      "SYCLBIN/ir module metadata";
  static constexpr char SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA[] =
      "SYCLBIN/native device code image metadata";

  static std::unique_ptr<PropertySetRegistry> read(std::string_view Src) {
    auto Res = std::make_unique<PropertySetRegistry>();
    PropertySet *CurPropSet = nullptr;

    // special case when there is no property data, i.e. the resulting property
    // set registry should be empty
    if (Src.size() == 0)
      return Res;

    size_t CurrentStart = 0;
    while (CurrentStart < Src.size()) {
      size_t CurrentEnd = CurrentStart;
      size_t SkipChars = 0;
      for (CurrentEnd = CurrentStart; CurrentEnd < Src.size(); ++CurrentEnd) {
        if (Src[CurrentEnd] == '\n') {
          SkipChars = 1;
          break;
        }
        if (Src[CurrentEnd] == '\r' && CurrentEnd + 1 != Src.size() &&
            Src[CurrentEnd + 1] == '\n') {
          SkipChars = 2;
          break;
        }
      }

      std::string_view Line =
          Src.substr(CurrentStart, CurrentEnd - CurrentStart);
      CurrentStart = CurrentEnd + SkipChars;

      // see if this line starts a new property set
      if (Line.front() == '[') {
        // yes - parse the category (property name)
        auto EndPos = Line.rfind(']');
        if (EndPos == std::string_view::npos)
          throw sycl::exception(make_error_code(errc::invalid),
                                "Invalid line: " + std::string{Line});
        std::string_view Category = Line.substr(1, EndPos - 1);
        CurPropSet = &(*Res)[Category];
        continue;
      }
      if (!CurPropSet)
        throw sycl::exception(make_error_code(errc::invalid),
                              "Property category missing.");

      auto SplitSW = [](const std::string_view &View, char C) {
        std::string_view Left = View.substr(0, View.find(C));
        if (Left.size() >= View.size() - 1)
          return std::make_pair(Left, std::string_view{});
        std::string_view Right = View.substr(Left.size() + 1);
        return std::make_pair(Left, Right);
      };

      // parse name and type+value
      auto Parts = SplitSW(Line, '=');

      if (Parts.first.empty() || Parts.second.empty())
        throw sycl::exception(make_error_code(errc::invalid),
                              "Invalid property line: " + std::string{Line});
      auto TypeVal = SplitSW(Parts.second, '|');

      if (TypeVal.first.empty() || TypeVal.second.empty())
        throw sycl::exception(make_error_code(errc::invalid),
                              "Invalid property value: " +
                                  std::string{Parts.second});

      // parse type
      int Tint = stringViewToInt<int>(TypeVal.first);
      PropertyValue::Type Ttag = PropertyValue::getTypeTag(Tint);
      std::string_view Val = TypeVal.second;

      PropertyValue Prop(Ttag);

      // parse value depending on its type
      switch (Ttag) {
      case PropertyValue::Type::UINT32: {
        Prop.set(stringViewToInt<uint32_t>(Val));
        break;
      }
      case PropertyValue::Type::BYTE_ARRAY: {
        std::unique_ptr<byte[]> DecArr = Base64::decode(Val.data(), Val.size());
        Prop.set(DecArr.release());
        break;
      }
      default:
        throw sycl::exception(make_error_code(errc::invalid),
                              "Unsupported property type: " +
                                  std::to_string(Tint));
      }
      (*CurPropSet)[std::string{Parts.first}] = std::move(Prop);
    }

    return Res;
  }

  MapTy::const_iterator begin() const { return PropSetMap.begin(); }
  MapTy::const_iterator end() const { return PropSetMap.end(); }

  /// Retrieves a property set with given \p Name .
  PropertySet &operator[](std::string_view Name) {
    return PropSetMap[std::string{Name}];
  }
  /// Constant access to the underlying map.
  const MapTy &getPropSets() const { return PropSetMap; }

private:
  MapTy PropSetMap;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
