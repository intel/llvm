//==----- device_binary_image.hpp --- SYCL device binary image abstraction -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/pi.hpp>

#include <sycl/detail/iostream_proxy.hpp>

#include <atomic>
#include <cstring>
#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

// A wrapper for passing around byte array properties
class ByteArray {
public:
  using ConstIterator = const std::uint8_t *;

  ByteArray(const std::uint8_t *Ptr, std::size_t Size) : Ptr{Ptr}, Size{Size} {}
  const std::uint8_t &operator[](std::size_t Idx) const { return Ptr[Idx]; }
  std::size_t size() const { return Size; }
  ConstIterator begin() const { return Ptr; }
  ConstIterator end() const { return Ptr + Size; }

  template <typename... Ts> auto consume() {
    if constexpr (sizeof...(Ts) == 1)
      return consumeOneElem<Ts...>();
    else
      return std::tuple{consumeOneElem<Ts>()...};
  }

  void dropBytes(std::size_t Bytes) {
    assert(Bytes <= Size && "Not enough bytes left!");
    Ptr += Bytes;
    Size -= Bytes;
  }

  template <typename T> void drop() { return dropBytes(sizeof(T)); }

  bool empty() const { return Size == 0; }

private:
  template <typename T> T consumeOneElem() {
    assert(sizeof(T) <= Size && "Out of bounds!");
    T Val;
    std::memcpy(&Val, Ptr, sizeof(T));
    drop<T>();
    return Val;
  }

  const std::uint8_t *Ptr;
  std::size_t Size;
};

// C++ wrapper over the _pi_device_binary_property_struct structure.
class DeviceBinaryProperty {
public:
  DeviceBinaryProperty(const _pi_device_binary_property_struct *Prop)
      : Prop(Prop) {}

  pi_uint32 asUint32() const;
  ByteArray asByteArray() const;
  const char *asCString() const;

protected:
  friend std::ostream &operator<<(std::ostream &Out,
                                  const DeviceBinaryProperty &P);
  const _pi_device_binary_property_struct *Prop;
};

std::ostream &operator<<(std::ostream &Out, const DeviceBinaryProperty &P);

// SYCL RT wrapper over PI binary image.
class RTDeviceBinaryImage {
public:
  // Represents a range of properties to enable iteration over them.
  // Implements the standard C++ STL input iterator interface.
  class PropertyRange {
  public:
    using ValTy = std::remove_pointer<pi_device_binary_property>::type;

    class ConstIterator {
      pi_device_binary_property Cur;

    public:
      using iterator_category = std::input_iterator_tag;
      using value_type = ValTy;
      using difference_type = ptrdiff_t;
      using pointer = const pi_device_binary_property;
      using reference = pi_device_binary_property;

      ConstIterator(pi_device_binary_property Cur = nullptr) : Cur(Cur) {}
      ConstIterator &operator++() {
        Cur++;
        return *this;
      }
      ConstIterator operator++(int) {
        ConstIterator Ret = *this;
        ++(*this);
        return Ret;
      }
      bool operator==(ConstIterator Other) const { return Cur == Other.Cur; }
      bool operator!=(ConstIterator Other) const { return !(*this == Other); }
      reference operator*() const { return Cur; }
    };
    ConstIterator begin() const { return ConstIterator(Begin); }
    ConstIterator end() const { return ConstIterator(End); }
    size_t size() const { return std::distance(begin(), end()); }
    friend class RTDeviceBinaryImage;
    bool isAvailable() const { return !(Begin == nullptr); }

  private:
    PropertyRange() : Begin(nullptr), End(nullptr) {}
    // Searches for a property set with given name and constructs a
    // PropertyRange spanning all its elements. If property set is not found,
    // the range will span zero elements.
    PropertyRange(pi_device_binary Bin, const char *PropSetName)
        : PropertyRange() {
      init(Bin, PropSetName);
    };
    void init(pi_device_binary Bin, const char *PropSetName);
    pi_device_binary_property Begin;
    pi_device_binary_property End;
  };

public:
  RTDeviceBinaryImage() : Bin(nullptr) {}
  RTDeviceBinaryImage(pi_device_binary Bin) { init(Bin); }
  // Explicitly delete copy constructor/operator= to avoid unintentional copies
  RTDeviceBinaryImage(const RTDeviceBinaryImage &) = delete;
  RTDeviceBinaryImage &operator=(const RTDeviceBinaryImage &) = delete;
  // Explicitly retain move constructors to facilitate potential moves across
  // collections
  RTDeviceBinaryImage(RTDeviceBinaryImage &&) = default;
  RTDeviceBinaryImage &operator=(RTDeviceBinaryImage &&) = default;

  virtual ~RTDeviceBinaryImage() {}

  bool supportsSpecConstants() const {
    return getFormat() == PI_DEVICE_BINARY_TYPE_SPIRV;
  }

  const pi_device_binary_struct &getRawData() const { return *get(); }

  virtual void print() const;
  virtual void dump(std::ostream &Out) const;

  size_t getSize() const {
    assert(Bin && "binary image data not set");
    return static_cast<size_t>(Bin->BinaryEnd - Bin->BinaryStart);
  }

  const char *getCompileOptions() const {
    assert(Bin && "binary image data not set");
    return Bin->CompileOptions;
  }

  const char *getLinkOptions() const {
    assert(Bin && "binary image data not set");
    return Bin->LinkOptions;
  }

  /// Returns the format of the binary image
  pi::PiDeviceBinaryType getFormat() const {
    assert(Bin && "binary image data not set");
    return Format;
  }

  /// Returns a single property from SYCL_MISC_PROP category.
  pi_device_binary_property getProperty(const char *PropName) const;

  /// Gets the iterator range over specialization constants in this binary
  /// image. For each property pointed to by an iterator within the
  /// range, the name of the property is the specialization constant symbolic ID
  /// and the value is a list of 3-element tuples of 32-bit unsigned integers,
  /// describing the specialization constant.
  /// This is done in order to unify representation of both scalar and composite
  /// specialization constants: composite specialization constant is represented
  /// by its leaf elements, so for scalars the list contains only a single
  /// tuple, while for composite there might be more of them.
  /// Each tuple consists of ID of scalar specialization constant, its location
  /// within a composite (offset in bytes from the beginning or 0 if it is not
  /// an element of a composite specialization constant) and its size.
  /// For example, for the following structure:
  /// struct A { int a; float b; };
  /// struct POD { A a[2]; int b; };
  /// List of tuples will look like:
  /// { ID0, 0, 4 },  // .a[0].a
  /// { ID1, 4, 4 },  // .a[0].b
  /// { ID2, 8, 4 },  // .a[1].a
  /// { ID3, 12, 4 }, // .a[1].b
  /// { ID4, 16, 4 }, // .b
  /// And for an interger specialization constant, the list of tuples will look
  /// like:
  /// { ID5, 0, 4 }
  const PropertyRange &getSpecConstants() const { return SpecConstIDMap; }
  const PropertyRange &getSpecConstantsDefaultValues() const {
    return SpecConstDefaultValuesMap;
  }
  const PropertyRange &getDeviceLibReqMask() const { return DeviceLibReqMask; }
  const PropertyRange &getKernelParamOptInfo() const {
    return KernelParamOptInfo;
  }
  const PropertyRange &getAssertUsed() const { return AssertUsed; }
  const PropertyRange &getProgramMetadata() const { return ProgramMetadata; }
  const PropertyRange &getExportedSymbols() const { return ExportedSymbols; }
  const PropertyRange &getDeviceGlobals() const { return DeviceGlobals; }
  const PropertyRange &getDeviceRequirements() const {
    return DeviceRequirements;
  }
  const PropertyRange &getHostPipes() const { return HostPipes; }
  const PropertyRange &getVirtualFunctions() const { return VirtualFunctions; }

  std::uintptr_t getImageID() const {
    assert(Bin && "Image ID is not available without a binary image.");
    return ImageId;
  }

protected:
  void init(pi_device_binary Bin);
  pi_device_binary get() const { return Bin; }

  pi_device_binary Bin;

  pi::PiDeviceBinaryType Format = PI_DEVICE_BINARY_TYPE_NONE;
  RTDeviceBinaryImage::PropertyRange SpecConstIDMap;
  RTDeviceBinaryImage::PropertyRange SpecConstDefaultValuesMap;
  RTDeviceBinaryImage::PropertyRange DeviceLibReqMask;
  RTDeviceBinaryImage::PropertyRange KernelParamOptInfo;
  RTDeviceBinaryImage::PropertyRange AssertUsed;
  RTDeviceBinaryImage::PropertyRange ProgramMetadata;
  RTDeviceBinaryImage::PropertyRange ExportedSymbols;
  RTDeviceBinaryImage::PropertyRange DeviceGlobals;
  RTDeviceBinaryImage::PropertyRange DeviceRequirements;
  RTDeviceBinaryImage::PropertyRange HostPipes;
  RTDeviceBinaryImage::PropertyRange VirtualFunctions;

private:
  static std::atomic<uintptr_t> ImageCounter;
  uintptr_t ImageId = 0;
};

// Dynamically allocated device binary image, which de-allocates its binary
// data in destructor.
class DynRTDeviceBinaryImage : public RTDeviceBinaryImage {
public:
  DynRTDeviceBinaryImage(std::unique_ptr<char[]> &&DataPtr, size_t DataSize);
  ~DynRTDeviceBinaryImage() override;

  void print() const override {
    RTDeviceBinaryImage::print();
    std::cerr << "    DYNAMICALLY CREATED\n";
  }

protected:
  std::unique_ptr<char[]> Data;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
