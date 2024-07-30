//==- jit_device_binaries.hpp - Runtime construction of UR device binaries -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/compiler.hpp>
#include <sycl/detail/ur.hpp>

#include <cstring>
#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

/// Representation of _sycl_offload_entry_struct for creation of JIT device
/// binaries at runtime. Owns the necessary data and provides raw pointers for
/// the UR struct.
class OffloadEntryContainer {
public:
  OffloadEntryContainer(const std::string &Name, void *Addr, size_t Size,
                        int32_t Flags, int32_t Reserved);

  OffloadEntryContainer(OffloadEntryContainer &&) = default;
  OffloadEntryContainer &operator=(OffloadEntryContainer &&) = default;
  ~OffloadEntryContainer() = default;
  // Copying of the container is not allowed.
  OffloadEntryContainer(const OffloadEntryContainer &) = delete;
  OffloadEntryContainer &operator=(const OffloadEntryContainer &) = delete;

  _sycl_offload_entry_struct getPIOffloadEntry();

private:
  std::unique_ptr<char[]> KernelName;

  void *Address;
  size_t EntrySize;
  int32_t EntryFlags;
  int32_t EntryReserved;
};

/// Representation of _sycl_device_binary_property_struct for creation of JIT
/// device binaries at runtime.
/// Owns the necessary data and provides raw pointers for the UR struct.
class PropertyContainer {

public:
  PropertyContainer(const std::string &Name, void *Data, size_t Size,
                    uint32_t Type);
  // Set a UR_PROPERTY_TYPE_UINT32 property
  PropertyContainer(const std::string &Name, uint32_t Data);

  PropertyContainer(PropertyContainer &&) = default;
  PropertyContainer &operator=(PropertyContainer &&) = default;
  ~PropertyContainer() = default;
  // Copying of the container is not allowed.
  PropertyContainer(const PropertyContainer &) = delete;
  PropertyContainer &operator=(const PropertyContainer &) = delete;

  _sycl_device_binary_property_struct getPIProperty();

private:
  std::unique_ptr<char[]> PropName;
  std::unique_ptr<unsigned char[]> Value;
  size_t ValueSize;
  uint32_t PropType;
};

/// Representation of _sycl_device_binary_property_set_struct for creation of
/// JIT device binaries at runtime. Owns the necessary data and provides raw
/// pointers for the UR struct.
class PropertySetContainer {
public:
  PropertySetContainer(const std::string &Name);

  PropertySetContainer(PropertySetContainer &&) = default;
  PropertySetContainer &operator=(PropertySetContainer &&) = default;
  ~PropertySetContainer() = default;
  // Copying of the container is not allowed, as it would invalidate UR structs.
  PropertySetContainer(const PropertySetContainer &) = delete;
  PropertySetContainer &operator=(const PropertySetContainer &) = delete;

  void addProperty(PropertyContainer &&Prop);

  _sycl_device_binary_property_set_struct getPIPropertySet();

private:
  std::unique_ptr<char[]> SetName;
  bool Fused = true;
  std::vector<PropertyContainer> Properties;
  std::vector<_sycl_device_binary_property_struct> PIProperties;
};

/// Representation of sycl_device_binary_struct for creation of JIT device
/// binaries at runtime.
/// Owns the necessary data and provides raw pointers for the UR struct.
class DeviceBinaryContainer {
public:
  DeviceBinaryContainer() = default;
  DeviceBinaryContainer(DeviceBinaryContainer &&) = default;
  DeviceBinaryContainer &operator=(DeviceBinaryContainer &&) = default;
  ~DeviceBinaryContainer() = default;
  // Copying of the container is not allowed, as it would invalidate UR structs.
  DeviceBinaryContainer(const DeviceBinaryContainer &) = delete;
  DeviceBinaryContainer &operator=(const DeviceBinaryContainer &) = delete;

  void addOffloadEntry(OffloadEntryContainer &&Cont);

  void addProperty(PropertySetContainer &&Cont);

  sycl_device_binary_struct getPIDeviceBinary(const unsigned char *BinaryStart,
                                              size_t BinarySize,
                                              const char *TargetSpec,
                                              sycl_device_binary_type Format);

private:
  bool Fused = true;
  std::vector<OffloadEntryContainer> OffloadEntries;
  std::vector<_sycl_offload_entry_struct> PIOffloadEntries;
  std::vector<PropertySetContainer> PropertySets;
  std::vector<_sycl_device_binary_property_set_struct> PIPropertySets;
};

/// Representation of sycl_device_binaries_struct for creation of JIT device
/// binaries at runtime.
/// Owns the necessary data and provides raw pointers for the UR struct.
class DeviceBinariesCollection {

public:
  DeviceBinariesCollection() = default;
  DeviceBinariesCollection(DeviceBinariesCollection &&) = default;
  DeviceBinariesCollection &operator=(DeviceBinariesCollection &&) = default;
  ~DeviceBinariesCollection() = default;
  // Copying of the container is not allowed.
  DeviceBinariesCollection(const DeviceBinariesCollection &) = delete;
  DeviceBinariesCollection &
  operator=(const DeviceBinariesCollection &) = delete;

  void addDeviceBinary(DeviceBinaryContainer &&Cont,
                       const unsigned char *BinaryStart, size_t BinarySize,
                       const char *TargetSpec, sycl_device_binary_type Format);
  sycl_device_binaries getPIDeviceStruct();

private:
  bool Fused = true;
  std::unique_ptr<sycl_device_binaries_struct> PIStruct;

  std::vector<DeviceBinaryContainer> Binaries;
  std::vector<sycl_device_binary_struct> PIBinaries;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
