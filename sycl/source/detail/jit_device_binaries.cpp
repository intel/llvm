//==- jit_device_binaries.cpp - Runtime construction of PI device binaries -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/jit_device_binaries.hpp>

#include <cassert>

namespace sycl {
inline namespace _V1 {
namespace detail {

OffloadEntryContainer::OffloadEntryContainer(const std::string &Name,
                                             void *Addr, size_t Size,
                                             int32_t Flags, int32_t Reserved)
    : KernelName{new char[Name.length() + 1]}, Address{Addr}, EntrySize{Size},
      EntryFlags{Flags}, EntryReserved{Reserved} {
  std::memcpy(KernelName.get(), Name.c_str(), Name.length() + 1);
}

_pi_offload_entry_struct OffloadEntryContainer::getPIOffloadEntry() {
  return _pi_offload_entry_struct{Address, KernelName.get(), EntrySize,
                                  EntryFlags, EntryReserved};
}

PropertyContainer::PropertyContainer(const std::string &Name, void *Data,
                                     size_t Size, uint32_t Type)
    : PropName{new char[Name.length() + 1]}, Value{new unsigned char[Size]},
      ValueSize{Size}, PropType{Type} {
  std::memcpy(PropName.get(), Name.c_str(), Name.length() + 1);
  std::memcpy(Value.get(), Data, Size);
}

PropertyContainer::PropertyContainer(const std::string &Name, uint32_t Data)
    : PropName{new char[Name.length() + 1]}, Value{}, ValueSize{Data},
      PropType{PI_PROPERTY_TYPE_UINT32} {
  std::memcpy(PropName.get(), Name.c_str(), Name.length() + 1);
}

_pi_device_binary_property_struct PropertyContainer::getPIProperty() {
  return _pi_device_binary_property_struct{PropName.get(), Value.get(),
                                           PropType, ValueSize};
}

PropertySetContainer::PropertySetContainer(const std::string &Name)
    : SetName{new char[Name.length() + 1]} {
  std::memcpy(SetName.get(), Name.c_str(), Name.length() + 1);
}

void PropertySetContainer::addProperty(PropertyContainer &&Prop) {
  // Adding to the vectors might trigger reallocation, which would invalidate
  // the pointers used for PI structs if a PI struct has already been created
  // via getPIPropertySet(). Forbid calls to this method after the first PI
  // struct has been created.
  assert(Fused && "Adding to container would invalidate existing PI structs");
  PIProperties.push_back(Prop.getPIProperty());
  Properties.push_back(std::move(Prop));
}

_pi_device_binary_property_set_struct PropertySetContainer::getPIPropertySet() {
  Fused = false;
  return _pi_device_binary_property_set_struct{
      const_cast<char *>(SetName.get()), PIProperties.data(),
      PIProperties.data() + Properties.size()};
}

void DeviceBinaryContainer::addOffloadEntry(OffloadEntryContainer &&Cont) {
  // Adding to the vectors might trigger reallocation, which would invalidate
  // the pointers used for PI structs if a PI struct has already been created
  // via getPIDeviceBinary(). Forbid calls to this method after the first PI
  // struct has been created.
  assert(Fused && "Adding to container would invalidate existing PI structs");
  PIOffloadEntries.push_back(Cont.getPIOffloadEntry());
  OffloadEntries.push_back(std::move(Cont));
}

void DeviceBinaryContainer::addProperty(PropertySetContainer &&Cont) {
  // Adding to the vectors might trigger reallocation, which would invalidate
  // the pointers used for PI structs if a PI struct has already been created
  // via getPIDeviceBinary(). Forbid calls to this method after the first PI
  // struct has been created.
  assert(Fused && "Adding to container would invalidate existing PI structs");
  PIPropertySets.push_back(Cont.getPIPropertySet());
  PropertySets.push_back(std::move(Cont));
}

pi_device_binary_struct DeviceBinaryContainer::getPIDeviceBinary(
    const unsigned char *BinaryStart, size_t BinarySize, const char *TargetSpec,
    pi_device_binary_type Format) {
  pi_device_binary_struct DeviceBinary;
  DeviceBinary.Version = PI_DEVICE_BINARY_VERSION;
  DeviceBinary.Kind = PI_DEVICE_BINARY_OFFLOAD_KIND_SYCL;
  DeviceBinary.Format = Format;
  DeviceBinary.CompileOptions = "";
  DeviceBinary.LinkOptions = "";
  DeviceBinary.ManifestStart = nullptr;
  DeviceBinary.ManifestEnd = nullptr;
  // It is safe to use these pointers here, as their lifetime is managed by
  // the JITContext.
  DeviceBinary.BinaryStart = BinaryStart;
  DeviceBinary.BinaryEnd = BinaryStart + BinarySize;
  DeviceBinary.DeviceTargetSpec = TargetSpec;
  DeviceBinary.EntriesBegin = PIOffloadEntries.data();
  DeviceBinary.EntriesEnd = PIOffloadEntries.data() + PIOffloadEntries.size();
  DeviceBinary.PropertySetsBegin = PIPropertySets.data();
  DeviceBinary.PropertySetsEnd = PIPropertySets.data() + PIPropertySets.size();
  Fused = false;
  return DeviceBinary;
}

void DeviceBinariesCollection::addDeviceBinary(DeviceBinaryContainer &&Cont,
                                               const unsigned char *BinaryStart,
                                               size_t BinarySize,
                                               const char *TargetSpec,
                                               pi_device_binary_type Format) {
  // Adding to the vectors might trigger reallocation, which would invalidate
  // the pointers used for PI structs if a PI struct has already been created
  // via getPIDeviceStruct(). Forbid calls to this method after the first PI
  // struct has been created.
  assert(Fused && "Adding to container would invalidate existing PI structs");
  PIBinaries.push_back(
      Cont.getPIDeviceBinary(BinaryStart, BinarySize, TargetSpec, Format));
  Binaries.push_back(std::move(Cont));
}

pi_device_binaries DeviceBinariesCollection::getPIDeviceStruct() {

  PIStruct = std::make_unique<pi_device_binaries_struct>();
  PIStruct->Version = PI_DEVICE_BINARIES_VERSION;
  PIStruct->NumDeviceBinaries = PIBinaries.size();
  PIStruct->DeviceBinaries = PIBinaries.data();
  // According to documentation in pi.h, the HostEntries are not used and
  // can therefore be null.
  PIStruct->HostEntriesBegin = nullptr;
  PIStruct->HostEntriesEnd = nullptr;
  Fused = false;
  return PIStruct.get();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
