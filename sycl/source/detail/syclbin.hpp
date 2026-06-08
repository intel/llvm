//==--------------------- syclbin.hpp - SYCLBIN parser ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Adjusted copy of llvm/include/llvm/Object/SYCLBIN.h.
// TODO: Remove once we can consistently link the SYCL runtime library with
// LLVMObject.

#pragma once

#include "detail/compiler.hpp"
#include "detail/device_binary_image.hpp"
#include "detail/property_set_io.hpp"
#include "sycl/exception.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {

class device;

namespace detail {

class device_impl;

// Representation of a SYCLBIN binary object. This is intended for use as an
// image inside a OffloadBinary.
// Adjusted from llvm/include/llvm/Object/SYCLBIN.h and can be removed if
// LLVMObject gets linked into the SYCL runtime library.
class SYCLBIN {
public:
  SYCLBIN(const char *Data, size_t Size);

  SYCLBIN(const SYCLBIN &Other) = delete;
  SYCLBIN(SYCLBIN &&Other) = default;

  ~SYCLBIN() = default;

  SYCLBIN &operator=(const SYCLBIN &Other) = delete;
  SYCLBIN &operator=(SYCLBIN &&Other) = default;

  /// The current version of the binary used for backwards compatibility.
  static constexpr uint32_t CurrentVersion = 1;

  /// Magic number used to identify SYCLBIN files.
  static constexpr uint32_t MagicNumber = 0x53594249;

  struct IRModule {
    std::unique_ptr<PropertySetRegistry> Metadata;
    std::string_view RawIRBytes;
  };
  struct NativeDeviceCodeImage {
    std::unique_ptr<PropertySetRegistry> Metadata;
    std::string_view RawDeviceCodeImageBytes;
  };

  struct AbstractModule {
    std::unique_ptr<PropertySetRegistry> Metadata;
    std::vector<IRModule> IRModules;
    std::vector<NativeDeviceCodeImage> NativeDeviceCodeImages;
  };

  uint32_t Version;
  std::unique_ptr<PropertySetRegistry> GlobalMetadata;
  std::vector<AbstractModule> AbstractModules;

  // Description of a single image to be packed into a SYCLBIN. Unlike the
  // LLVMObject SYCLBIN writer (which reads bytes from files on disk), the
  // runtime serializer always operates on in-memory bytes since kernel_bundles
  // hold their device images in memory.
  //
  // Lifetime contract: SYCLBINDesc and the ImageDesc instances it owns are
  // intended for **synchronous use** with SYCLBIN::write. The Bytes view
  // borrows from external storage (typically RTDeviceBinaryImage payloads
  // owned by the program manager / SYCLBINBinaries / DynRTDeviceBinaryImage)
  // and must remain valid until SYCLBIN::write returns. Do not store a
  // SYCLBINDesc beyond the call.
  struct ImageDesc {
    // Per-image metadata (e.g. the IR/native image property set, plus any
    // forwarded property sets from the originating RTDeviceBinaryImage).
    std::string Metadata;
    // Raw image payload bytes. Non-owning view; see the lifetime contract on
    // SYCLBINDesc above.
    std::string_view Bytes;
  };

  // Description of a single abstract module: per-module metadata plus the
  // collection of IR and native images it groups together.
  struct AbstractModuleDesc {
    std::string Metadata;
    std::vector<ImageDesc> IRModules;
    std::vector<ImageDesc> NativeDeviceCodeImages;
  };

  // Description of a full SYCLBIN-to-be: a global metadata blob plus all the
  // abstract modules.
  struct SYCLBINDesc {
    std::string GlobalMetadata;
    std::vector<AbstractModuleDesc> AbstractModules;
  };

  // Serialize \p Desc into a SYCLBIN binary blob. Throws sycl::exception on
  // malformed input. \p Desc must outlive the call: ImageDesc::Bytes views are
  // dereferenced during serialization (see SYCLBINDesc lifetime contract).
  static std::vector<char> write(const SYCLBINDesc &Desc);

  // Per-image input for serializeImages. \p Image carries metadata
  // (property sets, format, target spec) used to build the abstract module
  // metadata. \p NativeBinaries, when non-empty, supplies device-specific
  // native binary bytes (one entry per device) extracted from a built UR
  // program; in that case the abstract module emits one
  // NativeDeviceCodeImage per entry rather than re-emitting \p Image's own
  // bytes. Used for JIT-built executable bundles where the source image is
  // SPIR-V/IR but the bundle's native payload only exists in a UR program.
  // \p Devices matches \p NativeBinaries 1:1 and is used to label each native
  // image with its device's architecture string.
  struct ImageInput {
    const RTDeviceBinaryImage *Image;
    std::vector<std::vector<char>> NativeBinaries;
    std::vector<device_impl *> Devices;
  };

  // Build and serialize a SYCLBIN representing the given runtime device images
  // at the given bundle state. Each input becomes its own abstract module.
  // Property sets carried on the source image are forwarded verbatim into the
  // abstract module metadata so device-requirement-driven matching (e.g.
  // `compile_target`) survives a round-trip.
  //
  // When ImageInput::NativeBinaries is non-empty, those bytes are emitted as
  // NativeDeviceCodeImages (replacing what would otherwise be IR module
  // serialization of the source image). Otherwise the source image's own
  // bytes are emitted, mirroring its on-disk format (IR vs. native).
  //
  // The Image pointers and NativeBinaries storage must remain valid for the
  // duration of the call: the intermediate SYCLBINDesc holds non-owning views
  // into each input's binary payload (see ImageDesc::Bytes).
  static std::vector<char>
  serializeImages(const std::vector<ImageInput> &Inputs, uint8_t State);

private:
  struct alignas(8) FileHeaderType {
    uint32_t Magic;
    uint32_t Version;
    uint32_t AbstractModuleCount;
    uint32_t IRModuleCount;
    uint32_t NativeDeviceCodeImageCount;
    uint64_t MetadataByteTableSize;
    uint64_t BinaryByteTableSize;
    uint64_t GlobalMetadataOffset;
    uint64_t GlobalMetadataSize;
  };

  struct alignas(8) AbstractModuleHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint32_t IRModuleCount;
    uint32_t IRModuleOffset;
    uint32_t NativeDeviceCodeImageCount;
    uint32_t NativeDeviceCodeImageOffset;
  };

  struct alignas(8) IRModuleHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint64_t RawIRBytesOffset;
    uint64_t RawIRBytesSize;
  };

  struct alignas(8) NativeDeviceCodeImageHeaderType {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint64_t BinaryBytesOffset;
    uint64_t BinaryBytesSize;
  };
};

// Helper class for managing both a SYCLBIN and binaries created from it,
// allowing existing infrastructure to better understand the contents of the
// SYCLBINs.
struct SYCLBINBinaries {
  // Delete copy-ctor to keep binaries unique and avoid costly copies of a
  // heavy structure.
  SYCLBINBinaries(const SYCLBINBinaries &) = delete;
  SYCLBINBinaries &operator=(const SYCLBINBinaries &) = delete;

  SYCLBINBinaries(SYCLBINBinaries &&) = default;
  SYCLBINBinaries &operator=(SYCLBINBinaries &&) = default;

  SYCLBINBinaries(const char *SYCLBINContent, size_t SYCLBINSize);

  ~SYCLBINBinaries() = default;

  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(device_impl &Dev, bundle_state State);
  std::vector<const RTDeviceBinaryImage *>
  getBestCompatibleImages(devices_range Dev, bundle_state State);

  std::vector<const RTDeviceBinaryImage *>
  getNativeBinaryImages(device_impl &Dev);

  uint8_t getState() const {
    PropertySet &GlobalMetadata =
        (*ParsedSYCLBIN
              .GlobalMetadata)[PropertySetRegistry::SYCLBIN_GLOBAL_METADATA];
    return static_cast<uint8_t>(
        GlobalMetadata[PropertySet::key_type{"state"}].asUint32());
  }

private:
  std::vector<_sycl_offload_entry_struct> &
  convertAbstractModuleEntries(const SYCLBIN::AbstractModule &AM);

  std::vector<_sycl_device_binary_property_set_struct> &
  convertAbstractModuleProperties(SYCLBIN::AbstractModule &AM);

  size_t getNumAbstractModules() const {
    return ParsedSYCLBIN.AbstractModules.size();
  }

  std::unique_ptr<char[]> SYCLBINContentCopy = nullptr;
  size_t SYCLBINContentCopySize = 0;

  SYCLBIN ParsedSYCLBIN;

  // Buffers for holding entries in the binary structs alive.
  std::vector<std::vector<_sycl_offload_entry_struct>> BinaryOffloadEntries;
  std::vector<std::vector<_sycl_device_binary_property_struct>>
      BinaryProperties;
  std::vector<std::vector<_sycl_device_binary_property_set_struct>>
      BinaryPropertySets;

  std::vector<sycl_device_binary_struct> DeviceBinaries;

  struct AbstractModuleDesc {
    size_t NumJITBinaries;
    size_t NumNativeBinaries;
    RTDeviceBinaryImage *JITBinaries;
    RTDeviceBinaryImage *NativeBinaries;
  };

  std::unique_ptr<AbstractModuleDesc[]> AbstractModuleDescriptors;
  std::unique_ptr<RTDeviceBinaryImage[]> BinaryImages;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
