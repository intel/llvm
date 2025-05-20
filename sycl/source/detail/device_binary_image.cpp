//==----- device_binary_image.cpp --- SYCL device binary image abstraction -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_binary_image.hpp>
#include <sycl/detail/ur.hpp>

// For device image compression.
#include <detail/compression.hpp>

#include <llvm/Support/PropertySetIO.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace sycl {
inline namespace _V1 {
namespace detail {

std::ostream &operator<<(std::ostream &Out, const DeviceBinaryProperty &P) {
  switch (P.Prop->Type) {
  case SYCL_PROPERTY_TYPE_UINT32:
    Out << "[UINT32] ";
    break;
  case SYCL_PROPERTY_TYPE_BYTE_ARRAY:
    Out << "[Byte array] ";
    break;
  case SYCL_PROPERTY_TYPE_STRING:
    Out << "[String] ";
    break;
  default:
    assert(false && "unsupported property");
    return Out;
  }
  Out << P.Prop->Name << "=";

  switch (P.Prop->Type) {
  case SYCL_PROPERTY_TYPE_UINT32:
    Out << P.asUint32();
    break;
  case SYCL_PROPERTY_TYPE_BYTE_ARRAY: {
    ByteArray BA = P.asByteArray();
    std::ios_base::fmtflags FlagsBackup = Out.flags();
    Out << std::hex;
    for (const auto &Byte : BA) {
      Out << "0x" << static_cast<unsigned>(Byte) << " ";
    }
    Out.flags(FlagsBackup);
    break;
  }
  case SYCL_PROPERTY_TYPE_STRING:
    Out << P.asCString();
    break;
  default:
    assert(false && "Unsupported property");
    return Out;
  }
  return Out;
}

uint32_t DeviceBinaryProperty::asUint32() const {
  assert(Prop->Type == SYCL_PROPERTY_TYPE_UINT32 && "property type mismatch");
  // if type fits into the ValSize - it is used to store the property value
  assert(Prop->ValAddr == nullptr && "primitive types must be stored inline");
  const auto *P = reinterpret_cast<const unsigned char *>(&Prop->ValSize);
  return (*P) | (*(P + 1) << 8) | (*(P + 2) << 16) | (*(P + 3) << 24);
}

ByteArray DeviceBinaryProperty::asByteArray() const {
  assert(Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY &&
         "property type mismatch");
  assert(Prop->ValSize > 0 && "property size mismatch");
  const auto *Data = ur::cast<const std::uint8_t *>(Prop->ValAddr);
  return {Data, Prop->ValSize};
}

const char *DeviceBinaryProperty::asCString() const {
  assert((Prop->Type == SYCL_PROPERTY_TYPE_STRING ||
          Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY) &&
         "property type mismatch");
  assert(Prop->ValSize > 0 && "property size mismatch");
  // Byte array stores its size in first 8 bytes
  size_t Shift = Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY ? 8 : 0;
  return ur::cast<const char *>(Prop->ValAddr) + Shift;
}

void RTDeviceBinaryImage::PropertyRange::init(sycl_device_binary Bin,
                                              const char *PropSetName) {
  assert(!this->Begin && !this->End && "already initialized");
  sycl_device_binary_property_set PS = nullptr;

  for (PS = Bin->PropertySetsBegin; PS != Bin->PropertySetsEnd; ++PS) {
    assert(PS->Name && "nameless property set - bug in the offload wrapper?");
    if (!strcmp(PropSetName, PS->Name))
      break;
  }
  if (PS == Bin->PropertySetsEnd) {
    Begin = End = nullptr;
    return;
  }
  Begin = PS->PropertiesBegin;
  End = Begin ? PS->PropertiesEnd : nullptr;
}

void RTDeviceBinaryImage::print() const {
  std::cerr << "  --- Image " << Bin << "\n";
  if (!Bin)
    return;
  std::cerr << "    Version  : " << (int)Bin->Version << "\n";
  std::cerr << "    Kind     : " << (int)Bin->Kind << "\n";
  std::cerr << "    Format   : " << (int)Bin->Format << "\n";
  std::cerr << "    Target   : " << Bin->DeviceTargetSpec << "\n";
  std::cerr << "    Bin size : "
            << ((intptr_t)Bin->BinaryEnd - (intptr_t)Bin->BinaryStart) << "\n";
  std::cerr << "    Compile options : "
            << (Bin->CompileOptions ? Bin->CompileOptions : "NULL") << "\n";
  std::cerr << "    Link options    : "
            << (Bin->LinkOptions ? Bin->LinkOptions : "NULL") << "\n";
  std::cerr << "    Entries  : ";

  for (sycl_offload_entry EntriesIt = Bin->EntriesBegin;
       EntriesIt != Bin->EntriesEnd; EntriesIt = EntriesIt->Increment())
    std::cerr << EntriesIt->GetName() << " ";
  std::cerr << "\n";
  std::cerr << "    Properties [" << Bin->PropertySetsBegin << "-"
            << Bin->PropertySetsEnd << "]:\n";

  for (sycl_device_binary_property_set PS = Bin->PropertySetsBegin;
       PS != Bin->PropertySetsEnd; ++PS) {
    std::cerr << "      Category " << PS->Name << " [" << PS->PropertiesBegin
              << "-" << PS->PropertiesEnd << "]:\n";

    for (sycl_device_binary_property P = PS->PropertiesBegin;
         P != PS->PropertiesEnd; ++P) {
      std::cerr << "        " << DeviceBinaryProperty(P) << "\n";
    }
  }
}

void RTDeviceBinaryImage::dump(std::ostream &Out) const {
  size_t ImgSize = getSize();
  Out.write(reinterpret_cast<const char *>(Bin->BinaryStart), ImgSize);
}

sycl_device_binary_property
RTDeviceBinaryImage::getProperty(const char *PropName) const {
  if (!Misc.isAvailable())
    return nullptr;
  auto It = std::find_if(Misc.begin(), Misc.end(),
                         [=](sycl_device_binary_property Prop) {
                           return !strcmp(PropName, Prop->Name);
                         });
  if (It == Misc.end())
    return nullptr;

  return *It;
}

void RTDeviceBinaryImage::init(sycl_device_binary Bin) {
  ImageId = ImageCounter++;

  // If there was no binary, we let the owner handle initialization as they see
  // fit. This is used when merging binaries, e.g. during linking.
  if (!Bin)
    return;

  // Bin != nullptr is guaranteed here.
  this->Bin = Bin;
  // If device binary image format wasn't set by its producer, then can't change
  // now, because 'Bin' data is part of the executable image loaded into memory
  // which can't be modified (easily).
  // TODO clang driver + ClangOffloadWrapper can figure out the format and set
  // it when invoking the offload wrapper job
  Format = static_cast<ur::DeviceBinaryType>(Bin->Format);

  // For compressed images, we delay determining the format until the image is
  // decompressed.
  if (Format == SYCL_DEVICE_BINARY_TYPE_NONE)
    // try to determine the format; may remain "NONE"
    Format = ur::getBinaryImageFormat(Bin->BinaryStart, getSize());

  SpecConstIDMap.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_SPECIALIZATION_CONSTANTS);
  SpecConstDefaultValuesMap.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_SPEC_CONSTANTS_DEFAULT_VALUES);
  DeviceLibReqMask.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_DEVICELIB_REQ_MASK);
  DeviceLibMetadata.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_DEVICELIB_METADATA);
  KernelParamOptInfo.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO);
  AssertUsed.init(Bin, llvm::util::PropertySetRegistry::SYCL_ASSERT_USED);
  ImplicitLocalArg.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_IMPLICIT_LOCAL_ARG);
  ProgramMetadata.init(Bin,
                       llvm::util::PropertySetRegistry::SYCL_PROGRAM_METADATA);
  // Convert ProgramMetadata into the UR format
  for (const auto &Prop : ProgramMetadata) {
    ProgramMetadataUR.push_back(
        ur::mapDeviceBinaryPropertyToProgramMetadata(Prop));
  }
  ExportedSymbols.init(Bin,
                       llvm::util::PropertySetRegistry::SYCL_EXPORTED_SYMBOLS);
  ImportedSymbols.init(Bin,
                       llvm::util::PropertySetRegistry::SYCL_IMPORTED_SYMBOLS);
  DeviceGlobals.init(Bin, llvm::util::PropertySetRegistry::SYCL_DEVICE_GLOBALS);
  DeviceRequirements.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_DEVICE_REQUIREMENTS);
  HostPipes.init(Bin, llvm::util::PropertySetRegistry::SYCL_HOST_PIPES);
  VirtualFunctions.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_VIRTUAL_FUNCTIONS);
  RegisteredKernels.init(
      Bin, llvm::util::PropertySetRegistry::SYCL_REGISTERED_KERNELS);
  Misc.init(Bin, llvm::util::PropertySetRegistry::SYCL_MISC_PROP);
}

std::atomic<uintptr_t> RTDeviceBinaryImage::ImageCounter = 1;

DynRTDeviceBinaryImage::DynRTDeviceBinaryImage() : RTDeviceBinaryImage() {
  Bin = new sycl_device_binary_struct();
  Bin->Version = SYCL_DEVICE_BINARY_VERSION;
  Bin->Kind = SYCL_DEVICE_BINARY_OFFLOAD_KIND_SYCL;
  Bin->CompileOptions = "";
  Bin->LinkOptions = "";
  Bin->ManifestStart = nullptr;
  Bin->ManifestEnd = nullptr;
  Bin->BinaryStart = nullptr;
  Bin->BinaryEnd = nullptr;
  Bin->EntriesBegin = nullptr;
  Bin->EntriesEnd = nullptr;
  Bin->Format = SYCL_DEVICE_BINARY_TYPE_NONE;
  Bin->DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_UNKNOWN;
}

DynRTDeviceBinaryImage::DynRTDeviceBinaryImage(
    std::unique_ptr<char[], std::function<void(void *)>> &&DataPtr,
    size_t DataSize)
    : DynRTDeviceBinaryImage() {
  Data = std::move(DataPtr);
  Bin->BinaryStart = reinterpret_cast<unsigned char *>(Data.get());
  Bin->BinaryEnd = Bin->BinaryStart + DataSize;
  Bin->Format = ur::getBinaryImageFormat(Bin->BinaryStart, DataSize);
  switch (Bin->Format) {
  case SYCL_DEVICE_BINARY_TYPE_SPIRV:
    Bin->DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_SPIRV64;
    break;
  default:
    Bin->DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_UNKNOWN;
  }
  init(Bin);
}

DynRTDeviceBinaryImage::~DynRTDeviceBinaryImage() {
  delete Bin;
  Bin = nullptr;
}

// "Naive" property merge logic. It merges the properties into a single property
// vector without checking for duplicates. As such, duplicates may occur in the
// final result.
template <typename RangeGetterT>
static std::vector<sycl_device_binary_property>
naiveMergeBinaryProperties(const std::vector<const RTDeviceBinaryImage *> &Imgs,
                           const RangeGetterT &RangeGetter) {
  size_t PropertiesCount = 0;
  for (const RTDeviceBinaryImage *Img : Imgs)
    PropertiesCount += RangeGetter(*Img).size();

  std::vector<sycl_device_binary_property> Props;
  Props.reserve(PropertiesCount);
  for (const RTDeviceBinaryImage *Img : Imgs) {
    const RTDeviceBinaryImage::PropertyRange &Range = RangeGetter(*Img);
    Props.insert(Props.end(), Range.begin(), Range.end());
  }

  return Props;
}

// Exclusive property merge logic. If IgnoreDuplicates is false it assumes there
// are no cases where properties have different values and throws otherwise.
template <typename RangeGetterT>
static std::unordered_map<std::string_view, const sycl_device_binary_property>
exclusiveMergeBinaryProperties(
    const std::vector<const RTDeviceBinaryImage *> &Imgs,
    const RangeGetterT &RangeGetter, bool IgnoreDuplicates = false) {
  std::unordered_map<std::string_view, const sycl_device_binary_property>
      MergeMap;
  for (const RTDeviceBinaryImage *Img : Imgs) {
    const RTDeviceBinaryImage::PropertyRange &Range = RangeGetter(*Img);
    for (const sycl_device_binary_property Prop : Range) {
      const auto [It, Inserted] =
          MergeMap.try_emplace(std::string_view{Prop->Name}, Prop);
      if (IgnoreDuplicates || Inserted)
        continue;
      // If we didn't insert a new entry, check that the old entry had the
      // exact same value.
      const sycl_device_binary_property OtherProp = It->second;
      if (OtherProp->Type != Prop->Type ||
          OtherProp->ValSize != Prop->ValSize ||
          (Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY &&
           std::memcmp(OtherProp->ValAddr, Prop->ValAddr, Prop->ValSize) != 0))
        throw sycl::exception(make_error_code(errc::invalid),
                              "Unable to merge incompatible images.");
    }
  }
  return MergeMap;
}

// Device requirements needs the ability to produce new properties. The
// information for these are kept in this struct.
struct MergedDeviceRequirements {
  std::unordered_map<std::string_view, const sycl_device_binary_property>
      MergeMap;
  std::unordered_set<uint32_t> Aspects;
  std::unordered_set<std::string_view> JointMatrix;
  std::unordered_set<std::string_view> JointMatrixMad;

  size_t getPropertiesCount() const {
    return MergeMap.size() + !Aspects.empty() + !JointMatrix.empty() +
           !JointMatrixMad.empty();
  }

  size_t getAspectsContentSize() const {
    return Aspects.size() * sizeof(uint32_t);
  }

  static size_t
  getStringSetContentSize(const std::unordered_set<std::string_view> &Set) {
    size_t Result = 0;
    Result += Set.size() - 1;               // Semi-colon delimiters.
    for (const std::string_view &Str : Set) // Strings.
      Result += Str.size();
    return Result;
  }

  size_t getPropertiesContentByteSize() const {
    size_t Result = 0;
    for (const auto &PropIt : MergeMap)
      Result += strlen(PropIt.second->Name) + 1 + PropIt.second->ValSize;

    if (!Aspects.empty())
      Result += strlen("aspects") + 1 + getAspectsContentSize();

    if (!JointMatrix.empty())
      Result +=
          strlen("joint_matrix") + 1 + getStringSetContentSize(JointMatrix);

    if (!JointMatrixMad.empty())
      Result += strlen("joint_matrix_mad") + 1 +
                getStringSetContentSize(JointMatrixMad);

    return Result;
  }

  void writeAspectProperty(sycl_device_binary_property &NextFreeProperty,
                           char *&NextFreeContent) const {
    if (Aspects.empty())
      return;
    // Get the next free property entry and move the needle.
    sycl_device_binary_property NewProperty = NextFreeProperty++;
    NewProperty->Type = SYCL_PROPERTY_TYPE_BYTE_ARRAY;
    NewProperty->ValSize = getAspectsContentSize();
    // Copy the name.
    const size_t NameLen = std::strlen("aspects");
    std::memcpy(NextFreeContent, "aspects", NameLen + 1);
    NewProperty->Name = NextFreeContent;
    NextFreeContent += NameLen + 1;
    // Copy the values.
    uint32_t *AspectContentIt = reinterpret_cast<uint32_t *>(NextFreeContent);
    for (uint32_t Aspect : Aspects)
      *(AspectContentIt++) = Aspect;
    NewProperty->ValAddr = NextFreeContent;
    NextFreeContent += NewProperty->ValSize;
  }

  static void writeStringSetProperty(
      const std::unordered_set<std::string_view> &Set, const char *SetName,
      sycl_device_binary_property &NextFreeProperty, char *&NextFreeContent) {
    if (Set.empty())
      return;
    // Get the next free property entry and move the needle.
    sycl_device_binary_property NewProperty = NextFreeProperty++;
    NewProperty->Type = SYCL_PROPERTY_TYPE_BYTE_ARRAY;
    NewProperty->ValSize = getStringSetContentSize(Set);
    // Copy the name.
    const size_t NameLen = std::strlen(SetName);
    std::memcpy(NextFreeContent, SetName, NameLen + 1);
    NewProperty->Name = NextFreeContent;
    NextFreeContent += NameLen + 1;
    // Copy the values.
    NewProperty->ValAddr = NextFreeContent;
    for (auto StrIt = Set.begin(); StrIt != Set.end(); ++StrIt) {
      if (StrIt != Set.begin())
        *(NextFreeContent++) = ';';
      std::memcpy(NextFreeContent, StrIt->data(), StrIt->size());
      NextFreeContent += StrIt->size();
    }
  }
};

// Merging device requirements is a little more involved, as it may impose
// new requirements.
static MergedDeviceRequirements
mergeDeviceRequirements(const std::vector<const RTDeviceBinaryImage *> &Imgs) {
  MergedDeviceRequirements MergedReqs;
  for (const RTDeviceBinaryImage *Img : Imgs) {
    const RTDeviceBinaryImage::PropertyRange &Range =
        Img->getDeviceRequirements();
    for (const sycl_device_binary_property Prop : Range) {
      std::string_view NameView{Prop->Name};

      // Aspects we collect in a set early and add them afterwards.
      if (NameView == "aspects") {
        // Skip size bytes.
        auto AspectIt = reinterpret_cast<const uint32_t *>(
            reinterpret_cast<char *>(Prop->ValAddr) + 8);
        for (size_t I = 0; I < Prop->ValSize / sizeof(uint32_t); ++I)
          MergedReqs.Aspects.emplace(AspectIt[I]);
        continue;
      }

      // joint_matrix and joint_matrix_mad have the same format, so we parse
      // them the same way.
      if (NameView == "joint_matrix" || NameView == "joint_matrix_mad") {
        std::unordered_set<std::string_view> &Set =
            NameView == "joint_matrix" ? MergedReqs.JointMatrix
                                       : MergedReqs.JointMatrixMad;

        // Skip size bytes.
        std::string_view Contents{reinterpret_cast<char *>(Prop->ValAddr) + 8,
                                  Prop->ValSize};
        size_t Pos = 0;
        do {
          const size_t NextPos = Contents.find(';', Pos);
          if (NextPos != Pos)
            Set.emplace(Contents.substr(Pos, NextPos - Pos));
          Pos = NextPos + 1;
        } while (Pos != 0);
        continue;
      }

      const auto [It, Inserted] =
          MergedReqs.MergeMap.try_emplace(NameView, Prop);
      if (Inserted)
        continue;
      // Special handling has already happened, so we assume the rest are
      // exclusive property values.
      const sycl_device_binary_property OtherProp = It->second;
      if (OtherProp->Type != Prop->Type ||
          OtherProp->ValSize != Prop->ValSize ||
          (Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY &&
           std::memcmp(OtherProp->ValAddr, Prop->ValAddr, Prop->ValSize) != 0))
        throw sycl::exception(make_error_code(errc::invalid),
                              "Unable to merge incompatible images.");
    }
  }
  return MergedReqs;
}

// Copies a property into new memory.
static void copyProperty(sycl_device_binary_property &NextFreeProperty,
                         char *&NextFreeContent,
                         const sycl_device_binary_property OldProperty) {
  // Get the next free property entry and move the needle.
  sycl_device_binary_property NewProperty = NextFreeProperty++;
  NewProperty->Type = OldProperty->Type;
  NewProperty->ValSize = OldProperty->ValSize;
  // Copy the name.
  const size_t NameLen = std::strlen(OldProperty->Name);
  std::memcpy(NextFreeContent, OldProperty->Name, NameLen + 1);
  NewProperty->Name = NextFreeContent;
  NextFreeContent += NameLen + 1;
  // Copy the values. If the type is uint32 it will have been stored in the size
  // instead of the value address.
  if (OldProperty->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY) {
    std::memcpy(NextFreeContent, OldProperty->ValAddr, OldProperty->ValSize);
    NewProperty->ValAddr = NextFreeContent;
    NextFreeContent += OldProperty->ValSize;
  } else {
    NewProperty->ValAddr = nullptr;
  }
}

DynRTDeviceBinaryImage::DynRTDeviceBinaryImage(
    const std::vector<const RTDeviceBinaryImage *> &Imgs)
    : DynRTDeviceBinaryImage() {
  init(nullptr);

  // Naive merges.
  auto MergedSpecConstants =
      naiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getSpecConstants();
      });
  auto MergedSpecConstantsDefaultValues =
      naiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getSpecConstantsDefaultValues();
      });
  auto MergedKernelParamOptInfo =
      naiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getKernelParamOptInfo();
      });
  auto MergedAssertUsed = naiveMergeBinaryProperties(
      Imgs, [](const RTDeviceBinaryImage &Img) { return Img.getAssertUsed(); });
  auto MergedDeviceGlobals =
      naiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getDeviceGlobals();
      });
  auto MergedHostPipes = naiveMergeBinaryProperties(
      Imgs, [](const RTDeviceBinaryImage &Img) { return Img.getHostPipes(); });
  auto MergedVirtualFunctions =
      naiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getVirtualFunctions();
      });
  auto MergedImplicitLocalArg =
      naiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getImplicitLocalArg();
      });
  auto MergedExportedSymbols =
      naiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getExportedSymbols();
      });
  auto MergedRegisteredKernels =
      naiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getRegisteredKernels();
      });

  std::array<const std::vector<sycl_device_binary_property> *, 10> MergedVecs{
      &MergedSpecConstants,      &MergedSpecConstantsDefaultValues,
      &MergedKernelParamOptInfo, &MergedAssertUsed,
      &MergedDeviceGlobals,      &MergedHostPipes,
      &MergedVirtualFunctions,   &MergedImplicitLocalArg,
      &MergedExportedSymbols,    &MergedRegisteredKernels};

  // Exclusive merges.
  auto MergedDeviceLibReqMask =
      exclusiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getDeviceLibReqMask();
      });
  auto MergedProgramMetadata =
      exclusiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getProgramMetadata();
      });
  auto MergedImportedSymbols = exclusiveMergeBinaryProperties(
      Imgs,
      [](const RTDeviceBinaryImage &Img) { return Img.getImportedSymbols(); },
      /*IgnoreDuplicates=*/true);
  auto MergedMisc =
      exclusiveMergeBinaryProperties(Imgs, [](const RTDeviceBinaryImage &Img) {
        return Img.getMiscProperties();
      });

  std::array<const std::unordered_map<std::string_view,
                                      const sycl_device_binary_property> *,
             4>
      MergedMaps{&MergedDeviceLibReqMask, &MergedProgramMetadata,
                 &MergedImportedSymbols, &MergedMisc};

  // When merging exported and imported, the exported symbols may cancel out
  // some of the imported symbols.
  for (const sycl_device_binary_property Prop : MergedExportedSymbols)
    MergedImportedSymbols.erase(std::string_view{Prop->Name});

  // For device requirements we need to do special handling to merge the
  // property values as well.
  MergedDeviceRequirements MergedDevReqs = mergeDeviceRequirements(Imgs);

  // Now that we have merged all properties, we need to calculate how much
  // memory we need to store the new property sets.
  constexpr size_t PropertyByteSize =
      sizeof(_sycl_device_binary_property_struct);
  constexpr size_t PropertyAlignment =
      alignof(_sycl_device_binary_property_struct);
  constexpr size_t PaddedPropertyByteSize =
      (1 + ((PropertyByteSize - 1) / PropertyAlignment)) * PropertyAlignment;

  // Count the total number of property entries.
  size_t PropertyCount = 0;
  for (const auto &Vec : MergedVecs)
    PropertyCount += Vec->size();
  for (const auto &Map : MergedMaps)
    PropertyCount += Map->size();
  PropertyCount += MergedDevReqs.getPropertiesCount();

  // Count the bytes needed for the values and names of the properties.
  auto GetPropertyContentSize = [](const sycl_device_binary_property Prop) {
    return Prop->Type == SYCL_PROPERTY_TYPE_BYTE_ARRAY ? Prop->ValSize : 0;
  };
  size_t PropertyContentByteSize = 0;
  for (const auto &Vec : MergedVecs)
    for (const auto &Prop : *Vec)
      PropertyContentByteSize +=
          strlen(Prop->Name) + 1 + GetPropertyContentSize(Prop);
  for (const auto &Map : MergedMaps)
    for (const auto &PropIt : *Map)
      PropertyContentByteSize += strlen(PropIt.second->Name) + 1 +
                                 GetPropertyContentSize(PropIt.second);
  PropertyContentByteSize += MergedDevReqs.getPropertiesContentByteSize();

  const size_t PropertySectionSize = PropertyCount * PaddedPropertyByteSize;

  // Allocate the memory aligned to the property entry alignment.
#ifdef _MSC_VER
  // Note: MSVC does not implement std::aligned_alloc.
  Data = std::unique_ptr<char[], std::function<void(void *)>>(
      static_cast<char *>(_aligned_malloc(sizeof(char) * PropertySectionSize +
                                              PropertyContentByteSize,
                                          PropertyAlignment)),
      _aligned_free);
#else
  // std::aligned_alloc requires the allocation size to be a multiple of the
  // alignment, so we may over-allocate a little.
  const size_t AllocSize =
      sizeof(char) * PropertySectionSize + PropertyContentByteSize;
  const size_t AlignedAllocSize = (AllocSize + PropertyAlignment - 1) /
                                  PropertyAlignment * PropertyAlignment;
  Data = std::unique_ptr<char[], std::function<void(void *)>>(
      static_cast<char *>(
          std::aligned_alloc(PropertyAlignment, AlignedAllocSize)),
      std::free);
#endif

  auto NextFreeProperty =
      reinterpret_cast<sycl_device_binary_property>(Data.get());
  char *NextFreeContent = Data.get() + PropertySectionSize;

  auto CopyPropertiesVec =
      [&](const auto &Properties,
          RTDeviceBinaryImage::PropertyRange &TargetRange) {
        if (Properties.empty())
          return;
        TargetRange.Begin = NextFreeProperty;
        for (const sycl_device_binary_property Prop : Properties)
          copyProperty(NextFreeProperty, NextFreeContent, Prop);
        TargetRange.End = NextFreeProperty;
      };
  auto CopyPropertiesMap =
      [&](const auto &Properties,
          RTDeviceBinaryImage::PropertyRange &TargetRange) {
        if (Properties.empty())
          return;
        TargetRange.Begin = NextFreeProperty;
        for (const auto &PropIt : Properties)
          copyProperty(NextFreeProperty, NextFreeContent, PropIt.second);
        TargetRange.End = NextFreeProperty;
      };

  CopyPropertiesVec(MergedSpecConstants, SpecConstIDMap);
  CopyPropertiesVec(MergedSpecConstantsDefaultValues,
                    SpecConstDefaultValuesMap);
  CopyPropertiesVec(MergedKernelParamOptInfo, KernelParamOptInfo);
  CopyPropertiesVec(MergedAssertUsed, AssertUsed);
  CopyPropertiesVec(MergedDeviceGlobals, DeviceGlobals);
  CopyPropertiesVec(MergedHostPipes, HostPipes);
  CopyPropertiesVec(MergedVirtualFunctions, VirtualFunctions);
  CopyPropertiesVec(MergedImplicitLocalArg, ImplicitLocalArg);
  CopyPropertiesVec(MergedExportedSymbols, ExportedSymbols);
  CopyPropertiesVec(MergedRegisteredKernels, RegisteredKernels);

  CopyPropertiesMap(MergedDeviceLibReqMask, DeviceLibReqMask);
  CopyPropertiesMap(MergedProgramMetadata, ProgramMetadata);
  CopyPropertiesMap(MergedImportedSymbols, ImportedSymbols);
  CopyPropertiesMap(MergedMisc, Misc);

  // Special handling for new device requirements.
  {
    DeviceRequirements.Begin = NextFreeProperty;
    for (const auto &PropIt : MergedDevReqs.MergeMap)
      copyProperty(NextFreeProperty, NextFreeContent, PropIt.second);
    MergedDevReqs.writeAspectProperty(NextFreeProperty, NextFreeContent);
    MergedDeviceRequirements::writeStringSetProperty(
        MergedDevReqs.JointMatrix, "joint_matrix", NextFreeProperty,
        NextFreeContent);
    MergedDeviceRequirements::writeStringSetProperty(
        MergedDevReqs.JointMatrixMad, "joint_matrix_mad", NextFreeProperty,
        NextFreeContent);
    DeviceRequirements.End = NextFreeProperty;
  }
}

#ifndef SYCL_RT_ZSTD_NOT_AVAIABLE
CompressedRTDeviceBinaryImage::CompressedRTDeviceBinaryImage(
    sycl_device_binary CompressedBin)
    : RTDeviceBinaryImage() {

  // 'CompressedBin' is part of the executable image loaded into memory
  // which can't be modified easily. So, we need to make a copy of it.
  Bin = new sycl_device_binary_struct(*CompressedBin);

  // Get the decompressed size of the binary image.
  m_ImageSize = ZSTDCompressor::GetDecompressedSize(
      reinterpret_cast<const char *>(Bin->BinaryStart),
      static_cast<size_t>(Bin->BinaryEnd - Bin->BinaryStart));

  init(Bin);
}

void CompressedRTDeviceBinaryImage::Decompress() {

  size_t CompressedDataSize =
      static_cast<size_t>(Bin->BinaryEnd - Bin->BinaryStart);

  size_t DecompressedSize = 0;
  m_DecompressedData = ZSTDCompressor::DecompressBlob(
      reinterpret_cast<const char *>(Bin->BinaryStart), CompressedDataSize,
      DecompressedSize);

  Bin->BinaryStart =
      reinterpret_cast<const unsigned char *>(m_DecompressedData.get());
  Bin->BinaryEnd = Bin->BinaryStart + DecompressedSize;

  Bin->Format = ur::getBinaryImageFormat(Bin->BinaryStart, getSize());
  Format = static_cast<ur::DeviceBinaryType>(Bin->Format);
}

CompressedRTDeviceBinaryImage::~CompressedRTDeviceBinaryImage() {
  // De-allocate device binary struct.
  delete Bin;
  Bin = nullptr;
}
#endif // SYCL_RT_ZSTD_NOT_AVAIABLE

} // namespace detail
} // namespace _V1
} // namespace sycl
