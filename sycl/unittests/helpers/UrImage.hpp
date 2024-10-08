//==------------- UrImage.hpp --- UR mock image unit testing library -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/compiler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/ur.hpp>

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {
namespace unittest {
using namespace sycl::detail;

/// Convinience wrapper around _ur_device_binary_property_struct.
class UrProperty {
public:
  using NativeType = _sycl_device_binary_property_struct;

  /// Constructs a UR property.
  ///
  /// \param Name is a property name.
  /// \param Data is a vector of raw property value bytes.
  /// \param Type is one of ur_property_type values.
  UrProperty(const std::string &Name, std::vector<char> Data, uint32_t Type)
      : MName(Name), MData(std::move(Data)), MType(Type) {
    updateNativeType();
  }

  NativeType convertToNativeType() const { return MNative; }

  UrProperty(const UrProperty &Src) {
    MName = Src.MName;
    MData = Src.MData;
    MType = Src.MType;
    updateNativeType();
  }

  UrProperty &operator=(const UrProperty &Src) {
    MName = Src.MName;
    MData = Src.MData;
    MType = Src.MType;
    updateNativeType();
    return *this;
  }

private:
  void updateNativeType() {
    if (MType == SYCL_PROPERTY_TYPE_UINT32) {
      MNative = NativeType{const_cast<char *>(MName.c_str()), nullptr, MType,
                           *((uint32_t *)MData.data())};
    } else {
      MNative =
          NativeType{const_cast<char *>(MName.c_str()),
                     const_cast<char *>(MData.data()), MType, MData.size()};
    }
  }
  std::string MName;
  std::vector<char> MData;
  uint32_t MType;
  NativeType MNative;
};

/// Convinience wrapper for _ur_offload_entry_struct.
class UrOffloadEntry {
public:
  using NativeType = _sycl_offload_entry_struct;

  UrOffloadEntry(const std::string &Name, std::vector<char> Data, int32_t Flags)
      : MName(Name), MData(std::move(Data)), MFlags(Flags) {
    updateNativeType();
  }

  UrOffloadEntry(const UrOffloadEntry &Src) {
    MName = Src.MName;
    MData = Src.MData;
    MFlags = Src.MFlags;
    updateNativeType();
  }
  UrOffloadEntry &operator=(const UrOffloadEntry &Src) {
    MName = Src.MName;
    MData = Src.MData;
    MFlags = Src.MFlags;
    updateNativeType();
    return *this;
  }

  NativeType convertToNativeType() const { return MNative; }

private:
  void updateNativeType() {
    MNative = NativeType{
        const_cast<char *>(MData.data()), MName.data(), MData.size(), MFlags,
        0 // Reserved
    };
  }
  std::string MName;
  std::vector<char> MData;
  int32_t MFlags;
  NativeType MNative;
};

namespace internal {
// Content from this namespace shouldn't be used anywhere outside of this file

/// "native" data structures used by SYCL RT do not hold the data, but only
/// point to it. The data itself is embedded into the binary data sections in
/// real applications.
/// In unit-tests we mock those data structures and therefore we need to ensure
/// that the lifetime of the underlying data is correct and we won't perform
/// any illegal memory accesses in unit-tests.
template <typename T> class LifetimeExtender {
public:
  explicit LifetimeExtender(std::vector<T> Entries)
      : MMockEntries(std::move(Entries)) {
    MEntries.clear();
    std::transform(MMockEntries.begin(), MMockEntries.end(),
                   std::back_inserter(MEntries),
                   [](const T &Entry) { return Entry.convertToNativeType(); });
  }

  LifetimeExtender() = default;

  typename T::NativeType *begin() {
    if (MEntries.empty())
      return nullptr;

    return &*MEntries.begin();
  }
  typename T::NativeType *end() {
    if (MEntries.empty())
      return nullptr;

    return &*MEntries.rbegin() + 1;
  }

private:
  std::vector<T> MMockEntries;
  std::vector<typename T::NativeType> MEntries;
};

#ifdef __cpp_deduction_guides
template <typename T> LifetimeExtender(std::vector<T>) -> LifetimeExtender<T>;
#endif // __cpp_deduction_guides
} // namespace internal

/// Convenience wrapper for sycl_device_binary_property_set.
class UrPropertySet {
public:
  UrPropertySet() {
    // Most of unit-tests are statically linked with SYCL RT. On Linux and Mac
    // systems that causes incorrect RT installation directory detection, which
    // prevents proper loading of fallback libraries. See intel/llvm#6945
    //
    // Fallback libraries are automatically loaded and linked into device image
    // unless there is a special property attached to it or special env variable
    // is set which forces RT to skip fallback libraries.
    //
    // Setting this property here so unit-tests can be launched under any
    // environment.

    std::vector<char> Data(/* eight elements */ 8,
                           /* each element is zero */ 0);
    // Name doesn't matter here, it is not used by RT
    // Value must be an all-zero 32-bit mask, which would mean that no fallback
    // libraries are needed to be loaded.
    UrProperty DeviceLibReqMask("", Data, SYCL_PROPERTY_TYPE_UINT32);
    insert(__SYCL_PROPERTY_SET_DEVICELIB_REQ_MASK, std::move(DeviceLibReqMask));
  }

  /// Adds a new property to the set.
  ///
  /// \param Name is a property name. See ur.hpp for list of known names.
  /// \param Prop is a property value.
  void insert(const std::string &Name, UrProperty &&Props) {
    insert(Name, internal::LifetimeExtender{std::vector{std::move(Props)}});
  }

  /// Adds a new array of properties to the set.
  ///
  /// \param Name is a property array name. See ur.hpp for list of known names.
  /// \param Props is an array of property values.
  void insert(const std::string &Name, std::vector<UrProperty> &&Props) {
    insert(Name, internal::LifetimeExtender{std::move(Props)});
  }

  _sycl_device_binary_property_set_struct *begin() {
    if (MProperties.empty())
      return nullptr;
    return &*MProperties.begin();
  }

  _sycl_device_binary_property_set_struct *end() {
    if (MProperties.empty())
      return nullptr;
    return &*MProperties.rbegin() + 1;
  }

private:
  /// Adds a new array of properties to the set.
  ///
  /// \param Name is a property array name. See ur.hpp for list of known names.
  /// \param Props is an array of property values.
  void insert(const std::string &Name,
              internal::LifetimeExtender<UrProperty> Props) {
    MNames.push_back(Name);
    MMockProperties.push_back(std::move(Props));
    MProperties.push_back(_sycl_device_binary_property_set_struct{
        MNames.back().data(), MMockProperties.back().begin(),
        MMockProperties.back().end()});
  }

  std::vector<std::string> MNames;
  std::vector<internal::LifetimeExtender<UrProperty>> MMockProperties;
  std::vector<_sycl_device_binary_property_set_struct> MProperties;
};

/// Convenience wrapper around UR internal structures, that manages UR binary
/// image data lifecycle.
class UrImage {
private:
  /// Constructs an arbitrary device image.
  UrImage(uint16_t Version, uint8_t Kind, uint8_t Format,
          const std::string &DeviceTargetSpec,
          const std::string &CompileOptions, const std::string &LinkOptions,
          std::vector<char> &&Manifest, std::vector<unsigned char> &&Binary,
          internal::LifetimeExtender<UrOffloadEntry> OffloadEntries,
          UrPropertySet PropertySet)
      : MVersion(Version), MKind(Kind), MFormat(Format),
        MDeviceTargetSpec(DeviceTargetSpec), MCompileOptions(CompileOptions),
        MLinkOptions(LinkOptions), MManifest(std::move(Manifest)),
        MBinary(std::move(Binary)), MOffloadEntries(std::move(OffloadEntries)),
        MPropertySet(std::move(PropertySet)) {}

public:
  /// Constructs an arbitrary device image.
  UrImage(uint16_t Version, uint8_t Kind, uint8_t Format,
          const std::string &DeviceTargetSpec,
          const std::string &CompileOptions, const std::string &LinkOptions,
          std::vector<char> &&Manifest, std::vector<unsigned char> &&Binary,
          std::vector<UrOffloadEntry> &&OffloadEntries,
          UrPropertySet PropertySet)
      : UrImage(Version, Kind, Format, DeviceTargetSpec, CompileOptions,
                LinkOptions, std::move(Manifest), std::move(Binary),
                internal::LifetimeExtender(std::move(OffloadEntries)),
                std::move(PropertySet)) {}

  /// Constructs a SYCL device image of the latest version.
  UrImage(uint8_t Format, const std::string &DeviceTargetSpec,
          const std::string &CompileOptions, const std::string &LinkOptions,
          std::vector<unsigned char> &&Binary,
          std::vector<UrOffloadEntry> &&OffloadEntries,
          UrPropertySet PropertySet)
      : UrImage(SYCL_DEVICE_BINARY_VERSION,
                SYCL_DEVICE_BINARY_OFFLOAD_KIND_SYCL, Format, DeviceTargetSpec,
                CompileOptions, LinkOptions, {}, std::move(Binary),
                internal::LifetimeExtender(std::move(OffloadEntries)),
                std::move(PropertySet)) {}

  sycl_device_binary_struct convertToNativeType() {
    return sycl_device_binary_struct{
        MVersion,
        MKind,
        MFormat,
        MDeviceTargetSpec.c_str(),
        MCompileOptions.c_str(),
        MLinkOptions.c_str(),
        MManifest.empty() ? nullptr : &*MManifest.cbegin(),
        MManifest.empty() ? nullptr : &*MManifest.crbegin() + 1,
        &*MBinary.begin(),
        (&*MBinary.begin()) + MBinary.size(),
        MOffloadEntries.begin(),
        MOffloadEntries.end(),
        MPropertySet.begin(),
        MPropertySet.end(),
    };
  }
  const unsigned char *getBinaryPtr() { return &*MBinary.begin(); }

private:
  uint16_t MVersion;
  uint8_t MKind;
  uint8_t MFormat;
  std::string MDeviceTargetSpec;
  std::string MCompileOptions;
  std::string MLinkOptions;
  std::vector<char> MManifest;
  std::vector<unsigned char> MBinary;
  internal::LifetimeExtender<UrOffloadEntry> MOffloadEntries;
  UrPropertySet MPropertySet;
};

/// Convenience wrapper around sycl_device_binaries_struct, that manages mock
/// device images' lifecycle.
template <size_t __NumberOfImages> class UrImageArray {
public:
  static constexpr size_t NumberOfImages = __NumberOfImages;

  UrImageArray(UrImage *Imgs) {
    for (size_t Idx = 0; Idx < NumberOfImages; ++Idx)
      MNativeImages[Idx] = Imgs[Idx].convertToNativeType();

    MAllBinaries = sycl_device_binaries_struct{
        SYCL_DEVICE_BINARIES_VERSION,
        NumberOfImages,
        MNativeImages,
        nullptr, // not used, put here for compatibility with OpenMP
        nullptr, // not used, put here for compatibility with OpenMP
    };

    __sycl_register_lib(&MAllBinaries);
  }

  ~UrImageArray() { __sycl_unregister_lib(&MAllBinaries); }

private:
  sycl_device_binary_struct MNativeImages[NumberOfImages];
  sycl_device_binaries_struct MAllBinaries;
};

template <typename Func, uint32_t Idx = 0, typename... Ts>
std::enable_if_t<Idx == sizeof...(Ts)> iterate_tuple(Func &F,
                                                     std::tuple<Ts...> &Tuple) {
  return;
}
template <typename Func, uint32_t Idx = 0, typename... Ts>
    std::enable_if_t <
    Idx<sizeof...(Ts)> inline iterate_tuple(Func &F, std::tuple<Ts...> &Tuple) {
  const auto &Value = std::get<Idx>(Tuple);
  const char *Begin = reinterpret_cast<const char *>(&Value);
  const char *End = Begin + sizeof(Value);
  F(Idx, Begin, End);

  iterate_tuple<Func, Idx + 1, Ts...>(F, Tuple);
  return;
}

/// Utility function to create a single spec constant property.
///
/// \param ValData is a reference to blob array, that stores default values.
/// \param Name is a spec constant name.
/// \param IDs is a list of spec IDs.
/// \param Offsets is a list of offsets inside composite spec constant.
/// \param DefaultValues is a tuple of default values for composite spec const.
template <typename... T>
inline UrProperty makeSpecConstant(std::vector<char> &ValData,
                                   const std::string &Name,
                                   std::initializer_list<uint32_t> IDs,
                                   std::initializer_list<uint32_t> Offsets,
                                   std::tuple<T...> DefaultValues) {
  const size_t PropByteArraySize = sizeof...(T) * sizeof(uint32_t) * 3;
  std::vector<char> DescData;
  DescData.resize(8 + PropByteArraySize);
  std::uninitialized_copy(&PropByteArraySize, &PropByteArraySize + 8,
                          DescData.data());

  if (ValData.empty())
    ValData.resize(8); // Reserve first 8 bytes for array size.
  size_t PrevSize = ValData.size();

  {
    // Resize raw data blob to current size + offset of the last element + size
    // of the last element.
    ValData.resize(
        PrevSize + *std::prev(Offsets.end()) +
        sizeof(typename std::tuple_element<sizeof...(T) - 1,
                                           decltype(DefaultValues)>::type));
    // Update raw data array size
    uint64_t NewValSize = ValData.size();
    std::uninitialized_copy(&NewValSize, &NewValSize + sizeof(uint64_t),
                            ValData.data());
  }

  auto FillData = [PrevOffset = 0, PrevSize, &ValData, &IDs, &Offsets,
                   &DescData](uint32_t Idx, const char *Begin,
                              const char *End) mutable {
    const size_t Offset = 8 + Idx * sizeof(uint32_t) * 3;

    uint32_t ValSize = std::distance(Begin, End);
    const char *IDsBegin =
        reinterpret_cast<const char *>(&*std::next(IDs.begin(), Idx));
    const char *OffsetBegin =
        reinterpret_cast<const char *>(&*std::next(Offsets.begin(), Idx));
    const char *ValSizeBegin = reinterpret_cast<const char *>(&ValSize);

    std::uninitialized_copy(IDsBegin, IDsBegin + sizeof(uint32_t),
                            DescData.data() + Offset);
    std::uninitialized_copy(OffsetBegin, OffsetBegin + sizeof(uint32_t),
                            DescData.data() + Offset + sizeof(uint32_t));
    std::uninitialized_copy(ValSizeBegin, ValSizeBegin + sizeof(uint32_t),
                            DescData.data() + Offset + 2 * sizeof(uint32_t));
    std::uninitialized_copy(Begin, End, ValData.data() + PrevSize + PrevOffset);
    PrevOffset += *std::next(Offsets.begin(), Idx);
  };

  iterate_tuple(FillData, DefaultValues);

  UrProperty Prop{Name, DescData, SYCL_PROPERTY_TYPE_BYTE_ARRAY};

  return Prop;
}

/// Utility function to mark kernel as the one using assert
inline void setKernelUsesAssert(const std::vector<std::string> &Names,
                                UrPropertySet &Set) {
  std::vector<UrProperty> Value;
  for (const std::string &N : Names)
    Value.push_back({N, {0, 0, 0, 0}, SYCL_PROPERTY_TYPE_UINT32});
  Set.insert(__SYCL_PROPERTY_SET_SYCL_ASSERT_USED, std::move(Value));
}

/// Utility function to add specialization constants to property set.
///
/// This function overrides the default spec constant values.
inline void addSpecConstants(std::vector<UrProperty> &&SpecConstants,
                             std::vector<char> ValData, UrPropertySet &Props) {
  Props.insert(__SYCL_PROPERTY_SET_SPEC_CONST_MAP, std::move(SpecConstants));

  UrProperty Prop{"all", std::move(ValData), SYCL_PROPERTY_TYPE_BYTE_ARRAY};

  Props.insert(__SYCL_PROPERTY_SET_SPEC_CONST_DEFAULT_VALUES_MAP,
               std::move(Prop));
}

/// Utility function to add ESIMD kernel flag to property set.
inline void addESIMDFlag(UrPropertySet &Props) {
  std::vector<char> ValData(sizeof(uint32_t));
  ValData[0] = 1;
  UrProperty Prop{"isEsimdImage", ValData, SYCL_PROPERTY_TYPE_UINT32};

  Props.insert(__SYCL_PROPERTY_SET_SYCL_MISC_PROP, std::move(Prop));
}

/// Utility function to generate offload entries for kernels without arguments.
inline std::vector<UrOffloadEntry>
makeEmptyKernels(std::initializer_list<std::string> KernelNames) {
  std::vector<UrOffloadEntry> Entries;

  for (const auto &Name : KernelNames) {
    UrOffloadEntry E{Name, {}, 0};
    Entries.push_back(std::move(E));
  }
  return Entries;
}

/// Utility function to create a kernel params optimization info property.
///
/// \param Name is a property name.
/// \param NumArgs is a total number of arguments of a kernel.
/// \param ElimArgMask is a bit mask of eliminated kernel arguments IDs.
inline UrProperty
makeKernelParamOptInfo(const std::string &Name, const size_t NumArgs,
                       const std::vector<unsigned char> &ElimArgMask) {
  const size_t BYTES_FOR_SIZE = 8;
  auto *EAMSizePtr = reinterpret_cast<const unsigned char *>(&NumArgs);
  std::vector<char> DescData;
  DescData.resize(BYTES_FOR_SIZE + ElimArgMask.size());
  std::uninitialized_copy(EAMSizePtr, EAMSizePtr + BYTES_FOR_SIZE,
                          DescData.data());
  std::uninitialized_copy(ElimArgMask.begin(), ElimArgMask.end(),
                          DescData.data() + BYTES_FOR_SIZE);

  UrProperty Prop{Name, DescData, SYCL_PROPERTY_TYPE_BYTE_ARRAY};

  return Prop;
}

/// Utility function to create a device global info property.
///
/// \param Name is the name of the device global name.
/// \param TypeSize is the size of the underlying type in the device global.
/// \param DeviceImageScoped is whether the device global was device image scope
/// decorated.
inline UrProperty makeDeviceGlobalInfo(const std::string &Name,
                                       const uint32_t TypeSize,
                                       const std::uint32_t DeviceImageScoped) {
  constexpr size_t BYTES_FOR_SIZE = 8;
  const std::uint64_t BytesForArgs = 2 * sizeof(std::uint32_t);
  std::vector<char> DescData;
  DescData.resize(BYTES_FOR_SIZE + BytesForArgs);
  std::memcpy(DescData.data(), &BytesForArgs, sizeof(BytesForArgs));
  std::memcpy(DescData.data() + BYTES_FOR_SIZE, &TypeSize, sizeof(TypeSize));
  std::memcpy(DescData.data() + BYTES_FOR_SIZE + sizeof(TypeSize),
              &DeviceImageScoped, sizeof(DeviceImageScoped));

  UrProperty Prop{Name, DescData, SYCL_PROPERTY_TYPE_BYTE_ARRAY};

  return Prop;
}

/// Utility function to create a host pipe info property.
///
/// \param Name is the name of the hostpipe name.
/// \param TypeSize is the size of the underlying type in the hostpipe.
/// decorated.
inline UrProperty makeHostPipeInfo(const std::string &Name,
                                   const uint32_t TypeSize) {
  constexpr size_t BYTES_FOR_SIZE = 8;
  const std::uint64_t BytesForArgs = sizeof(std::uint32_t);
  std::vector<char> DescData;
  DescData.resize(BYTES_FOR_SIZE + BytesForArgs);
  std::memcpy(DescData.data(), &BytesForArgs, sizeof(BytesForArgs));
  std::memcpy(DescData.data() + BYTES_FOR_SIZE, &TypeSize, sizeof(TypeSize));

  UrProperty Prop{Name, DescData, SYCL_PROPERTY_TYPE_BYTE_ARRAY};

  return Prop;
}

/// Utility function to add aspects to property set.
inline UrProperty makeAspectsProp(const std::vector<sycl::aspect> &Aspects) {
  const size_t BYTES_FOR_SIZE = 8;
  std::vector<char> ValData(BYTES_FOR_SIZE +
                            Aspects.size() * sizeof(sycl::aspect));
  uint64_t ValDataSize = ValData.size();
  std::uninitialized_copy(&ValDataSize, &ValDataSize + sizeof(uint64_t),
                          ValData.data());
  auto *AspectsPtr = reinterpret_cast<const unsigned char *>(&Aspects[0]);
  std::uninitialized_copy(AspectsPtr, AspectsPtr + Aspects.size(),
                          ValData.data() + BYTES_FOR_SIZE);
  return {"aspects", ValData, SYCL_PROPERTY_TYPE_BYTE_ARRAY};
}

inline UrProperty makeReqdWGSizeProp(const std::vector<int> &ReqdWGSize) {
  const size_t BYTES_FOR_SIZE = 8;
  std::vector<char> ValData(BYTES_FOR_SIZE + ReqdWGSize.size() * sizeof(int));
  uint64_t ValDataSize = ValData.size();
  std::uninitialized_copy(&ValDataSize, &ValDataSize + sizeof(uint64_t),
                          ValData.data());
  auto *ReqdWGSizePtr = reinterpret_cast<const unsigned char *>(&ReqdWGSize[0]);
  std::uninitialized_copy(ReqdWGSizePtr,
                          ReqdWGSizePtr + ReqdWGSize.size() * sizeof(int),
                          ValData.data() + BYTES_FOR_SIZE);
  return {"reqd_work_group_size", ValData, SYCL_PROPERTY_TYPE_BYTE_ARRAY};
}

inline void
addDeviceRequirementsProps(UrPropertySet &Props,
                           const std::vector<sycl::aspect> &Aspects,
                           const std::vector<int> &ReqdWGSize = {}) {
  std::vector<UrProperty> Value{makeAspectsProp(Aspects)};
  if (!ReqdWGSize.empty())
    Value.push_back(makeReqdWGSizeProp(ReqdWGSize));
  Props.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS, std::move(Value));
}

inline UrImage
generateDefaultImage(std::initializer_list<std::string> KernelNames) {
  UrPropertySet PropSet;

  std::string Combined;
  for (auto it = KernelNames.begin(); it != KernelNames.end(); ++it) {
    if (it != KernelNames.begin())
      Combined += ", ";
    Combined += *it;
  }
  std::vector<unsigned char> Bin(Combined.begin(), Combined.end());
  Bin.push_back(0);

  std::vector<UrOffloadEntry> Entries = makeEmptyKernels(KernelNames);

  UrImage Img{SYCL_DEVICE_BINARY_TYPE_SPIRV,       // Format
              __SYCL_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                  // Compile options
              "",                                  // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

} // namespace unittest
} // namespace _V1
} // namespace sycl
