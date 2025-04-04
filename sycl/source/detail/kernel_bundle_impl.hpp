//==------- kernel_bundle_impl.hpp - SYCL kernel_bundle_impl ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/device_image_impl.hpp>
#include <detail/kernel_compiler/kernel_compiler_opencl.hpp>
#include <detail/kernel_compiler/kernel_compiler_sycl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/persistent_device_code_cache.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/device.hpp>
#include <sycl/kernel_bundle.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_set>
#include <vector>

#include "split_string.hpp"

namespace sycl {
inline namespace _V1 {
namespace detail {

static bool checkAllDevicesAreInContext(const std::vector<device> &Devices,
                                        const context &Context) {
  return std::all_of(
      Devices.begin(), Devices.end(), [&Context](const device &Dev) {
        return getSyclObjImpl(Context)->isDeviceValid(getSyclObjImpl(Dev));
      });
}

static bool checkAllDevicesHaveAspect(const std::vector<device> &Devices,
                                      aspect Aspect) {
  return std::all_of(Devices.begin(), Devices.end(),
                     [&Aspect](const device &Dev) { return Dev.has(Aspect); });
}

namespace syclex = sycl::ext::oneapi::experimental;

class kernel_impl;

/// The class is an impl counterpart of the sycl::kernel_bundle.
// It provides an access and utilities to manage set of sycl::device_images
// objects.
class kernel_bundle_impl {

  using SpecConstMapT = std::map<std::string, std::vector<unsigned char>>;

  void common_ctor_checks() const {
    const bool AllDevicesInTheContext =
        checkAllDevicesAreInContext(MDevices, MContext);
    if (MDevices.empty() || !AllDevicesInTheContext)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are associated with the context or "
          "vector of devices is empty");

    if (bundle_state::input == MState &&
        !checkAllDevicesHaveAspect(MDevices, aspect::online_compiler))
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices have aspect::online_compiler");

    if (bundle_state::object == MState &&
        !checkAllDevicesHaveAspect(MDevices, aspect::online_linker))
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices have aspect::online_linker");
  }

public:
  kernel_bundle_impl(context Ctx, std::vector<device> Devs, bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, State);
    fillUniqueDeviceImages();
  }

  // Interop constructor used by make_kernel
  kernel_bundle_impl(context Ctx, std::vector<device> Devs)
      : MContext(Ctx), MDevices(Devs), MState(bundle_state::executable) {
    if (!checkAllDevicesAreInContext(Devs, Ctx))
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are associated with the context or "
          "vector of devices is empty");
    MIsInterop = true;
  }

  // Interop constructor
  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     device_image_plain &DevImage)
      : kernel_bundle_impl(Ctx, Devs) {
    MDeviceImages.emplace_back(DevImage);
    MUniqueDeviceImages.emplace_back(DevImage);
  }

  // Matches sycl::build and sycl::compile
  // Have one constructor because sycl::build and sycl::compile have the same
  // signature
  kernel_bundle_impl(const kernel_bundle<bundle_state::input> &InputBundle,
                     std::vector<device> Devs, const property_list &PropList,
                     bundle_state TargetState)
      : MContext(InputBundle.get_context()), MDevices(std::move(Devs)),
        MState(TargetState) {

    const std::shared_ptr<kernel_bundle_impl> &InputBundleImpl =
        getSyclObjImpl(InputBundle);
    MSpecConstValues = InputBundleImpl->get_spec_const_map_ref();

    const std::vector<device> &InputBundleDevices =
        InputBundleImpl->get_devices();
    const bool AllDevsAssociatedWithInputBundle =
        std::all_of(MDevices.begin(), MDevices.end(),
                    [&InputBundleDevices](const device &Dev) {
                      return InputBundleDevices.end() !=
                             std::find(InputBundleDevices.begin(),
                                       InputBundleDevices.end(), Dev);
                    });
    if (MDevices.empty() || !AllDevsAssociatedWithInputBundle)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are in the set of associated "
          "devices for input bundle or vector of devices is empty");

    for (const DevImgPlainWithDeps &DevImgWithDeps :
         InputBundleImpl->MDeviceImages) {
      // Skip images which are not compatible with devices provided
      if (std::none_of(MDevices.begin(), MDevices.end(),
                       [&DevImgWithDeps](const device &Dev) {
                         return getSyclObjImpl(DevImgWithDeps.getMain())
                             ->compatible_with_device(Dev);
                       }))
        continue;

      switch (TargetState) {
      case bundle_state::object: {
        DevImgPlainWithDeps CompiledImgWithDeps =
            detail::ProgramManager::getInstance().compile(DevImgWithDeps,
                                                          MDevices, PropList);

        MUniqueDeviceImages.insert(MUniqueDeviceImages.end(),
                                   CompiledImgWithDeps.begin(),
                                   CompiledImgWithDeps.end());
        MDeviceImages.push_back(std::move(CompiledImgWithDeps));
        break;
      }

      case bundle_state::executable: {
        device_image_plain BuiltImg =
            detail::ProgramManager::getInstance().build(DevImgWithDeps,
                                                        MDevices, PropList);
        MDeviceImages.emplace_back(BuiltImg);
        MUniqueDeviceImages.emplace_back(BuiltImg);
        break;
      }
      case bundle_state::input:
      case bundle_state::ext_oneapi_source:
        throw exception(make_error_code(errc::runtime),
                        "Internal error. The target state should not be input "
                        "or ext_oneapi_source");
        break;
      }
    }
    removeDuplicateImages();
  }

  // Matches sycl::link
  kernel_bundle_impl(
      const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
      std::vector<device> Devs, const property_list &PropList)
      : MDevices(std::move(Devs)), MState(bundle_state::executable) {

    if (MDevices.empty())
      throw sycl::exception(make_error_code(errc::invalid),
                            "Vector of devices is empty");

    if (ObjectBundles.empty())
      return;

    MContext = ObjectBundles[0].get_context();
    for (size_t I = 1; I < ObjectBundles.size(); ++I) {
      if (ObjectBundles[I].get_context() != MContext)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Not all input bundles have the same associated context");
    }

    // Check if any of the devices in devs are not in the set of associated
    // devices for any of the bundles in ObjectBundles
    const bool AllDevsAssociatedWithInputBundles = std::all_of(
        MDevices.begin(), MDevices.end(), [&ObjectBundles](const device &Dev) {
          // Number of devices is expected to be small
          return std::all_of(
              ObjectBundles.begin(), ObjectBundles.end(),
              [&Dev](const kernel_bundle<bundle_state::object> &KernelBundle) {
                const std::vector<device> &BundleDevices =
                    getSyclObjImpl(KernelBundle)->get_devices();
                return BundleDevices.end() != std::find(BundleDevices.begin(),
                                                        BundleDevices.end(),
                                                        Dev);
              });
        });
    if (!AllDevsAssociatedWithInputBundles)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices are in the set of associated "
                            "devices for input bundles");

    // TODO: Unify with c'tor for sycl::compile and sycl::build by calling
    // sycl::join on vector of kernel_bundles

    // The loop below just links each device image separately, not linking any
    // two device images together. This is correct so long as each device image
    // has no unresolved symbols. That's the case when device images are created
    // from generic SYCL APIs. There's no way in generic SYCL to create a kernel
    // which references an undefined symbol. If we decide in the future to allow
    // a backend interop API to create a "sycl::kernel_bundle" that references
    // undefined symbols, then the logic in this loop will need to be changed.
    for (const kernel_bundle<bundle_state::object> &ObjectBundle :
         ObjectBundles) {
      for (const DevImgPlainWithDeps &DeviceImageWithDeps :
           getSyclObjImpl(ObjectBundle)->MDeviceImages) {

        // Skip images which are not compatible with devices provided
        if (std::none_of(MDevices.begin(), MDevices.end(),
                         [&DeviceImageWithDeps](const device &Dev) {
                           return getSyclObjImpl(DeviceImageWithDeps.getMain())
                               ->compatible_with_device(Dev);
                         }))
          continue;

        std::vector<device_image_plain> LinkedResults =
            detail::ProgramManager::getInstance().link(DeviceImageWithDeps,
                                                       MDevices, PropList);
        MDeviceImages.insert(MDeviceImages.end(), LinkedResults.begin(),
                             LinkedResults.end());
        MUniqueDeviceImages.insert(MUniqueDeviceImages.end(),
                                   LinkedResults.begin(), LinkedResults.end());
      }
    }
    removeDuplicateImages();

    for (const kernel_bundle<bundle_state::object> &Bundle : ObjectBundles) {
      const KernelBundleImplPtr &BundlePtr = getSyclObjImpl(Bundle);
      for (const std::pair<const std::string, std::vector<unsigned char>>
               &SpecConst : BundlePtr->MSpecConstValues) {
        MSpecConstValues[SpecConst.first] = SpecConst.second;
      }
    }
  }

  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     const std::vector<kernel_id> &KernelIDs,
                     bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, KernelIDs, State);
    fillUniqueDeviceImages();
  }

  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     const DevImgSelectorImpl &Selector, bundle_state State)
      : MContext(std::move(Ctx)), MDevices(std::move(Devs)), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, Selector, State);
    fillUniqueDeviceImages();
  }

  // C'tor matches sycl::join API
  kernel_bundle_impl(const std::vector<detail::KernelBundleImplPtr> &Bundles,
                     bundle_state State)
      : MState(State) {
    if (Bundles.empty())
      return;

    MContext = Bundles[0]->MContext;
    MDevices = Bundles[0]->MDevices;
    for (size_t I = 1; I < Bundles.size(); ++I) {
      if (Bundles[I]->MContext != MContext)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Not all input bundles have the same associated context.");
      if (Bundles[I]->MDevices != MDevices)
        throw sycl::exception(
            make_error_code(errc::invalid),
            "Not all input bundles have the same set of associated devices.");
    }

    for (const detail::KernelBundleImplPtr &Bundle : Bundles) {

      MDeviceImages.insert(MDeviceImages.end(), Bundle->MDeviceImages.begin(),
                           Bundle->MDeviceImages.end());
    }

    fillUniqueDeviceImages();

    if (get_bundle_state() == bundle_state::input) {
      // Copy spec constants values from the device images.
      auto MergeSpecConstants = [this](const device_image_plain &Img) {
        const detail::DeviceImageImplPtr &ImgImpl = getSyclObjImpl(Img);
        const std::map<std::string,
                       std::vector<device_image_impl::SpecConstDescT>>
            &SpecConsts = ImgImpl->get_spec_const_data_ref();
        const std::vector<unsigned char> &Blob =
            ImgImpl->get_spec_const_blob_ref();
        for (const std::pair<const std::string,
                             std::vector<device_image_impl::SpecConstDescT>>
                 &SpecConst : SpecConsts) {
          if (SpecConst.second.front().IsSet)
            set_specialization_constant_raw_value(
                SpecConst.first.c_str(),
                Blob.data() + SpecConst.second.front().BlobOffset,
                SpecConst.second.back().CompositeOffset +
                    SpecConst.second.back().Size);
        }
      };
      std::for_each(begin(), end(), MergeSpecConstants);
    }

    for (const detail::KernelBundleImplPtr &Bundle : Bundles) {
      for (const std::pair<const std::string, std::vector<unsigned char>>
               &SpecConst : Bundle->MSpecConstValues) {
        set_specialization_constant_raw_value(SpecConst.first.c_str(),
                                              SpecConst.second.data(),
                                              SpecConst.second.size());
      }
    }
  }

  using include_pairs_t =
      std::vector<std::pair<std::string /* name */, std::string /* content */>>;
  // oneapi_ext_kernel_compiler
  // construct from source string
  kernel_bundle_impl(const context &Context, syclex::source_language Lang,
                     const std::string &Src, include_pairs_t IncludePairsVec)
      : MContext(Context), MDevices(Context.get_devices()),
        MState(bundle_state::ext_oneapi_source), MLanguage(Lang), MSource(Src),
        MIncludePairs(IncludePairsVec) {}

  // oneapi_ext_kernel_compiler
  // construct from source bytes
  kernel_bundle_impl(const context &Context, syclex::source_language Lang,
                     const std::vector<std::byte> &Bytes)
      : MContext(Context), MDevices(Context.get_devices()),
        MState(bundle_state::ext_oneapi_source), MLanguage(Lang),
        MSource(Bytes) {}

  // oneapi_ext_kernel_compiler
  // interop constructor
  kernel_bundle_impl(context Ctx, std::vector<device> Devs,
                     device_image_plain &DevImage,
                     std::vector<std::string> KNames,
                     syclex::source_language Lang)
      : kernel_bundle_impl(Ctx, Devs, DevImage) {
    MState = bundle_state::executable;
    MKernelNames = std::move(KNames);
    MLanguage = Lang;
  }

  // oneapi_ext_kernel_compiler
  // program manager integration, only for sycl language
  kernel_bundle_impl(
      context Ctx, std::vector<device> Devs,
      const std::vector<kernel_id> &KernelIDs,
      std::vector<std::string> &&KernelNames,
      std::unordered_map<std::string, std::string> &&MangledKernelNames,
      std::vector<std::string> &&DeviceGlobalNames,
      std::vector<std::unique_ptr<std::byte[]>> &&DeviceGlobalAllocations,
      sycl_device_binaries Binaries, std::string &&Prefix,
      syclex::source_language Lang)
      : kernel_bundle_impl(std::move(Ctx), std::move(Devs), KernelIDs,
                           bundle_state::executable) {
    assert(Lang == syclex::source_language::sycl);
    // Mark this bundle explicitly as "interop" to ensure that its kernels are
    // enqueued with the info from the kernel object passed by the application,
    // cf. `enqueueImpKernel` in `commands.cpp`. While runtime-compiled kernels
    // loaded via the program manager have `kernel_id`s, they can't be looked up
    // from the (unprefixed) kernel name.
    MIsInterop = true;
    MKernelNames = std::move(KernelNames);
    MMangledKernelNames = std::move(MangledKernelNames);
    MDeviceGlobalNames = std::move(DeviceGlobalNames);
    MDeviceGlobalAllocations = std::move(DeviceGlobalAllocations);
    MDeviceBinaries = Binaries;
    MPrefix = std::move(Prefix);
    MLanguage = Lang;
  }

  std::string trimXsFlags(std::string &str) {
    // Trim first and last quote if they exist, but no others.
    char EncounteredQuote = '\0';
    auto Start = std::find_if(str.begin(), str.end(), [&](char c) {
      if (!EncounteredQuote && (c == '\'' || c == '"')) {
        EncounteredQuote = c;
        return false;
      }
      return !std::isspace(c);
    });
    auto End = std::find_if(str.rbegin(), str.rend(), [&](char c) {
                 if (c == EncounteredQuote) {
                   EncounteredQuote = '\0';
                   return false;
                 }
                 return !std::isspace(c);
               }).base();
    if (Start != std::end(str) && End != std::begin(str) && Start < End) {
      return std::string(Start, End);
    }

    return "";
  }

  std::string extractXsFlags(const std::vector<std::string> &BuildOptions) {
    std::stringstream SS;
    for (std::string Option : BuildOptions) {
      auto Where = Option.find("-Xs");
      if (Where != std::string::npos) {
        Where += 3;
        std::string Flags = Option.substr(Where);
        SS << trimXsFlags(Flags) << " ";
      }
    }
    return SS.str();
  }

  bool
  extKernelCompilerFetchFromCache(const std::vector<device> Devices,
                                  const std::vector<std::string> &BuildOptions,
                                  const std::string &SourceStr,
                                  ur_program_handle_t &UrProgram) {
    using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
    const ContextImplPtr &ContextImpl = getSyclObjImpl(MContext);
    const AdapterPtr &Adapter = ContextImpl->getAdapter();

    std::string UserArgs = syclex::detail::userArgsAsString(BuildOptions);

    std::vector<ur_device_handle_t> DeviceHandles;
    std::transform(
        Devices.begin(), Devices.end(), std::back_inserter(DeviceHandles),
        [](const device &Dev) { return getSyclObjImpl(Dev)->getHandleRef(); });

    std::vector<const uint8_t *> Binaries;
    std::vector<size_t> Lengths;
    std::vector<std::vector<char>> BinProgs =
        PersistentDeviceCodeCache::getCompiledKernelFromDisc(Devices, UserArgs,
                                                             SourceStr);
    if (BinProgs.empty()) {
      return false;
    }
    for (auto &BinProg : BinProgs) {
      Binaries.push_back((uint8_t *)(BinProg.data()));
      Lengths.push_back(BinProg.size());
    }

    ur_program_properties_t Properties = {};
    Properties.stype = UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES;
    Properties.pNext = nullptr;
    Properties.count = 0;
    Properties.pMetadatas = nullptr;

    Adapter->call<UrApiKind::urProgramCreateWithBinary>(
        ContextImpl->getHandleRef(), DeviceHandles.size(), DeviceHandles.data(),
        Lengths.data(), Binaries.data(), &Properties, &UrProgram);

    return true;
  }

  std::shared_ptr<kernel_bundle_impl>
  build_from_source(const std::vector<device> Devices,
                    const std::vector<std::string> &BuildOptions,
                    std::string *LogPtr,
                    const std::vector<std::string> &RegisteredKernelNames) {
    assert(MState == bundle_state::ext_oneapi_source &&
           "bundle_state::ext_oneapi_source required");

    using ContextImplPtr = std::shared_ptr<sycl::detail::context_impl>;
    const ContextImplPtr &ContextImpl = getSyclObjImpl(MContext);
    const AdapterPtr &Adapter = ContextImpl->getAdapter();

    std::vector<ur_device_handle_t> DeviceVec;
    DeviceVec.reserve(Devices.size());
    for (const auto &SyclDev : Devices) {
      const DeviceImplPtr &DevImpl = getSyclObjImpl(SyclDev);
      if (!ContextImpl->hasDevice(DevImpl)) {
        throw sycl::exception(make_error_code(errc::invalid),
                              "device not part of kernel_bundle context");
      }
      if (!DevImpl->extOneapiCanCompile(MLanguage)) {
        // This error cannot not be exercised in the current implementation, as
        // compatibility with a source language depends on the backend's
        // capabilities and all devices in one context share the same backend in
        // the current implementation, so this would lead to an error already
        // during construction of the source bundle.
        throw sycl::exception(make_error_code(errc::invalid),
                              "device does not support source language");
      }
      ur_device_handle_t Dev = DevImpl->getHandleRef();
      DeviceVec.push_back(Dev);
    }

    if (MLanguage == syclex::source_language::sycl) {
      // Build device images via the program manager.
      const std::string &SourceStr = std::get<std::string>(MSource);
      std::ostringstream SourceExt;
      if (!RegisteredKernelNames.empty()) {
        SourceExt << SourceStr << '\n';

        auto EmitEntry =
            [&SourceExt](const std::string &Name) -> std::ostringstream & {
          SourceExt << "  {\"" << Name << "\", " << Name << "}";
          return SourceExt;
        };

        SourceExt << "[[__sycl_detail__::__registered_kernels__(\n";
        for (auto It = RegisteredKernelNames.begin(),
                  SecondToLast = RegisteredKernelNames.end() - 1;
             It != SecondToLast; ++It) {
          EmitEntry(*It) << ",\n";
        }
        EmitEntry(RegisteredKernelNames.back()) << "\n";
        SourceExt << ")]];\n";
      }

      auto [Binaries, Prefix] = syclex::detail::SYCL_JIT_Compile(
          RegisteredKernelNames.empty() ? SourceStr : SourceExt.str(),
          MIncludePairs, BuildOptions, LogPtr);

      auto &PM = detail::ProgramManager::getInstance();
      PM.addImages(Binaries);

      std::vector<kernel_id> KernelIDs;
      std::vector<std::string> KernelNames;
      std::unordered_map<std::string, std::string> MangledKernelNames;

      std::unordered_set<std::string> DeviceGlobalIDSet;
      std::vector<std::string> DeviceGlobalIDVec;
      std::vector<std::string> DeviceGlobalNames;
      std::vector<std::unique_ptr<std::byte[]>> DeviceGlobalAllocations;

      for (const auto &KernelID : PM.getAllSYCLKernelIDs()) {
        std::string_view KernelName{KernelID.get_name()};
        if (KernelName.find(Prefix) == 0) {
          KernelIDs.push_back(KernelID);
          KernelName.remove_prefix(Prefix.length());
          KernelNames.emplace_back(KernelName);
          static constexpr std::string_view SYCLKernelMarker{"__sycl_kernel_"};
          if (KernelName.find(SYCLKernelMarker) == 0) {
            // extern "C" declaration, implicitly register kernel without the
            // marker.
            std::string_view KernelNameWithoutMarker{KernelName};
            KernelNameWithoutMarker.remove_prefix(SYCLKernelMarker.length());
            MangledKernelNames.emplace(KernelNameWithoutMarker, KernelName);
          }
        }
      }

      for (const auto *RawImg : PM.getRawDeviceImages(KernelIDs)) {
        // Mangled names.
        for (const sycl_device_binary_property &RKProp :
             RawImg->getRegisteredKernels()) {

          auto BA = DeviceBinaryProperty(RKProp).asByteArray();
          auto MangledNameLen = BA.consume<uint64_t>() / 8 /*bits in a byte*/;
          std::string_view MangledName{
              reinterpret_cast<const char *>(BA.begin()), MangledNameLen};
          MangledKernelNames.emplace(RKProp->Name, MangledName);
        }

        // Device globals.
        for (const auto &DeviceGlobalProp : RawImg->getDeviceGlobals()) {
          std::string_view DeviceGlobalName{DeviceGlobalProp->Name};
          assert(DeviceGlobalName.find(Prefix) == 0);
          bool Inserted = false;
          std::tie(std::ignore, Inserted) =
              DeviceGlobalIDSet.emplace(DeviceGlobalName);
          if (Inserted) {
            DeviceGlobalIDVec.emplace_back(DeviceGlobalName);
            DeviceGlobalName.remove_prefix(Prefix.length());
            DeviceGlobalNames.emplace_back(DeviceGlobalName);
          }
        }
      }

      // Device globals are usually statically allocated and registered in the
      // integration footer, which we don't have in the RTC context. Instead, we
      // dynamically allocate storage tied to the executable kernel bundle.
      for (DeviceGlobalMapEntry *DeviceGlobalEntry :
           PM.getDeviceGlobalEntries(DeviceGlobalIDVec)) {

        size_t AllocSize = DeviceGlobalEntry->MDeviceGlobalTSize; // init value
        if (!DeviceGlobalEntry->MIsDeviceImageScopeDecorated) {
          // Consider storage for device USM pointer.
          AllocSize += sizeof(void *);
        }
        auto Alloc = std::make_unique<std::byte[]>(AllocSize);
        std::string_view DeviceGlobalName{DeviceGlobalEntry->MUniqueId};
        PM.addOrInitDeviceGlobalEntry(Alloc.get(), DeviceGlobalName.data());
        DeviceGlobalAllocations.push_back(std::move(Alloc));

        // Drop the RTC prefix from the entry's symbol name. Note that the PM
        // still manages this device global under its prefixed name.
        assert(DeviceGlobalName.find(Prefix) == 0);
        DeviceGlobalName.remove_prefix(Prefix.length());
        DeviceGlobalEntry->MUniqueId = DeviceGlobalName;
      }

      return std::make_shared<kernel_bundle_impl>(
          MContext, MDevices, KernelIDs, std::move(KernelNames),
          std::move(MangledKernelNames), std::move(DeviceGlobalNames),
          std::move(DeviceGlobalAllocations), Binaries, std::move(Prefix),
          MLanguage);
    }

    ur_program_handle_t UrProgram = nullptr;
    // SourceStrPtr will be null when source is Spir-V bytes.
    const std::string *SourceStrPtr = std::get_if<std::string>(&MSource);
    bool FetchedFromCache = false;
    if (PersistentDeviceCodeCache::isEnabled() && SourceStrPtr) {
      FetchedFromCache = extKernelCompilerFetchFromCache(
          Devices, BuildOptions, *SourceStrPtr, UrProgram);
    }

    if (!FetchedFromCache) {
      const auto spirv = [&]() -> std::vector<uint8_t> {
        if (MLanguage == syclex::source_language::opencl) {
          // if successful, the log is empty. if failed, throws an error with
          // the compilation log.
          std::vector<uint32_t> IPVersionVec(Devices.size());
          std::transform(DeviceVec.begin(), DeviceVec.end(),
                         IPVersionVec.begin(), [&](ur_device_handle_t d) {
                           uint32_t ipVersion = 0;
                           Adapter->call<UrApiKind::urDeviceGetInfo>(
                               d, UR_DEVICE_INFO_IP_VERSION, sizeof(uint32_t),
                               &ipVersion, nullptr);
                           return ipVersion;
                         });
          return syclex::detail::OpenCLC_to_SPIRV(*SourceStrPtr, IPVersionVec,
                                                  BuildOptions, LogPtr);
        }
        if (MLanguage == syclex::source_language::spirv) {
          const auto &SourceBytes = std::get<std::vector<std::byte>>(MSource);
          std::vector<uint8_t> Result(SourceBytes.size());
          std::transform(SourceBytes.cbegin(), SourceBytes.cend(),
                         Result.begin(),
                         [](std::byte B) { return static_cast<uint8_t>(B); });
          return Result;
        }
        throw sycl::exception(
            make_error_code(errc::invalid),
            "SYCL C++, OpenCL C and SPIR-V are the only supported "
            "languages at this time");
      }();

      Adapter->call<UrApiKind::urProgramCreateWithIL>(
          ContextImpl->getHandleRef(), spirv.data(), spirv.size(), nullptr,
          &UrProgram);
      // program created by urProgramCreateWithIL is implicitly retained.
      if (UrProgram == nullptr)
        throw sycl::exception(
            sycl::make_error_code(errc::invalid),
            "urProgramCreateWithIL resulted in a null program handle.");

    } // if(!FetchedFromCache)

    std::string XsFlags = extractXsFlags(BuildOptions);
    auto Res = Adapter->call_nocheck<UrApiKind::urProgramBuildExp>(
        UrProgram, DeviceVec.size(), DeviceVec.data(), XsFlags.c_str());
    if (Res == UR_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      Res = Adapter->call_nocheck<UrApiKind::urProgramBuild>(
          ContextImpl->getHandleRef(), UrProgram, XsFlags.c_str());
    }
    Adapter->checkUrResult<errc::build>(Res);

    // Get the number of kernels in the program.
    size_t NumKernels;
    Adapter->call<UrApiKind::urProgramGetInfo>(
        UrProgram, UR_PROGRAM_INFO_NUM_KERNELS, sizeof(size_t), &NumKernels,
        nullptr);

    // Get the kernel names.
    size_t KernelNamesSize;
    Adapter->call<UrApiKind::urProgramGetInfo>(
        UrProgram, UR_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr, &KernelNamesSize);

    // semi-colon delimited list of kernel names.
    std::string KernelNamesStr(KernelNamesSize, ' ');
    Adapter->call<UrApiKind::urProgramGetInfo>(
        UrProgram, UR_PROGRAM_INFO_KERNEL_NAMES, KernelNamesStr.size(),
        &KernelNamesStr[0], nullptr);
    std::vector<std::string> KernelNames =
        detail::split_string(KernelNamesStr, ';');

    // make the device image and the kernel_bundle_impl
    auto KernelIDs = std::make_shared<std::vector<kernel_id>>();
    auto DevImgImpl = std::make_shared<device_image_impl>(
        nullptr, MContext, MDevices, bundle_state::executable, KernelIDs,
        UrProgram);
    device_image_plain DevImg{DevImgImpl};

    // If caching enabled and kernel not fetched from cache, cache.
    if (PersistentDeviceCodeCache::isEnabled() && !FetchedFromCache &&
        SourceStrPtr) {
      PersistentDeviceCodeCache::putCompiledKernelToDisc(
          Devices, syclex::detail::userArgsAsString(BuildOptions),
          *SourceStrPtr, UrProgram);
    }

    return std::make_shared<kernel_bundle_impl>(MContext, MDevices, DevImg,
                                                KernelNames, MLanguage);
  }

  // Utility methods for kernel_compiler functionality
private:
  std::string adjust_kernel_name(const std::string &Name) {
    if (MLanguage == syclex::source_language::sycl) {
      auto It = MMangledKernelNames.find(Name);
      return It == MMangledKernelNames.end() ? Name : It->second;
    }

    return Name;
  }

  bool is_kernel_name(const std::string &Name) {
    return std::find(MKernelNames.begin(), MKernelNames.end(), Name) !=
           MKernelNames.end();
  }

  std::string mangle_device_global_name(const std::string &Name) {
    // TODO: Support device globals declared in namespaces.
    return "_Z" + std::to_string(Name.length()) + Name;
  }

  DeviceGlobalMapEntry *get_device_global_entry(const std::string &Name) {
    if (MKernelNames.empty() || MLanguage != syclex::source_language::sycl) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Querying device globals by name is only available "
                            "in kernel_bundles successfully built from "
                            "kernel_bundle<bundle_state>::ext_oneapi_source> "
                            "with 'sycl' source language.");
    }

    if (!ext_oneapi_has_device_global(Name)) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "device global '" + Name +
                                "' not found in kernel_bundle");
    }

    std::vector<DeviceGlobalMapEntry *> Entries =
        ProgramManager::getInstance().getDeviceGlobalEntries(
            {MPrefix + mangle_device_global_name(Name)});
    assert(Entries.size() == 1);
    return Entries.front();
  }

  void unregister_device_globals_from_context() {
    if (MDeviceGlobalNames.empty())
      return;

    // Manually trigger the release of resources for all device global map
    // entries associated with this runtime-compiled bundle. Normally, this
    // would happen in `~context_impl()`, however in the RTC setting, the
    // context outlives the DG map entries owned by the program manager.

    std::vector<std::string> DeviceGlobalIDs;
    std::transform(MDeviceGlobalNames.begin(), MDeviceGlobalNames.end(),
                   std::back_inserter(DeviceGlobalIDs),
                   [&](const std::string &DGName) { return MPrefix + DGName; });
    const auto &ContextImpl = getSyclObjImpl(MContext);
    for (DeviceGlobalMapEntry *Entry :
         ProgramManager::getInstance().getDeviceGlobalEntries(
             DeviceGlobalIDs)) {
      Entry->removeAssociatedResources(ContextImpl.get());
      ContextImpl->removeAssociatedDeviceGlobal(Entry->MDeviceGlobalPtr);
    }
  }

public:
  bool ext_oneapi_has_kernel(const std::string &Name) {
    return !MKernelNames.empty() && is_kernel_name(adjust_kernel_name(Name));
  }

  kernel
  ext_oneapi_get_kernel(const std::string &Name,
                        const std::shared_ptr<kernel_bundle_impl> &Self) {
    if (MKernelNames.empty())
      throw sycl::exception(make_error_code(errc::invalid),
                            "'ext_oneapi_get_kernel' is only available in "
                            "kernel_bundles successfully built from "
                            "kernel_bundle<bundle_state::ext_oneapi_source>.");

    std::string AdjustedName = adjust_kernel_name(Name);
    if (!is_kernel_name(AdjustedName))
      throw sycl::exception(make_error_code(errc::invalid),
                            "kernel '" + Name + "' not found in kernel_bundle");

    if (MLanguage == syclex::source_language::sycl) {
      auto &PM = ProgramManager::getInstance();
      auto KID = PM.getSYCLKernelID(MPrefix + AdjustedName);

      for (const auto &DevImgWithDeps : MDeviceImages) {
        const auto &DevImg = DevImgWithDeps.getMain();
        if (!DevImg.has_kernel(KID))
          continue;

        const auto &DevImgImpl = getSyclObjImpl(DevImg);
        auto UrProgram = DevImgImpl->get_ur_program_ref();
        auto [UrKernel, CacheMutex, ArgMask] =
            PM.getOrCreateKernel(MContext, AdjustedName,
                                 /*PropList=*/{}, UrProgram);
        auto KernelImpl = std::make_shared<kernel_impl>(
            UrKernel, getSyclObjImpl(MContext), DevImgImpl, Self, ArgMask,
            UrProgram, CacheMutex);
        return createSyclObjFromImpl<kernel>(std::move(KernelImpl));
      }

      assert(false && "Malformed RTC kernel bundle");
    }

    assert(MDeviceImages.size() > 0);
    const std::shared_ptr<detail::device_image_impl> &DeviceImageImpl =
        detail::getSyclObjImpl(MDeviceImages[0].getMain());
    ur_program_handle_t UrProgram = DeviceImageImpl->get_ur_program_ref();
    const ContextImplPtr &ContextImpl = getSyclObjImpl(MContext);
    const AdapterPtr &Adapter = ContextImpl->getAdapter();
    ur_kernel_handle_t UrKernel = nullptr;
    Adapter->call<UrApiKind::urKernelCreate>(UrProgram, AdjustedName.c_str(),
                                             &UrKernel);
    // Kernel created by urKernelCreate is implicitly retained.

    std::shared_ptr<kernel_impl> KernelImpl = std::make_shared<kernel_impl>(
        UrKernel, detail::getSyclObjImpl(MContext), Self);

    return detail::createSyclObjFromImpl<kernel>(std::move(KernelImpl));
  }

  std::string ext_oneapi_get_raw_kernel_name(const std::string &Name) {
    if (MKernelNames.empty())
      throw sycl::exception(
          make_error_code(errc::invalid),
          "'ext_oneapi_get_raw_kernel_name' is only available in "
          "kernel_bundles successfully built from "
          "kernel_bundle<bundle_state::ext_oneapi_source>.");

    std::string AdjustedName = adjust_kernel_name(Name);
    if (!is_kernel_name(AdjustedName))
      throw sycl::exception(make_error_code(errc::invalid),
                            "kernel '" + Name + "' not found in kernel_bundle");

    return AdjustedName;
  }

  bool ext_oneapi_has_device_global(const std::string &Name) {
    return !MDeviceGlobalNames.empty() &&
           std::find(MDeviceGlobalNames.begin(), MDeviceGlobalNames.end(),
                     mangle_device_global_name(Name)) !=
               MDeviceGlobalNames.end();
  }

  void *ext_oneapi_get_device_global_address(const std::string &Name,
                                             const device &Dev) {
    DeviceGlobalMapEntry *Entry = get_device_global_entry(Name);

    if (std::find(MDevices.begin(), MDevices.end(), Dev) == MDevices.end()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "kernel_bundle not built for device");
    }

    if (Entry->MIsDeviceImageScopeDecorated) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Cannot query USM pointer for device global with "
                            "'device_image_scope' property");
    }

    // TODO: Add context-only initialization via `urUSMContextMemcpyExp` instead
    // of using a throw-away queue.
    queue InitQueue{MContext, Dev};
    auto &USMMem =
        Entry->getOrAllocateDeviceGlobalUSM(getSyclObjImpl(InitQueue));
    InitQueue.wait_and_throw();
    return USMMem.getPtr();
  }

  size_t ext_oneapi_get_device_global_size(const std::string &Name) {
    return get_device_global_entry(Name)->MDeviceGlobalTSize;
  }

  bool empty() const noexcept { return MDeviceImages.empty(); }

  backend get_backend() const noexcept {
    return MContext.get_platform().get_backend();
  }

  context get_context() const noexcept { return MContext; }

  const std::vector<device> &get_devices() const noexcept { return MDevices; }

  std::vector<kernel_id> get_kernel_ids() const {
    // RTC kernel bundles shouldn't have user-facing kernel ids, return an
    // empty vector when the bundle contains RTC kernels.
    if (MLanguage == syclex::source_language::sycl) {
      return {};
    }
    // Collect kernel ids from all device images, then remove duplicates

    std::vector<kernel_id> Result;
    for (const device_image_plain &DeviceImage : MUniqueDeviceImages) {
      const std::vector<kernel_id> &KernelIDs =
          getSyclObjImpl(DeviceImage)->get_kernel_ids();

      Result.insert(Result.end(), KernelIDs.begin(), KernelIDs.end());
    }
    std::sort(Result.begin(), Result.end(), LessByNameComp{});

    auto NewIt = std::unique(Result.begin(), Result.end(), EqualByNameComp{});
    Result.erase(NewIt, Result.end());

    return Result;
  }

  kernel
  get_kernel(const kernel_id &KernelID,
             const std::shared_ptr<detail::kernel_bundle_impl> &Self) const {
    using ImageImpl = std::shared_ptr<detail::device_image_impl>;
    // Selected image.
    ImageImpl SelectedImage = nullptr;
    // Image where specialization constants are replaced with default values.
    ImageImpl ImageWithReplacedSpecConsts = nullptr;
    // Original image where specialization constants are not replaced with
    // default values.
    ImageImpl OriginalImage = nullptr;
    // Used to track if any of the candidate images has specialization values
    // set.
    bool SpecConstsSet = false;
    for (const DevImgPlainWithDeps &DeviceImageWithDeps : MDeviceImages) {
      const device_image_plain &DeviceImage = DeviceImageWithDeps.getMain();
      if (!DeviceImageWithDeps.getMain().has_kernel(KernelID))
        continue;

      const auto DeviceImageImpl = detail::getSyclObjImpl(DeviceImage);
      SpecConstsSet |= DeviceImageImpl->is_any_specialization_constant_set();

      // Remember current image in corresponding variable depending on whether
      // specialization constants are replaced with default value or not.
      (DeviceImageImpl->specialization_constants_replaced_with_default()
           ? ImageWithReplacedSpecConsts
           : OriginalImage) = DeviceImageImpl;

      if (SpecConstsSet) {
        // If specialization constant is set in any of the candidate images
        // then we can't use ReplacedImage, so we select NativeImage if any or
        // we select OriginalImage and keep iterating in case there is an image
        // with native support.
        SelectedImage = OriginalImage;
        if (SelectedImage &&
            SelectedImage->all_specialization_constant_native())
          break;
      } else {
        // For now select ReplacedImage but it may be reset if any of the
        // further device images has specialization constant value set. If after
        // all iterations specialization constant values are not set in any of
        // the candidate images then that will be the selected image.
        // Also we don't want to use ReplacedImage if device image has native
        // support.
        if (ImageWithReplacedSpecConsts &&
            !ImageWithReplacedSpecConsts->all_specialization_constant_native())
          SelectedImage = ImageWithReplacedSpecConsts;
        else
          // In case if we don't have or don't use ReplacedImage.
          SelectedImage = OriginalImage;
      }
    }

    if (!SelectedImage)
      throw sycl::exception(make_error_code(errc::invalid),
                            "The kernel bundle does not contain the kernel "
                            "identified by kernelId.");

    auto [Kernel, CacheMutex, ArgMask] =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            MContext, KernelID.get_name(), /*PropList=*/{},
            SelectedImage->get_ur_program_ref());

    std::shared_ptr<kernel_impl> KernelImpl = std::make_shared<kernel_impl>(
        Kernel, detail::getSyclObjImpl(MContext), SelectedImage, Self, ArgMask,
        SelectedImage->get_ur_program_ref(), CacheMutex);

    return detail::createSyclObjFromImpl<kernel>(std::move(KernelImpl));
  }

  bool has_kernel(const kernel_id &KernelID) const noexcept {
    return std::any_of(begin(), end(),
                       [&KernelID](const device_image_plain &DeviceImage) {
                         return DeviceImage.has_kernel(KernelID);
                       });
  }

  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept {
    return std::any_of(
        begin(), end(),
        [&KernelID, &Dev](const device_image_plain &DeviceImage) {
          return DeviceImage.has_kernel(KernelID, Dev);
        });
  }

  bool contains_specialization_constants() const noexcept {
    return std::any_of(
        begin(), end(), [](const device_image_plain &DeviceImage) {
          return getSyclObjImpl(DeviceImage)->has_specialization_constants();
        });
  }

  bool native_specialization_constant() const noexcept {
    return contains_specialization_constants() &&
           std::all_of(begin(), end(),
                       [](const device_image_plain &DeviceImage) {
                         return getSyclObjImpl(DeviceImage)
                             ->all_specialization_constant_native();
                       });
  }

  bool has_specialization_constant(const char *SpecName) const noexcept {
    return std::any_of(begin(), end(),
                       [SpecName](const device_image_plain &DeviceImage) {
                         return getSyclObjImpl(DeviceImage)
                             ->has_specialization_constant(SpecName);
                       });
  }

  void set_specialization_constant_raw_value(const char *SpecName,
                                             const void *Value,
                                             size_t Size) noexcept {
    if (has_specialization_constant(SpecName))
      for (const device_image_plain &DeviceImage : MUniqueDeviceImages)
        getSyclObjImpl(DeviceImage)
            ->set_specialization_constant_raw_value(SpecName, Value);
    else {
      std::vector<unsigned char> &Val = MSpecConstValues[std::string{SpecName}];
      Val.resize(Size);
      std::memcpy(Val.data(), Value, Size);
    }
  }

  void get_specialization_constant_raw_value(const char *SpecName,
                                             void *ValueRet) const noexcept {
    for (const device_image_plain &DeviceImage : MUniqueDeviceImages)
      if (getSyclObjImpl(DeviceImage)->has_specialization_constant(SpecName)) {
        getSyclObjImpl(DeviceImage)
            ->get_specialization_constant_raw_value(SpecName, ValueRet);
        return;
      }

    // Specialization constant wasn't found in any of the device images,
    // try to fetch value from kernel_bundle.
    if (MSpecConstValues.count(std::string{SpecName}) != 0) {
      const std::vector<unsigned char> &Val =
          MSpecConstValues.at(std::string{SpecName});
      auto *Dest = static_cast<unsigned char *>(ValueRet);
      std::uninitialized_copy(Val.begin(), Val.end(), Dest);
      return;
    }

    assert(false &&
           "get_specialization_constant_raw_value called for missing constant");
  }

  bool is_specialization_constant_set(const char *SpecName) const noexcept {
    bool SetInDevImg = std::any_of(
        begin(), end(), [SpecName](const device_image_plain &DeviceImage) {
          return getSyclObjImpl(DeviceImage)
              ->is_specialization_constant_set(SpecName);
        });
    return SetInDevImg || MSpecConstValues.count(std::string{SpecName}) != 0;
  }

  const device_image_plain *begin() const { return MUniqueDeviceImages.data(); }

  const device_image_plain *end() const {
    return MUniqueDeviceImages.data() + MUniqueDeviceImages.size();
  }

  size_t size() const noexcept { return MUniqueDeviceImages.size(); }

  bundle_state get_bundle_state() const { return MState; }

  const SpecConstMapT &get_spec_const_map_ref() const noexcept {
    return MSpecConstValues;
  }

  bool isInterop() const { return MIsInterop; }

  bool add_kernel(const kernel_id &KernelID, const device &Dev) {
    // Skip if kernel is already there
    if (has_kernel(KernelID, Dev))
      return true;

    // First try and get images in current bundle state
    const bundle_state BundleState = get_bundle_state();
    std::vector<DevImgPlainWithDeps> NewDevImgs =
        detail::ProgramManager::getInstance().getSYCLDeviceImages(
            MContext, {Dev}, {KernelID}, BundleState);

    // No images found so we report as not inserted
    if (NewDevImgs.empty())
      return false;

    // Propagate already set specialization constants to the new images
    for (DevImgPlainWithDeps &DevImgWithDeps : NewDevImgs)
      for (device_image_plain &DevImg : DevImgWithDeps)
        for (auto SpecConst : MSpecConstValues)
          getSyclObjImpl(DevImg)->set_specialization_constant_raw_value(
              SpecConst.first.c_str(), SpecConst.second.data());

    // Add the images to the collection
    MDeviceImages.insert(MDeviceImages.end(), NewDevImgs.begin(),
                         NewDevImgs.end());
    removeDuplicateImages();
    return true;
  }

  ~kernel_bundle_impl() {
    try {
      if (MDeviceBinaries) {
        unregister_device_globals_from_context();
        ProgramManager::getInstance().removeImages(MDeviceBinaries);
        syclex::detail::SYCL_JIT_Destroy(MDeviceBinaries);
      }
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~kernel_bundle_impl", e);
    }
  }

private:
  void fillUniqueDeviceImages() {
    assert(MUniqueDeviceImages.empty());
    for (const DevImgPlainWithDeps &Imgs : MDeviceImages)
      MUniqueDeviceImages.insert(MUniqueDeviceImages.end(), Imgs.begin(),
                                 Imgs.end());
    removeDuplicateImages();
  }
  void removeDuplicateImages() {
    std::sort(MUniqueDeviceImages.begin(), MUniqueDeviceImages.end(),
              LessByHash<device_image_plain>{});
    const auto It =
        std::unique(MUniqueDeviceImages.begin(), MUniqueDeviceImages.end());
    MUniqueDeviceImages.erase(It, MUniqueDeviceImages.end());
  }
  context MContext;
  std::vector<device> MDevices;
  std::vector<DevImgPlainWithDeps> MDeviceImages;
  std::vector<device_image_plain> MUniqueDeviceImages;
  // This map stores values for specialization constants, that are missing
  // from any device image.
  SpecConstMapT MSpecConstValues;
  bool MIsInterop = false;
  bundle_state MState;

  // ext_oneapi_kernel_compiler : Source, Languauge, KernelNames, IncludePairs
  // Language is for both state::source and state::executable.
  syclex::source_language MLanguage = syclex::source_language::opencl;
  const std::variant<std::string, std::vector<std::byte>> MSource;
  // only kernel_bundles created from source have KernelNames member.
  std::vector<std::string> MKernelNames;
  std::unordered_map<std::string, std::string> MMangledKernelNames;
  std::vector<std::string> MDeviceGlobalNames;
  std::vector<std::unique_ptr<std::byte[]>> MDeviceGlobalAllocations;
  sycl_device_binaries MDeviceBinaries = nullptr;
  std::string MPrefix;
  include_pairs_t MIncludePairs;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
