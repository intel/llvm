//==------- kernel_bundle_impl.hpp - SYCL kernel_bundle_impl ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/device_global_map.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/link_graph.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/syclbin.hpp>
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
#include <string_view>
#include <unordered_set>
#include <vector>

#include "split_string.hpp"

namespace sycl {
inline namespace _V1 {

namespace ext::oneapi::experimental::detail {
using namespace sycl::detail;
bool is_source_kernel_bundle_supported(
    sycl::ext::oneapi::experimental::source_language Language,
    const context &Ctx);

bool is_source_kernel_bundle_supported(
    sycl::ext::oneapi::experimental::source_language Language,
    const std::vector<device_impl *> &Devices);
} // namespace ext::oneapi::experimental::detail

namespace detail {

inline bool checkAllDevicesAreInContext(devices_range Devices,
                                        const context &Context) {
  return std::all_of(Devices.begin(), Devices.end(),
                     [&Context](device_impl &Dev) {
                       return getSyclObjImpl(Context)->isDeviceValid(Dev);
                     });
}

inline bool checkAllDevicesHaveAspect(devices_range Devices, aspect Aspect) {
  return std::all_of(Devices.begin(), Devices.end(),
                     [&Aspect](device_impl &Dev) { return Dev.has(Aspect); });
}

// Creates a link graph where the edges represent the relationship between
// imported and exported symbols in the provided images.
// The link graph takes a vector of images and a vector of vectors of integral
// values. The latter is the dependencies of the images in the former argument.
// Each vector of dependencies correspond 1:1 with the images in the device
// images, and the values in each of these vectors correspond to the index of
// each of the images it depends on.
inline LinkGraph<device_image_plain>
CreateLinkGraph(const std::vector<device_image_plain> &DevImages) {
  // Create a map between exported symbols and their indices in the device
  // images collection.
  std::map<std::string_view, size_t> ExportMap;
  for (size_t I = 0; I < DevImages.size(); ++I) {
    device_image_impl &DevImageImpl = *getSyclObjImpl(DevImages[I]);
    if (DevImageImpl.get_bin_image_ref() == nullptr)
      continue;
    for (const sycl_device_binary_property &ESProp :
         DevImageImpl.get_bin_image_ref()->getExportedSymbols()) {
      if (!ExportMap.insert({ESProp->Name, I}).second)
        throw sycl::exception(make_error_code(errc::invalid),
                              "Duplicate exported symbol \"" +
                                  std::string{ESProp->Name} +
                                  "\" found in binaries.");
    }
  }

  // Create dependency mappings.
  std::vector<std::vector<size_t>> Dependencies;
  Dependencies.resize(DevImages.size());
  for (size_t I = 0; I < DevImages.size(); ++I) {
    device_image_impl &DevImageImpl = *getSyclObjImpl(DevImages[I]);
    if (DevImageImpl.get_bin_image_ref() == nullptr)
      continue;
    std::set<size_t> DeviceImageDepsSet;
    for (const sycl_device_binary_property &ISProp :
         DevImageImpl.get_bin_image_ref()->getImportedSymbols()) {
      auto ExportSymbolIt = ExportMap.find(ISProp->Name);
      if (ExportSymbolIt == ExportMap.end())
        throw sycl::exception(make_error_code(errc::invalid),
                              "No exported symbol \"" +
                                  std::string{ISProp->Name} +
                                  "\" found in linked images.");
      DeviceImageDepsSet.emplace(ExportSymbolIt->second);
    }
    Dependencies[I].insert(Dependencies[I].end(), DeviceImageDepsSet.begin(),
                           DeviceImageDepsSet.end());
  }
  return LinkGraph<device_image_plain>{DevImages, Dependencies};
}

inline void ThrowIfConflictingKernels(device_images_range DevImages) {
  std::set<std::string_view> SeenKernelNames;
  std::set<std::string_view> Conflicts;
  for (const device_image_impl &DevImage : DevImages) {
    const KernelNameSetT &KernelNames = DevImage.getKernelNames();
    std::set_intersection(SeenKernelNames.begin(), SeenKernelNames.end(),
                          KernelNames.begin(), KernelNames.end(),
                          std::inserter(Conflicts, Conflicts.begin()));
    SeenKernelNames.insert(KernelNames.begin(), KernelNames.end());
  }
  if (Conflicts.empty())
    return;
  std::stringstream MsgS;
  MsgS << "Conflicting kernel definitions: ";
  for (const std::string_view &Conflict : Conflicts)
    MsgS << " " << Conflict;
  throw sycl::exception(make_error_code(errc::invalid), MsgS.str());
}

namespace syclex = sycl::ext::oneapi::experimental;

class kernel_impl;

/// The class is an impl counterpart of the sycl::kernel_bundle.
// It provides an access and utilities to manage set of sycl::device_images
// objects.
class kernel_bundle_impl
    : public std::enable_shared_from_this<kernel_bundle_impl> {

  using SpecConstMapT = std::map<std::string, std::vector<unsigned char>>;
  using Base = std::enable_shared_from_this<kernel_bundle_impl>;

  struct private_tag {
    explicit private_tag() = default;
  };

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
  kernel_bundle_impl(context Ctx, devices_range Devs, bundle_state State,
                     private_tag)
      : MContext(std::move(Ctx)),
        MDevices(Devs.to<std::vector<device_impl *>>()), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, State);
    fillUniqueDeviceImages();
  }

  // Interop constructor used by make_kernel
  kernel_bundle_impl(context Ctx, devices_range Devs, private_tag)
      : MContext(Ctx), MDevices(Devs.to<std::vector<device_impl *>>()),
        MState(bundle_state::executable) {
    if (!checkAllDevicesAreInContext(Devs, Ctx))
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are associated with the context or "
          "vector of devices is empty");
  }

  // Interop constructor
  kernel_bundle_impl(context Ctx, devices_range Devs,
                     device_image_plain &&DevImage, private_tag Tag)
      : kernel_bundle_impl(std::move(Ctx), Devs, Tag) {
    MDeviceImages.emplace_back(DevImage);
    MUniqueDeviceImages.emplace_back(std::move(DevImage));
  }

  // Matches sycl::build and sycl::compile
  // Have one constructor because sycl::build and sycl::compile have the same
  // signature
  kernel_bundle_impl(const kernel_bundle<bundle_state::input> &InputBundle,
                     devices_range Devs, const property_list &PropList,
                     bundle_state TargetState, private_tag)
      : MContext(InputBundle.get_context()),
        MDevices(Devs.to<std::vector<device_impl *>>()), MState(TargetState) {

    kernel_bundle_impl &InputBundleImpl = *getSyclObjImpl(InputBundle);
    MSpecConstValues = InputBundleImpl.get_spec_const_map_ref();

    devices_range InputBundleDevices = InputBundleImpl.get_devices();
    const bool AllDevsAssociatedWithInputBundle =
        std::all_of(get_devices().begin(), get_devices().end(),
                    [&InputBundleDevices](device_impl &Dev) {
                      return InputBundleDevices.contains(Dev);
                    });
    if (MDevices.empty() || !AllDevsAssociatedWithInputBundle)
      throw sycl::exception(
          make_error_code(errc::invalid),
          "Not all devices are in the set of associated "
          "devices for input bundle or vector of devices is empty");

    // Copy SYCLBINs to ensure lifetime is preserved by the executable bundle.
    MSYCLBINs.insert(MSYCLBINs.end(), InputBundleImpl.MSYCLBINs.begin(),
                     InputBundleImpl.MSYCLBINs.end());

    for (const DevImgPlainWithDeps &DevImgWithDeps :
         InputBundleImpl.MDeviceImages) {
      // Skip images which are not compatible with devices provided
      if (none_of(get_devices(),
                  [&MainImg = *getSyclObjImpl(DevImgWithDeps.getMain())](
                      device_impl &Dev) {
                    return MainImg.compatible_with_device(Dev);
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
        populateDeviceGlobalsForSYCLBIN();
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
      sycl::span<const kernel_bundle<bundle_state::object>> &ObjectBundles,
      devices_range Devs, const property_list &PropList, bool FastLink,
      private_tag)
      : MDevices(Devs.to<std::vector<device_impl *>>()),
        MState(bundle_state::executable) {
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
        get_devices().begin(), get_devices().end(),
        [&ObjectBundles](device_impl &Dev) {
          // Number of devices is expected to be small
          return std::all_of(
              ObjectBundles.begin(), ObjectBundles.end(),
              [&Dev](const kernel_bundle<bundle_state::object> &KernelBundle) {
                devices_range BundleDevices =
                    getSyclObjImpl(KernelBundle)->get_devices();
                return BundleDevices.contains(Dev);
              });
        });
    if (!AllDevsAssociatedWithInputBundles)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Not all devices are in the set of associated "
                            "devices for input bundles");

    // TODO: Unify with c'tor for sycl::compile and sycl::build by calling
    // sycl::join on vector of kernel_bundles

    // Due to a bug in L0, specializations with conflicting IDs will overwrite
    // each other when linked together, so to avoid this issue we link
    // regular offline-compiled SYCL device images in separation.
    // TODO: Remove when spec const overwriting issue has been fixed in L0.
    std::vector<const DevImgPlainWithDeps *> OfflineDeviceImages;
    std::unordered_set<device_image_impl *> OfflineDeviceImageSet;
    for (const kernel_bundle<bundle_state::object> &ObjectBundle :
         ObjectBundles) {
      for (const DevImgPlainWithDeps &DeviceImageWithDeps :
           getSyclObjImpl(ObjectBundle)->MDeviceImages) {
        if (getSyclObjImpl(DeviceImageWithDeps.getMain())->getOriginMask() &
            ImageOriginSYCLOffline) {
          OfflineDeviceImages.push_back(&DeviceImageWithDeps);
          for (const device_image_plain &DevImg : DeviceImageWithDeps)
            OfflineDeviceImageSet.insert(&*getSyclObjImpl(DevImg));
        }
      }
    }

    std::map<device_impl *, LinkGraph<device_image_plain>> DevImageLinkGraphs;

    // The linking logic differs depending on whether fast-linking is enabled:
    // When doing fast-linking, we insert the suitable AOT binaries from the
    // object bundles. This needs to be done per-device, as AOT binaries may
    // not be compatible across different architectures. As such, the link-graph
    // needs to be constructed for each device.
    // When doing regular linking, the JIT images in the graph will be the same
    // for all devices, so the link-graph can be created once and then resolved
    // for each device in separation, with a unification at the end to bundle
    // together the linking for devices with the same resulting sets of linked
    // images.
    if (FastLink) {
      for (device_impl &Dev : get_devices()) {
        std::vector<device_image_plain> DevImages;
        std::set<device_image_impl *> DevImagesSet;
        for (const kernel_bundle<bundle_state::object> &ObjectBundle :
             ObjectBundles) {
          detail::kernel_bundle_impl &ObjectBundleImpl =
              *getSyclObjImpl(ObjectBundle);

          // Firstly find all suitable AOT binaries, if the object bundle was
          // made from SYCLBIN.
          std::vector<const RTDeviceBinaryImage *> AOTBinaries =
              ObjectBundleImpl.GetSYCLBINAOTBinaries(Dev);

          // The AOT binaries need to be brought into executable state. They
          // are considered unique, so they are placed directly into the unique
          // images list.
          DevImages.reserve(AOTBinaries.size());
          for (const detail::RTDeviceBinaryImage *Image : AOTBinaries) {
            device_image_plain &AOTDevImg =
                DevImages.emplace_back(device_image_impl::create(
                    Image, MContext, devices_range{Dev},
                    bundle_state::executable,
                    /*KernelIDs=*/nullptr, Managed<ur_program_handle_t>{},
                    ImageOriginSYCLBIN));
            DevImgPlainWithDeps AOTDevImgWithDeps{AOTDevImg};
            ProgramManager::getInstance().bringSYCLDeviceImageToState(
                AOTDevImgWithDeps, bundle_state::executable);
          }

          // Record all the AOT exported symbols and kernels.
          std::unordered_set<std::string_view> AOTExportedSymbols;
          std::unordered_set<std::string_view> AOTKernelNames;
          for (const RTDeviceBinaryImage *AOTBin : AOTBinaries) {
            for (const sycl_device_binary_property &ESProp :
                 AOTBin->getExportedSymbols())
              AOTExportedSymbols.insert(ESProp->Name);
            for (const sycl_device_binary_property &KNProp :
                 AOTBin->getKernelNames())
              AOTKernelNames.insert(KNProp->Name);
          }

          for (device_image_impl &DevImg : ObjectBundleImpl.device_images()) {
            // If the image is the same as one of the offline images, we can
            // skip it.
            if (OfflineDeviceImageSet.find(&DevImg) !=
                OfflineDeviceImageSet.end())
              continue;

            // If any of the exported symbols overlap with an AOT binary, skip
            // this image as fast-linking prioritizes AOT binaries.
            // This can happen if the same source files have been compiled to
            // both a usable AOT and JIT binary.
            for (const sycl_device_binary_property &ESProp :
                 DevImg.get_bin_image_ref()->getExportedSymbols())
              if (AOTExportedSymbols.find(ESProp->Name) !=
                  AOTExportedSymbols.end())
                continue;

            // Same as above, but regarding kernels instead of exported symbols.
            for (const sycl_device_binary_property &KNProp :
                 DevImg.get_bin_image_ref()->getKernelNames())
              if (AOTKernelNames.find(KNProp->Name) != AOTKernelNames.end())
                continue;

            DevImagesSet.insert(&DevImg);
          }
        }

        // Move the images from the set to images collection.
        // Note: Move iterators do not allow moving from sets, so instead we
        //       need to call extract.
        //       https://godbolt.org/z/h3YGoPYbr illustrates this limitation.
        DevImages.reserve(DevImages.size() + DevImagesSet.size());
        for (auto It = DevImagesSet.begin(); It != DevImagesSet.end();)
          DevImages.push_back(createSyclObjFromImpl<device_image_plain>(
              *DevImagesSet.extract(It++).value()));

        // Check for conflicting kernels in RTC kernel bundles.
        ThrowIfConflictingKernels(DevImages);

        // Create and insert the corresponding link graph.
        DevImageLinkGraphs.emplace(&Dev, CreateLinkGraph(DevImages));
      }
    } else {
      // Collect all unique images.
      std::vector<device_image_plain> DevImages;
      {
        std::set<device_image_impl *> DevImagesSet;
        for (const kernel_bundle<bundle_state::object> &ObjectBundle :
             ObjectBundles)
          for (device_image_impl &DevImg :
               getSyclObjImpl(ObjectBundle)->device_images())
            if (OfflineDeviceImageSet.find(&DevImg) ==
                OfflineDeviceImageSet.end())
              DevImagesSet.insert(&DevImg);
        DevImages.reserve(DevImagesSet.size());
        for (auto It = DevImagesSet.begin(); It != DevImagesSet.end();)
          DevImages.push_back(createSyclObjFromImpl<device_image_plain>(
              *DevImagesSet.extract(It++).value()));
      }

      // Check for conflicting kernels in RTC kernel bundles.
      ThrowIfConflictingKernels(DevImages);

      // Create a link graph and clone it for each device.
      device_impl &FirstDevice = get_devices().front();
      const auto &FirstGraph =
          DevImageLinkGraphs.emplace(&FirstDevice, CreateLinkGraph(DevImages))
              .first->second;
      for (device_impl &Dev : get_devices())
        DevImageLinkGraphs.emplace(&Dev, FirstGraph.Clone());
    }

    // Poison the images based on whether the corresponding device supports it.
    for (auto &GraphIt : DevImageLinkGraphs) {
      device_impl &Dev = *GraphIt.first;
      GraphIt.second.Poison([&Dev](const device_image_plain &DevImg) {
        return !getSyclObjImpl(DevImg)->compatible_with_device(Dev);
      });
    }

    // Unify graphs after poisoning.
    std::map<std::vector<device_impl *>, LinkGraph<device_image_plain>>
        UnifiedGraphs = UnifyGraphs(DevImageLinkGraphs);

    // Link based on the resulting graphs.
    for (auto &GraphIt : UnifiedGraphs) {
      const std::vector<device_impl *> &GraphDevs = GraphIt.first;
      std::vector<device_image_plain> GraphImgs =
          GraphIt.second.GetNodeValues();

      auto [JITImgs, AOTImgs] =
          [&]() -> std::pair<device_images_range, device_images_range> {
        // If we are not fast-linking, all images must be JIT.
        if (!FastLink)
          return {GraphImgs, {}};

        std::sort(
            GraphImgs.begin(), GraphImgs.end(),
            [](const device_image_plain &LHS, const device_image_plain &RHS) {
              // Sort by state: That leaves objects (JIT) at the beginning and
              // executables (AOT) at the end.
              return getSyclObjImpl(LHS)->get_state() <
                     getSyclObjImpl(RHS)->get_state();
            });
        auto AOTImgsBegin =
            std::find_if(GraphImgs.begin(), GraphImgs.end(),
                         [](const device_image_plain &Img) {
                           return getSyclObjImpl(Img)->get_state() ==
                                  bundle_state::executable;
                         });
        size_t NumJITImgs = std::distance(GraphImgs.begin(), AOTImgsBegin);
        return {{GraphImgs.begin(), GraphImgs.begin() + NumJITImgs},
                {GraphImgs.begin() + NumJITImgs, GraphImgs.end()}};
      }();

      // If there AOT binaries, the link should allow unresolved symbols.
      std::vector<device_image_plain> LinkedResults =
          detail::ProgramManager::getInstance().link(
              JITImgs, GraphDevs, PropList,
              /*AllowUnresolvedSymbols=*/!AOTImgs.empty());

      if (!AOTImgs.empty()) {
        // In dynamic linking, AOT binaries count as results as well.
        LinkedResults.reserve(LinkedResults.size() + AOTImgs.size());
        for (device_image_impl &AOTImg : AOTImgs)
          LinkedResults.push_back(
              createSyclObjFromImpl<device_image_plain>(AOTImg));
        detail::ProgramManager::getInstance().dynamicLink(LinkedResults);
      }

      MDeviceImages.insert(MDeviceImages.end(), LinkedResults.begin(),
                           LinkedResults.end());
      MUniqueDeviceImages.insert(MUniqueDeviceImages.end(),
                                 LinkedResults.begin(), LinkedResults.end());
      // TODO: Kernels may be in multiple device images, so mapping should be
      //       added.
    }

    // ... And link the offline images in separation. (Workaround.)
    for (const DevImgPlainWithDeps *DeviceImageWithDeps : OfflineDeviceImages) {
      // Skip images which are not compatible with devices provided
      if (none_of(get_devices(),
                  [&MainImg = *getSyclObjImpl(DeviceImageWithDeps->getMain())](
                      device_impl &Dev) {
                    return MainImg.compatible_with_device(Dev);
                  }))
        continue;

      std::vector<device_image_plain> LinkedResults =
          detail::ProgramManager::getInstance().link(
              DeviceImageWithDeps->getAll(), MDevices, PropList);
      MDeviceImages.insert(MDeviceImages.end(), LinkedResults.begin(),
                           LinkedResults.end());
      MUniqueDeviceImages.insert(MUniqueDeviceImages.end(),
                                 LinkedResults.begin(), LinkedResults.end());
    }

    removeDuplicateImages();

    populateDeviceGlobalsForSYCLBIN();

    for (const kernel_bundle<bundle_state::object> &Bundle : ObjectBundles) {
      kernel_bundle_impl &BundleImpl = *getSyclObjImpl(Bundle);
      for (const auto &[Name, Values] : BundleImpl.MSpecConstValues) {
        MSpecConstValues[Name] = Values;
      }
    }
  }

  kernel_bundle_impl(context Ctx, devices_range Devs,
                     const std::vector<kernel_id> &KernelIDs,
                     bundle_state State, private_tag)
      : MContext(std::move(Ctx)),
        MDevices(Devs.to<std::vector<device_impl *>>()), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, KernelIDs, State);
    fillUniqueDeviceImages();
  }

  kernel_bundle_impl(context Ctx, devices_range Devs,
                     const DevImgSelectorImpl &Selector, bundle_state State,
                     private_tag)
      : MContext(std::move(Ctx)),
        MDevices(Devs.to<std::vector<device_impl *>>()), MState(State) {

    common_ctor_checks();

    MDeviceImages = detail::ProgramManager::getInstance().getSYCLDeviceImages(
        MContext, MDevices, Selector, State);
    fillUniqueDeviceImages();
  }

  // C'tor matches sycl::join API
  kernel_bundle_impl(const std::vector<detail::KernelBundleImplPtr> &Bundles,
                     bundle_state State, private_tag)
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

    // Pre-count and reserve space in vectors.
    {
      size_t NumDevImgs = 0, NumSharedDevBins = 0, NumSYCLBINs = 0;
      for (const detail::KernelBundleImplPtr &Bundle : Bundles) {
        NumDevImgs += Bundle->MDeviceImages.size();
        NumSharedDevBins += Bundle->MSharedDeviceBinaries.size();
        NumSYCLBINs += Bundle->MSYCLBINs.size();
      }
      MDeviceImages.reserve(NumDevImgs);
      MSharedDeviceBinaries.reserve(NumSharedDevBins);
      MSYCLBINs.reserve(NumSYCLBINs);
    }

    for (const detail::KernelBundleImplPtr &Bundle : Bundles) {
      MDeviceImages.insert(MDeviceImages.end(), Bundle->MDeviceImages.begin(),
                           Bundle->MDeviceImages.end());
      MSharedDeviceBinaries.insert(MSharedDeviceBinaries.end(),
                                   Bundle->MSharedDeviceBinaries.begin(),
                                   Bundle->MSharedDeviceBinaries.end());
      MSYCLBINs.insert(MSYCLBINs.end(), Bundle->MSYCLBINs.begin(),
                       Bundle->MSYCLBINs.end());
    }

    fillUniqueDeviceImages();

    if (get_bundle_state() == bundle_state::executable)
      populateDeviceGlobalsForSYCLBIN();

    if (get_bundle_state() == bundle_state::input) {
      // Copy spec constants values from the device images.
      for (detail::device_image_impl &ImgImpl : device_images()) {
        const std::map<std::string,
                       std::vector<device_image_impl::SpecConstDescT>>
            &SpecConsts = ImgImpl.get_spec_const_data_ref();
        const std::vector<unsigned char> &Blob =
            ImgImpl.get_spec_const_blob_ref();
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
      }
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

  // oneapi_ext_kernel_compiler
  // construct from source string
  kernel_bundle_impl(const context &Context, syclex::source_language Lang,
                     const std::string &Src, include_pairs_t IncludePairsVec,
                     private_tag)
      : MContext(Context), MDevices(getSyclObjImpl(Context)
                                        ->getDevices()
                                        .to<std::vector<device_impl *>>()),
        MDeviceImages{device_image_plain{device_image_impl::create(
            Src, MContext, MDevices, Lang, std::move(IncludePairsVec))}},
        MUniqueDeviceImages{MDeviceImages[0].getMain()},
        MState(bundle_state::ext_oneapi_source) {
    common_ctor_checks();
  }

  // oneapi_ext_kernel_compiler
  // construct from source bytes
  kernel_bundle_impl(const context &Context, syclex::source_language Lang,
                     const std::vector<std::byte> &Bytes, private_tag)
      : MContext(Context), MDevices(getSyclObjImpl(Context)
                                        ->getDevices()
                                        .to<std::vector<device_impl *>>()),
        MDeviceImages{device_image_plain{
            device_image_impl::create(Bytes, MContext, MDevices, Lang)}},
        MUniqueDeviceImages{MDeviceImages[0].getMain()},
        MState(bundle_state::ext_oneapi_source) {
    common_ctor_checks();
  }

  // oneapi_ext_kernel_compiler
  // construct from built source files
  kernel_bundle_impl(
      const context &Context, devices_range Devs,
      std::vector<device_image_plain> &&DevImgs,
      std::vector<std::shared_ptr<ManagedDeviceBinaries>> &&DevBinaries,
      bundle_state State, private_tag)
      : MContext(Context), MDevices(Devs.to<std::vector<device_impl *>>()),
        MSharedDeviceBinaries(std::move(DevBinaries)),
        MUniqueDeviceImages(std::move(DevImgs)), MState(State) {
    common_ctor_checks();

    removeDuplicateImages();
    MDeviceImages.reserve(MUniqueDeviceImages.size());
    for (const device_image_plain &DevImg : MUniqueDeviceImages)
      MDeviceImages.emplace_back(DevImg);
  }

  // SYCLBIN constructor
  kernel_bundle_impl(const context &Context, devices_range Devs,
                     const sycl::span<char> Bytes, bundle_state State,
                     private_tag)
      : MContext(Context), MDevices(Devs.to<std::vector<device_impl *>>()),
        MState(State) {
    common_ctor_checks();

    auto &SYCLBIN = MSYCLBINs.emplace_back(
        std::make_shared<SYCLBINBinaries>(Bytes.data(), Bytes.size()));

    if (SYCLBIN->getState() != static_cast<uint8_t>(State))
      throw sycl::exception(
          make_error_code(errc::invalid),
          "kernel_bundle state does not match the state of the SYCLBIN file.");

    std::vector<const detail::RTDeviceBinaryImage *> BestImages =
        SYCLBIN->getBestCompatibleImages(Devs, State);
    MDeviceImages.reserve(BestImages.size());
    for (const detail::RTDeviceBinaryImage *Image : BestImages)
      MDeviceImages.emplace_back(device_image_impl::create(
          Image, Context, Devs, ProgramManager::getBinImageState(Image),
          /*KernelIDs=*/nullptr, Managed<ur_program_handle_t>{},
          ImageOriginSYCLBIN));
    ProgramManager::getInstance().bringSYCLDeviceImagesToState(MDeviceImages,
                                                               State);
    fillUniqueDeviceImages();
    if (State == bundle_state::executable)
      populateDeviceGlobalsForSYCLBIN();
  }

  template <typename... Ts>
  static std::shared_ptr<kernel_bundle_impl> create(Ts &&...args) {
    return std::make_shared<kernel_bundle_impl>(std::forward<Ts>(args)...,
                                                private_tag{});
  }

  std::shared_ptr<kernel_bundle_impl> build_from_source(
      devices_range Devices,
      const std::vector<sycl::detail::string_view> &BuildOptions,
      std::string *LogPtr,
      const std::vector<sycl::detail::string_view> &RegisteredKernelNames) {
    assert(MState == bundle_state::ext_oneapi_source &&
           "bundle_state::ext_oneapi_source required");
    assert(allSourceBasedImages() && "All images must be source-based.");

    std::vector<device_image_plain> NewDevImgs;
    std::vector<std::shared_ptr<ManagedDeviceBinaries>> NewBinReso;
    for (device_image_impl &DevImg : device_images()) {
      std::vector<std::shared_ptr<device_image_impl>> NewDevImgImpls =
          DevImg.buildFromSource(Devices, BuildOptions, LogPtr,
                                 RegisteredKernelNames, NewBinReso);
      NewDevImgs.reserve(NewDevImgs.size() + NewDevImgImpls.size());
      for (std::shared_ptr<device_image_impl> &DevImgImpl : NewDevImgImpls)
        NewDevImgs.emplace_back(std::move(DevImgImpl));
    }
    return create(MContext, Devices, std::move(NewDevImgs),
                  std::move(NewBinReso), bundle_state::executable);
  }

  std::shared_ptr<kernel_bundle_impl> compile_from_source(
      devices_range Devices,
      const std::vector<sycl::detail::string_view> &CompileOptions,
      std::string *LogPtr,
      const std::vector<sycl::detail::string_view> &RegisteredKernelNames) {
    assert(MState == bundle_state::ext_oneapi_source &&
           "bundle_state::ext_oneapi_source required");
    assert(allSourceBasedImages() && "All images must be source-based.");

    std::vector<device_image_plain> NewDevImgs;
    std::vector<std::shared_ptr<ManagedDeviceBinaries>> NewBinReso;
    for (device_image_impl &DevImg : device_images()) {
      std::vector<std::shared_ptr<device_image_impl>> NewDevImgImpls =
          DevImg.compileFromSource(Devices, CompileOptions, LogPtr,
                                   RegisteredKernelNames, NewBinReso);
      NewDevImgs.reserve(NewDevImgs.size() + NewDevImgImpls.size());
      for (std::shared_ptr<device_image_impl> &DevImgImpl : NewDevImgImpls)
        NewDevImgs.emplace_back(std::move(DevImgImpl));
    }
    return create(MContext, Devices, std::move(NewDevImgs),
                  std::move(NewBinReso), bundle_state::object);
  }

public:
  bool ext_oneapi_has_kernel(const std::string &Name) const {
    return any_of(device_images(), [&Name](device_image_impl &DevImg) {
      return DevImg.hasKernelName(Name);
    });
  }

  kernel ext_oneapi_get_kernel(const std::string &Name) const {
    if (!hasSourceBasedImages() && !hasSYCLBINImages())
      throw sycl::exception(make_error_code(errc::invalid),
                            "'ext_oneapi_get_kernel' is only available in "
                            "kernel_bundles created from SYCLBIN files and "
                            "kernel_bundles successfully built from "
                            "kernel_bundle<bundle_state::ext_oneapi_source>.");

    // TODO: When linking is properly implemented for kernel compiler binaries,
    //       there can be scenarios where multiple binaries have the same
    //       kernels. In this case, all these bundles should be found and the
    //       resulting kernel object should be able to map devices to their
    //       respective backend kernel objects.
    for (device_image_impl &DevImg : device_images()) {
      if (std::shared_ptr<kernel_impl> PotentialKernelImpl =
              DevImg.tryGetExtensionKernel(Name, MContext, *this))
        return detail::createSyclObjFromImpl<kernel>(
            std::move(PotentialKernelImpl));
    }
    throw sycl::exception(make_error_code(errc::invalid),
                          "kernel '" + Name + "' not found in kernel_bundle");
  }

  std::string ext_oneapi_get_raw_kernel_name(const std::string &Name) {
    if (!hasSourceBasedImages() && !hasSYCLBINImages())
      throw sycl::exception(make_error_code(errc::invalid),
                            "'ext_oneapi_get_raw_kernel_name' is only "
                            "available in kernel_bundles created from SYCLBIN "
                            "files and kernel_bundles successfully built from "
                            "kernel_bundle<bundle_state::ext_oneapi_source>.");

    auto It = std::find_if(device_images().begin(), device_images().end(),
                           [&Name](device_image_impl &DevImg) {
                             return DevImg.hasKernelName(Name);
                           });
    if (It == device_images().end())
      throw sycl::exception(make_error_code(errc::invalid),
                            "kernel '" + Name + "' not found in kernel_bundle");

    return It->getAdjustedKernelNameStr(Name);
  }

  bool ext_oneapi_has_device_global(const std::string &Name) const {
    std::string MangledName = mangleDeviceGlobalName(Name);
    return (MDeviceGlobals.size() &&
            MDeviceGlobals.tryGetEntryLockless(MangledName)) ||
           std::any_of(device_images().begin(), device_images().end(),
                       [&MangledName](device_image_impl &DeviceImage) {
                         return DeviceImage.hasDeviceGlobalName(MangledName);
                       });
  }

  void *ext_oneapi_get_device_global_address(const std::string &Name,
                                             const device &Dev) const {
    DeviceGlobalMapEntry *Entry = getDeviceGlobalEntry(Name);
    device_impl &DeviceImpl = *getSyclObjImpl(Dev);

    if (!get_devices().contains(DeviceImpl)) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "kernel_bundle not built for device");
    }

    if (Entry->MIsDeviceImageScopeDecorated) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Cannot query USM pointer for device global with "
                            "'device_image_scope' property");
    }

    bool SupportContextMemcpy = false;
    DeviceImpl.getAdapter().call<UrApiKind::urDeviceGetInfo>(
        DeviceImpl.getHandleRef(),
        UR_DEVICE_INFO_USM_CONTEXT_MEMCPY_SUPPORT_EXP,
        sizeof(SupportContextMemcpy), &SupportContextMemcpy, nullptr);
    if (SupportContextMemcpy) {
      return Entry->getOrAllocateDeviceGlobalUSM(MContext).getPtr();
    } else {
      queue InitQueue{MContext, Dev};
      auto &USMMem =
          Entry->getOrAllocateDeviceGlobalUSM(*getSyclObjImpl(InitQueue));
      InitQueue.wait_and_throw();
      return USMMem.getPtr();
    }
  }

  size_t ext_oneapi_get_device_global_size(const std::string &Name) const {
    return getDeviceGlobalEntry(Name)->MDeviceGlobalTSize;
  }

  bool empty() const noexcept { return MDeviceImages.empty(); }

  backend get_backend() const noexcept { return MContext.get_backend(); }

  context get_context() const noexcept { return MContext; }

  devices_range get_devices() const noexcept { return MDevices; }

  std::vector<kernel_id> get_kernel_ids() const {
    // Collect kernel ids from all device images, then remove duplicates
    std::vector<kernel_id> Result;
    for (device_image_impl &DevImg : device_images()) {
      // RTC kernel bundles shouldn't have user-facing kernel ids, return an
      // empty vector when the bundle contains RTC kernels.
      if (DevImg.getRTCInfo())
        continue;

      auto KernelIDs = DevImg.get_kernel_ids();
      Result.insert(Result.end(), KernelIDs.begin(), KernelIDs.end());
    }
    std::sort(Result.begin(), Result.end(), LessByNameComp{});

    auto NewIt = std::unique(Result.begin(), Result.end(), EqualByNameComp{});
    Result.erase(NewIt, Result.end());

    return Result;
  }

  kernel get_kernel(const kernel_id &KernelID) const {
    if (std::shared_ptr<kernel_impl> KernelImpl = tryGetOfflineKernel(KernelID))
      return detail::createSyclObjFromImpl<kernel>(std::move(KernelImpl));
    throw sycl::exception(make_error_code(errc::invalid),
                          "The kernel bundle does not contain the kernel "
                          "identified by kernelId.");
  }

  bool has_kernel(const kernel_id &KernelID) const noexcept {
    return any_of(device_images(), [&KernelID](device_image_impl &DeviceImage) {
      return DeviceImage.has_kernel(KernelID);
    });
  }

  bool has_kernel(const kernel_id &KernelID, const device &Dev) const noexcept {
    return any_of(device_images(),
                  [&KernelID, &Dev](device_image_impl &DeviceImage) {
                    return DeviceImage.has_kernel(KernelID, Dev);
                  });
  }

  bool contains_specialization_constants() const noexcept {
    return any_of(device_images(), [](device_image_impl &DeviceImage) {
      return DeviceImage.has_specialization_constants();
    });
  }

  bool native_specialization_constant() const noexcept {
    return contains_specialization_constants() &&
           all_of(device_images(), [](device_image_impl &DeviceImage) {
             return DeviceImage.all_specialization_constant_native();
           });
  }

  bool has_specialization_constant(const char *SpecName) const noexcept {
    return any_of(device_images(), [SpecName](device_image_impl &DeviceImage) {
      return DeviceImage.has_specialization_constant(SpecName);
    });
  }

  void set_specialization_constant_raw_value(const char *SpecName,
                                             const void *Value,
                                             size_t Size) noexcept {
    if (has_specialization_constant(SpecName))
      for (device_image_impl &DeviceImage : device_images())
        DeviceImage.set_specialization_constant_raw_value(SpecName, Value);
    else {
      std::vector<unsigned char> &Val = MSpecConstValues[std::string{SpecName}];
      Val.resize(Size);
      std::memcpy(Val.data(), Value, Size);
    }
  }

  void get_specialization_constant_raw_value(const char *SpecName,
                                             void *ValueRet) const noexcept {
    for (device_image_impl &DeviceImage : device_images())
      if (DeviceImage.has_specialization_constant(SpecName)) {
        DeviceImage.get_specialization_constant_raw_value(SpecName, ValueRet);
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
    bool SetInDevImg =
        any_of(device_images(), [SpecName](device_image_impl &DeviceImage) {
          return DeviceImage.is_specialization_constant_set(SpecName);
        });
    return SetInDevImg || MSpecConstValues.count(std::string{SpecName}) != 0;
  }

  // Don't use these two for code under `source/detail`, they are only needed to
  // communicate across DSO boundary.
  const device_image_plain *begin() const { return MUniqueDeviceImages.data(); }
  const device_image_plain *end() const {
    return MUniqueDeviceImages.data() + MUniqueDeviceImages.size();
  }
  // ...use that instead.
  device_images_range device_images() const { return MUniqueDeviceImages; }

  size_t size() const noexcept { return MUniqueDeviceImages.size(); }

  bundle_state get_bundle_state() const { return MState; }

  const SpecConstMapT &get_spec_const_map_ref() const noexcept {
    return MSpecConstValues;
  }

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

  bool hasSourceBasedImages() const noexcept {
    return any_of(device_images(), [](device_image_impl &DevImg) {
      return DevImg.getOriginMask() & ImageOriginKernelCompiler;
    });
  }

  bool hasSYCLBINImages() const noexcept {
    return any_of(device_images(), [](device_image_impl &DevImg) {
      return DevImg.getOriginMask() & ImageOriginSYCLBIN;
    });
  }

  bool hasSYCLOfflineImages() const noexcept {
    return any_of(device_images(), [](device_image_impl &DevImg) {
      return DevImg.getOriginMask() & ImageOriginSYCLOffline;
    });
  }

  bool allSourceBasedImages() const noexcept {
    return all_of(device_images(), [](device_image_impl &DevImg) {
      return DevImg.getOriginMask() & ImageOriginKernelCompiler;
    });
  }

  std::shared_ptr<kernel_impl>
  tryGetOfflineKernel(const kernel_id &KernelID) const {
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
      return nullptr;

    auto UrProgram = SelectedImage->get_ur_program();
    auto [Kernel, CacheMutex, ArgMask] =
        detail::ProgramManager::getInstance().getOrCreateKernel(
            MContext, KernelID.get_name(), /*PropList=*/{}, UrProgram);

    return std::make_shared<kernel_impl>(
        std::move(Kernel), *detail::getSyclObjImpl(MContext),
        std::move(SelectedImage), *this, ArgMask, UrProgram, CacheMutex);
  }

  std::shared_ptr<kernel_impl> tryGetKernel(std::string_view Name) const {
    // TODO: For source-based kernels, it may be faster to keep a map between
    //       {kernel_name, device} and their corresponding image.
    // First look through the kernels registered in source-based images.
    for (device_image_impl &DevImg : device_images()) {
      if (std::shared_ptr<kernel_impl> SourceBasedKernel =
              DevImg.tryGetExtensionKernel(Name, MContext, *this))
        return SourceBasedKernel;
    }

    // Fall back to regular offline compiled kernel_bundle look-up.
    if (std::optional<kernel_id> MaybeKernelID =
            sycl::detail::ProgramManager::getInstance().tryGetSYCLKernelID(
                Name))
      return tryGetOfflineKernel(*MaybeKernelID);
    return nullptr;
  }

  std::shared_ptr<kernel_bundle_impl> shared_from_this() const {
    return const_cast<kernel_bundle_impl *>(this)->Base::shared_from_this();
  }

  DeviceGlobalMap &getDeviceGlobalMap() { return MDeviceGlobals; }

  std::optional<unsigned>
  tryGetKernelArgsSize(const std::string_view KernelName) const {
    auto &PM = sycl::detail::ProgramManager::getInstance();
    return PM.getKernelGlobalInfoDesc(KernelName.data());
  }

private:
  DeviceGlobalMapEntry *getDeviceGlobalEntry(const std::string &Name) const {
    if (!hasSourceBasedImages() && !hasSYCLBINImages()) {
      throw sycl::exception(make_error_code(errc::invalid),
                            "Querying device globals by name is only available "
                            "in kernel_bundles created from SYCLBIN files and "
                            "kernel_bundles successfully built from "
                            "kernel_bundle<bundle_state>::ext_oneapi_source> "
                            "with 'sycl' source language.");
    }

    std::string MangledName = mangleDeviceGlobalName(Name);

    if (MDeviceGlobals.size())
      if (DeviceGlobalMapEntry *Entry =
              MDeviceGlobals.tryGetEntryLockless(MangledName))
        return Entry;

    for (device_image_impl &DevImg : device_images())
      if (DeviceGlobalMapEntry *Entry =
              DevImg.tryGetDeviceGlobalEntry(MangledName))
        return Entry;

    throw sycl::exception(make_error_code(errc::invalid),
                          "device global '" + Name +
                              "' not found in kernel_bundle");
  }

  static std::string mangleDeviceGlobalName(const std::string &Name) {
    // TODO: Support device globals declared in namespaces.
    return "_Z" + std::to_string(Name.length()) + Name;
  }

  void populateDeviceGlobalsForSYCLBIN() {
    // This should only be called from ctors, so lockless initialization is
    // safe.
    for (device_image_impl &DevImg : device_images()) {
      if (DevImg.getOriginMask() & ImageOriginSYCLBIN)
        if (const RTDeviceBinaryImage *DevBinImg = DevImg.get_bin_image_ref())
          MDeviceGlobals.initializeEntriesLockless(DevBinImg);
    }
  }

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

  std::vector<const RTDeviceBinaryImage *>
  GetSYCLBINAOTBinaries(device_impl &Dev) {
    if (MSYCLBINs.size() == 1)
      return MSYCLBINs[0]->getNativeBinaryImages(Dev);

    std::vector<const RTDeviceBinaryImage *> Result;
    for (auto &SYCLBIN : MSYCLBINs) {
      std::vector<const RTDeviceBinaryImage *> NativeBinImgs =
          SYCLBIN->getNativeBinaryImages(Dev);
      Result.insert(Result.end(),
                    std::make_move_iterator(NativeBinImgs.begin()),
                    std::make_move_iterator(NativeBinImgs.end()));
    }

    return Result;
  }

  context MContext;
  std::vector<device_impl *> MDevices;

  // For sycl_jit, building from source may have produced sycl binaries that
  // the kernel_bundles now manage.
  // NOTE: This must appear before device images to enforce their freeing of
  //       device globals prior to unregistering the binaries.
  std::vector<std::shared_ptr<ManagedDeviceBinaries>> MSharedDeviceBinaries;

  // SYCLBINs manage their own binary information, so if we have any we store
  // them. These are stored as shared_ptr to ensure they stay alive across
  // kernel_bundles that use them.
  std::vector<std::shared_ptr<SYCLBINBinaries>> MSYCLBINs;

  std::vector<DevImgPlainWithDeps> MDeviceImages;
  std::vector<device_image_plain> MUniqueDeviceImages;
  // This map stores values for specialization constants, that are missing
  // from any device image.
  SpecConstMapT MSpecConstValues;
  bundle_state MState;

  // Map for isolating device_global variables owned by the SYCLBINs in the
  // kernel_bundle. This map will ensure the cleanup of its entries, unlike the
  // map in program_manager which has its entry cleanup managed by the
  // corresponding owner contexts.
  DeviceGlobalMap MDeviceGlobals{/*OwnerControlledCleanup=*/false};
};

inline bool is_compatible(const std::vector<kernel_id> &KernelIDs,
                          device_impl &Dev) {
  if (KernelIDs.empty())
    return true;
  // One kernel may be contained in several binary images depending on the
  // number of targets. This kernel is compatible with the device if there is
  // at least one image (containing this kernel) whose aspects are supported by
  // the device and whose target matches the device.
  for (const auto &KernelID : KernelIDs) {
    std::set<const detail::RTDeviceBinaryImage *> BinImages =
        detail::ProgramManager::getInstance().getRawDeviceImages({KernelID});

    if (std::none_of(BinImages.begin(), BinImages.end(),
                     [&](const detail::RTDeviceBinaryImage *Img) {
                       return doesDevSupportDeviceRequirements(Dev, *Img) &&
                              doesImageTargetMatchDevice(*Img, Dev);
                     }))
      return false;
  }

  return true;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
