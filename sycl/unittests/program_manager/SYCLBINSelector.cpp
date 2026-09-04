//==------------------- SYCLBINSelector.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for SYCLBINBinaries::getBestCompatibleImages, the selector that
// decides which image inside a loaded SYCLBIN is surfaced for a requested
// bundle_state (see syclbin.cpp). The native-AOT fallback branches added for
// the object-state SYCLBIN load fix are otherwise only exercised by
// Level-Zero-gated e2e tests (SYCLBIN/aot_object_load.cpp,
// SYCLBIN/link_object_aot.cpp); this drives the selector directly with a
// synthetic, hardware-independent SYCLBIN so a non-L0 coverage run reaches it.
//
// Covered branches:
//   1. Native-AOT image with imported symbols -> classified object, surfaced
//      for an object-state request (#22196).
//   2. Export-only native-AOT library (exports, no imports) -> classified
//      executable, but still surfaced for an object-state request so it can
//      act as the provider side of a cross-library link (#22509 bug 1).
//   3. Native-AOT image surfaced for an executable-state request (baseline
//      preferred-native path).
//
//===----------------------------------------------------------------------===//

#include <detail/compiler.hpp>
#include <detail/device_binary_image.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/syclbin.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace sycl;

namespace {

// Build a MockPropertySet carrying optional exported/imported symbol sets.
unittest::MockPropertySet
makeSymbolPropSet(const std::vector<std::string> &ExportedSymbols,
                  const std::vector<std::string> &ImportedSymbols) {
  auto makeProps = [](const std::vector<std::string> &Symbols) {
    std::vector<unittest::MockProperty> Props;
    for (const std::string &Sym : Symbols) {
      std::vector<char> Storage(sizeof(uint32_t), 0);
      Props.emplace_back(Sym, Storage, SYCL_PROPERTY_TYPE_UINT32);
    }
    return Props;
  };
  unittest::MockPropertySet PropSet;
  if (!ExportedSymbols.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                   makeProps(ExportedSymbols));
  if (!ImportedSymbols.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                   makeProps(ImportedSymbols));
  return PropSet;
}

// Holds a native-AOT MockDeviceImage and its native struct alive so the image
// bytes/props remain valid while it is serialized into a SYCLBIN. The mock
// must not be moved after construction (its native handle stores raw pointers
// into its own members); constructed in place, never moved.
class NativeImageHolder {
public:
  NativeImageHolder(const std::string &KernelName,
                    const std::vector<std::string> &ExportedSymbols,
                    const std::vector<std::string> &ImportedSymbols)
      : MMock(static_cast<uint8_t>(SYCL_DEVICE_BINARY_TYPE_NATIVE),
              __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
              /*CompileOptions=*/"", /*LinkOptions=*/"",
              std::vector<unsigned char>{0xDE, 0xAD, 0xBE, 0xEF},
              unittest::makeEmptyKernels({KernelName}),
              makeSymbolPropSet(ExportedSymbols, ImportedSymbols)),
        MStruct(MMock.convertToNativeType()), MImage(&MStruct) {}

  NativeImageHolder(NativeImageHolder &&) = delete;
  NativeImageHolder(const NativeImageHolder &) = delete;

  const detail::RTDeviceBinaryImage &image() const { return MImage; }

private:
  unittest::MockDeviceImage MMock;
  sycl_device_binary_struct MStruct;
  detail::RTDeviceBinaryImage MImage;
};

// Serialize a single native-AOT image into a SYCLBIN blob at the given state.
std::vector<char> serializeNativeSYCLBIN(const detail::RTDeviceBinaryImage &Img,
                                         detail::device_impl &Dev,
                                         bundle_state State) {
  detail::SYCLBIN::ImageInput Input;
  Input.Image = &Img;
  Input.Devices = {&Dev};
  return detail::SYCLBIN::serializeImages({Input}, static_cast<uint8_t>(State));
}

detail::device_impl &getMockDevice() {
  return *detail::getSyclObjImpl(platform().get_devices()[0]);
}

} // namespace

// Branch 1: a native-AOT image with imported symbols classifies as object and
// must be surfaced for an object-state request.
TEST(SYCLBINSelector, NativeObjectWithImportsSurfacedForObject) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  NativeImageHolder Holder{"SYCLBINSelectorImporter",
                           /*ExportedSymbols=*/{},
                           /*ImportedSymbols=*/{"ProvidedSym"}};
  std::vector<char> Bytes =
      serializeNativeSYCLBIN(Holder.image(), Dev, bundle_state::object);

  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
  std::vector<const detail::RTDeviceBinaryImage *> Selected =
      Binaries.getBestCompatibleImages(Dev, bundle_state::object);

  ASSERT_EQ(Selected.size(), 1u)
      << "Native-AOT object image must be surfaced for an object request";
  EXPECT_EQ(detail::ProgramManager::getBinImageState(Selected[0]),
            bundle_state::object);
}

// Branch 2: an export-only native-AOT library classifies as executable but
// must still be surfaced for an object-state request (it is the provider side
// of a cross-library link).
TEST(SYCLBINSelector, ExportOnlyLibrarySurfacedForObject) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  NativeImageHolder Holder{"SYCLBINSelectorExporter",
                           /*ExportedSymbols=*/{"ProvidedSym"},
                           /*ImportedSymbols=*/{}};
  std::vector<char> Bytes =
      serializeNativeSYCLBIN(Holder.image(), Dev, bundle_state::object);

  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
  std::vector<const detail::RTDeviceBinaryImage *> Selected =
      Binaries.getBestCompatibleImages(Dev, bundle_state::object);

  ASSERT_EQ(Selected.size(), 1u)
      << "Export-only native-AOT library must be surfaced for an object "
         "request even though it classifies as executable";
  // Its intrinsic classification is still executable; the object-state
  // downgrade happens later in the SYCLBIN loader (ReconcileState), not here.
  EXPECT_EQ(detail::ProgramManager::getBinImageState(Selected[0]),
            bundle_state::executable);
  EXPECT_FALSE(Selected[0]->getExportedSymbols().empty());
  EXPECT_TRUE(Selected[0]->getImportedSymbols().empty());
}

// Baseline: an executable-state request surfaces the native image via the
// preferred-native path.
TEST(SYCLBINSelector, NativeExecutableSurfacedForExecutable) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  NativeImageHolder Holder{"SYCLBINSelectorExec",
                           /*ExportedSymbols=*/{},
                           /*ImportedSymbols=*/{}};
  std::vector<char> Bytes =
      serializeNativeSYCLBIN(Holder.image(), Dev, bundle_state::executable);

  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
  std::vector<const detail::RTDeviceBinaryImage *> Selected =
      Binaries.getBestCompatibleImages(Dev, bundle_state::executable);

  ASSERT_EQ(Selected.size(), 1u);
  EXPECT_EQ(detail::ProgramManager::getBinImageState(Selected[0]),
            bundle_state::executable);
}

// Negative: an export-only library is not surfaced for an input-state request
// (input prefers JIT/SPIR-V and there is none; the object-only downgrade must
// not leak to input).
TEST(SYCLBINSelector, ExportOnlyLibraryNotSurfacedForInput) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  NativeImageHolder Holder{"SYCLBINSelectorExporterInput",
                           /*ExportedSymbols=*/{"ProvidedSym"},
                           /*ImportedSymbols=*/{}};
  // The SYCLBIN's declared state must match the request path under test; load
  // as object (its natural state) but query for input.
  std::vector<char> Bytes =
      serializeNativeSYCLBIN(Holder.image(), Dev, bundle_state::object);

  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
  std::vector<const detail::RTDeviceBinaryImage *> Selected =
      Binaries.getBestCompatibleImages(Dev, bundle_state::input);

  EXPECT_TRUE(Selected.empty())
      << "Export-only native library must not be surfaced for an input "
         "request";
}
