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
//   4. Native image with no symbols at all -> classified executable, still
//      surfaced for an object-state request, as it is the object-state content
//      of the SYCLBIN it was loaded from.
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

#include <algorithm>
#include <sstream>
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

// Branch 4: a native image with neither imported nor exported symbols
// classifies as executable, but it is the object-state content of the SYCLBIN
// it was loaded from, so an object-state request must surface it. The loader
// has already rejected a request whose state differs from the SYCLBIN's own
// declared state, and it presents the image as object (see ReconcileState),
// so the intrinsic classification must not veto the selection here.
// The same situation arises for a target the backend still JIT-compiles, whose
// images classify as input rather than executable.
TEST(SYCLBINSelector, NativeWithoutSymbolsSurfacedForObject) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  NativeImageHolder Holder{"SYCLBINSelectorPlain",
                           /*ExportedSymbols=*/{},
                           /*ImportedSymbols=*/{}};
  std::vector<char> Bytes =
      serializeNativeSYCLBIN(Holder.image(), Dev, bundle_state::object);

  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
  std::vector<const detail::RTDeviceBinaryImage *> Selected =
      Binaries.getBestCompatibleImages(Dev, bundle_state::object);

  ASSERT_EQ(Selected.size(), 1u)
      << "Native image without symbols must be surfaced for an object request "
         "even though it classifies as executable";
  EXPECT_EQ(detail::ProgramManager::getBinImageState(Selected[0]),
            bundle_state::executable);
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

namespace {

// Serialize a single SPIR-V IR module into an input-state SYCLBIN blob.
std::vector<char> serializeSPIRVSYCLBIN(detail::device_impl &Dev) {
  unittest::MockDeviceImage Mock{unittest::makeEmptyKernels({"IRModuleKernel"}),
                                 unittest::MockPropertySet{}};
  sycl_device_binary_struct Struct = Mock.convertToNativeType();
  detail::RTDeviceBinaryImage Img{&Struct};
  detail::SYCLBIN::ImageInput Input;
  Input.Image = &Img;
  Input.Devices = {&Dev};
  return detail::SYCLBIN::serializeImages(
      {Input}, static_cast<uint8_t>(bundle_state::input));
}

// Replaces the first occurrence of From in Bytes with an equally sized To. If
// RequireUnique is set, it must be the only occurrence.
void patchBytes(std::vector<char> &Bytes, std::string_view From,
                std::string_view To, bool RequireUnique = true) {
  ASSERT_EQ(From.size(), To.size());
  std::string_view View{Bytes.data(), Bytes.size()};
  size_t Pos = View.find(From);
  ASSERT_NE(Pos, std::string_view::npos);
  if (RequireUnique)
    ASSERT_EQ(View.find(From, Pos + 1), std::string_view::npos);
  std::copy(To.begin(), To.end(), Bytes.begin() + Pos);
}

} // namespace

// An IR module is written with the SPIR-V type and loaded back as SPIR-V.
TEST(SYCLBINIRModuleType, SPIRVRoundTrip) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  std::vector<char> Bytes = serializeSPIRVSYCLBIN(Dev);
  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
  std::vector<const detail::RTDeviceBinaryImage *> Selected =
      Binaries.getBestCompatibleImages(Dev, bundle_state::input);

  ASSERT_EQ(Selected.size(), 1u);
  EXPECT_EQ(Selected[0]->getFormat(), SYCL_DEVICE_BINARY_TYPE_SPIRV);
}

// A file whose IR module metadata has no type is treated as SPIR-V.
TEST(SYCLBINIRModuleType, MissingTypeDefaultsToSPIRV) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  std::vector<char> Bytes = serializeSPIRVSYCLBIN(Dev);
  ASSERT_NO_FATAL_FAILURE(patchBytes(Bytes, "type=1|0", "tyqe=1|0"));
  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
  std::vector<const detail::RTDeviceBinaryImage *> Selected =
      Binaries.getBestCompatibleImages(Dev, bundle_state::input);

  ASSERT_EQ(Selected.size(), 1u);
  EXPECT_EQ(Selected[0]->getFormat(), SYCL_DEVICE_BINARY_TYPE_SPIRV);
}

// PTX and AMDGCN IR modules (and unknown types) are not supported by the
// runtime and must be skipped instead of being loaded as SPIR-V.
TEST(SYCLBINIRModuleType, UnsupportedTypeIsSkipped) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  for (const char *Type : {"type=1|1", "type=1|2", "type=1|7"}) {
    std::vector<char> Bytes = serializeSPIRVSYCLBIN(Dev);
    ASSERT_NO_FATAL_FAILURE(patchBytes(Bytes, "type=1|0", Type));
    detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
    EXPECT_TRUE(
        Binaries.getBestCompatibleImages(Dev, bundle_state::input).empty())
        << Type;
  }
}

// An IR module of an unsupported type is skipped without affecting the other
// modules of the file: the remaining SPIR-V module and native image are still
// loaded and selected.
TEST(SYCLBINIRModuleType, UnsupportedTypeKeepsOtherModules) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  unittest::MockDeviceImage SkippedMock{
      unittest::makeEmptyKernels({"SkippedKernel"}),
      unittest::MockPropertySet{}};
  unittest::MockDeviceImage KeptMock{unittest::makeEmptyKernels({"KeptKernel"}),
                                     unittest::MockPropertySet{}};
  sycl_device_binary_struct SkippedStruct = SkippedMock.convertToNativeType();
  sycl_device_binary_struct KeptStruct = KeptMock.convertToNativeType();
  detail::RTDeviceBinaryImage SkippedImg{&SkippedStruct};
  detail::RTDeviceBinaryImage KeptImg{&KeptStruct};
  NativeImageHolder Native{"NativeKernel", /*ExportedSymbols=*/{},
                           /*ImportedSymbols=*/{}};

  std::vector<detail::SYCLBIN::ImageInput> Inputs(3);
  Inputs[0].Image = &SkippedImg;
  Inputs[1].Image = &KeptImg;
  Inputs[2].Image = &Native.image();
  for (detail::SYCLBIN::ImageInput &Input : Inputs)
    Input.Devices = {&Dev};
  std::vector<char> Bytes = detail::SYCLBIN::serializeImages(
      Inputs, static_cast<uint8_t>(bundle_state::object));

  auto CountFormat =
      [](const std::vector<const detail::RTDeviceBinaryImage *> &Images,
         detail::ur::DeviceBinaryType Format) {
        return std::count_if(Images.begin(), Images.end(),
                             [&](const detail::RTDeviceBinaryImage *Img) {
                               return Img->getFormat() == Format;
                             });
      };

  // Both SPIR-V modules and the native image are selected before patching.
  {
    detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
    std::vector<const detail::RTDeviceBinaryImage *> Selected =
        Binaries.getBestCompatibleImages(Dev, bundle_state::object);
    ASSERT_EQ(Selected.size(), 3u);
    EXPECT_EQ(CountFormat(Selected, SYCL_DEVICE_BINARY_TYPE_SPIRV), 2);
    EXPECT_EQ(CountFormat(Selected, SYCL_DEVICE_BINARY_TYPE_NATIVE), 1);
  }

  // Mark the first IR module as PTX. Only that module is dropped.
  ASSERT_NO_FATAL_FAILURE(patchBytes(Bytes, "type=1|0", "type=1|1",
                                     /*RequireUnique=*/false));
  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};
  std::vector<const detail::RTDeviceBinaryImage *> Selected =
      Binaries.getBestCompatibleImages(Dev, bundle_state::object);
  ASSERT_EQ(Selected.size(), 2u);
  EXPECT_EQ(CountFormat(Selected, SYCL_DEVICE_BINARY_TYPE_SPIRV), 1);
  EXPECT_EQ(CountFormat(Selected, SYCL_DEVICE_BINARY_TYPE_NATIVE), 1);
}

namespace {

std::string makeIRModuleMetadata(detail::SYCLBIN::IRModuleType Type) {
  detail::PropertySetRegistry Props;
  Props.add(detail::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "type",
            static_cast<uint32_t>(Type));
  Props.add(detail::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "target",
            std::string_view{"spirv64-unknown-unknown"});
  std::ostringstream OS;
  Props.write(OS);
  return OS.str();
}

std::string makeNativeMetadata() {
  detail::PropertySetRegistry Props;
  Props.add(
      detail::PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
      "arch", std::string_view{"unused"});
  Props.add(
      detail::PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
      "target", std::string_view{"spir64_gen-unknown-unknown"});
  std::ostringstream OS;
  Props.write(OS);
  return OS.str();
}

std::string_view getBytes(const detail::RTDeviceBinaryImage *Img) {
  const sycl_device_binary_struct &Raw = Img->getRawData();
  return {reinterpret_cast<const char *>(Raw.BinaryStart),
          static_cast<size_t>(Raw.BinaryEnd - Raw.BinaryStart)};
}

} // namespace

// IR modules of unsupported types are skipped within an abstract module that
// also holds supported IR modules and native images, e.g. a SYCLBIN built for
// several targets or by a newer compiler. The JIT binary count covers only the
// kept modules, so the kept SPIR-V module is found and the native image is
// still read from its own slot.
TEST(SYCLBINIRModuleType, UnsupportedTypesSkippedWithinAbstractModule) {
  unittest::UrMock<backend::ext_oneapi_level_zero> Mock;
  detail::device_impl &Dev = getMockDevice();

  using IRType = detail::SYCLBIN::IRModuleType;
  const std::string PTXMeta = makeIRModuleMetadata(IRType::PTX);
  const std::string AMDGCNMeta = makeIRModuleMetadata(IRType::AMDGCN);
  const std::string SPIRVMeta = makeIRModuleMetadata(IRType::SPIRV);
  const std::string NativeMeta = makeNativeMetadata();

  std::ostringstream EmptyOS;
  detail::PropertySetRegistry{}.write(EmptyOS);

  detail::SYCLBIN::SYCLBINDesc Desc;
  {
    detail::PropertySetRegistry GlobalProps;
    GlobalProps.add(detail::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA,
                    "state", static_cast<uint32_t>(bundle_state::object));
    std::ostringstream OS;
    GlobalProps.write(OS);
    Desc.GlobalMetadata = OS.str();
  }

  // Module 0: unsupported modules surround the only supported one.
  {
    auto &AMD = Desc.AbstractModules.emplace_back();
    AMD.Metadata = EmptyOS.str();
    AMD.IRModules.push_back({PTXMeta, "PTXX"});
    AMD.IRModules.push_back({SPIRVMeta, "SPRV"});
    AMD.IRModules.push_back({AMDGCNMeta, "AMDG"});
    AMD.NativeDeviceCodeImages.push_back({NativeMeta, "NAT0"});
  }
  // Module 1: no supported IR module is left, only the native image.
  {
    auto &AMD = Desc.AbstractModules.emplace_back();
    AMD.Metadata = EmptyOS.str();
    AMD.IRModules.push_back({PTXMeta, "PTXX"});
    AMD.NativeDeviceCodeImages.push_back({NativeMeta, "NAT1"});
  }
  std::vector<char> Bytes = detail::SYCLBIN::write(Desc);

  detail::SYCLBINBinaries Binaries{Bytes.data(), Bytes.size()};

  // Input state: only the SPIR-V module is usable.
  {
    std::vector<const detail::RTDeviceBinaryImage *> Selected =
        Binaries.getBestCompatibleImages(Dev, bundle_state::input);
    ASSERT_EQ(Selected.size(), 1u);
    EXPECT_EQ(Selected[0]->getFormat(), SYCL_DEVICE_BINARY_TYPE_SPIRV);
    EXPECT_EQ(getBytes(Selected[0]), "SPRV");
  }

  // Object state: JIT binaries are preferred, and module 1 falls back to its
  // native image.
  {
    std::vector<const detail::RTDeviceBinaryImage *> Selected =
        Binaries.getBestCompatibleImages(Dev, bundle_state::object);
    ASSERT_EQ(Selected.size(), 2u);
    EXPECT_EQ(Selected[0]->getFormat(), SYCL_DEVICE_BINARY_TYPE_SPIRV);
    EXPECT_EQ(getBytes(Selected[0]), "SPRV");
    EXPECT_EQ(Selected[1]->getFormat(), SYCL_DEVICE_BINARY_TYPE_NATIVE);
    EXPECT_EQ(getBytes(Selected[1]), "NAT1");
  }

  // Executable state: native images are preferred.
  {
    std::vector<const detail::RTDeviceBinaryImage *> Selected =
        Binaries.getBestCompatibleImages(Dev, bundle_state::executable);
    ASSERT_EQ(Selected.size(), 2u);
    EXPECT_EQ(getBytes(Selected[0]), "NAT0");
    EXPECT_EQ(getBytes(Selected[1]), "NAT1");
  }
}
