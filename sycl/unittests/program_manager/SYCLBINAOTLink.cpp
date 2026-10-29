//==------------------- SYCLBINAOTLink.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit test for the AOT-only cross-library link path in
// kernel_bundle_impl::link (#22509 bug 3). Loading two native-AOT object-state
// SYCLBINs and linking them must NOT invoke the JIT link entry point
// (urProgramLinkExp): the AOT partition leaves the JIT image set empty, and
// passing an empty program list to urProgramLinkExp is invalid and previously
// segfaulted in the L0 adapter. The AOT images are instead each built with
// ALLOW_UNRESOLVED_SYMBOLS (urProgramCreateWithBinary + urProgramBuildExp) and
// resolved via urProgramDynamicLinkExp.
//
// This exercises the empty-JIT branch through the genuine SYCLBIN-origin graph
// path (not the statically-registered SYCLOffline images, which take the
// separate offline-link loop). It is the hardware-independent unit-level
// counterpart of the L0-gated e2e SYCLBIN/link_object_aot.cpp.
//
//===----------------------------------------------------------------------===//

#include <detail/kernel_bundle_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/syclbin.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/RuntimeLinkingCommon.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

// The importing kernel; MOCK_INTEGRATION_HEADER registers its name so the
// SYCLBIN loader can recover a kernel_id for it.
class SYCLBINAOTLinkKernel;
MOCK_INTEGRATION_HEADER(SYCLBINAOTLinkKernel)

namespace {

std::vector<sycl::unittest::MockProperty>
makeSymbolProps(const std::vector<std::string> &Symbols) {
  std::vector<sycl::unittest::MockProperty> Props;
  for (const std::string &Sym : Symbols) {
    std::vector<char> Storage(sizeof(uint32_t), 0);
    Props.emplace_back(Sym, Storage, SYCL_PROPERTY_TYPE_UINT32);
  }
  return Props;
}

// A native-AOT MockDeviceImage kept alive alongside its native struct so its
// bytes/properties stay valid while serialized. Never moved after
// construction (the native handle stores raw pointers into its own members).
class NativeImageHolder {
public:
  NativeImageHolder(const std::string &KernelName, unsigned char Magic,
                    const std::vector<std::string> &ExportedSymbols,
                    const std::vector<std::string> &ImportedSymbols)
      : MMock(makeMock(KernelName, Magic, ExportedSymbols, ImportedSymbols)),
        MStruct(MMock.convertToNativeType()), MImage(&MStruct) {}

  NativeImageHolder(NativeImageHolder &&) = delete;
  NativeImageHolder(const NativeImageHolder &) = delete;

  const sycl::detail::RTDeviceBinaryImage &image() const { return MImage; }

private:
  static sycl::unittest::MockDeviceImage
  makeMock(const std::string &KernelName, unsigned char Magic,
           const std::vector<std::string> &ExportedSymbols,
           const std::vector<std::string> &ImportedSymbols) {
    sycl::unittest::MockPropertySet PropSet;
    if (!ExportedSymbols.empty())
      PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                     makeSymbolProps(ExportedSymbols));
    if (!ImportedSymbols.empty())
      PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                     makeSymbolProps(ImportedSymbols));
    return sycl::unittest::MockDeviceImage{
        SYCL_DEVICE_BINARY_TYPE_NATIVE,
        __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
        /*CompileOptions=*/"",
        /*LinkOptions=*/"",
        std::vector<unsigned char>{Magic},
        sycl::unittest::makeEmptyKernels({KernelName}),
        std::move(PropSet)};
  }

  sycl::unittest::MockDeviceImage MMock;
  sycl_device_binary_struct MStruct;
  sycl::detail::RTDeviceBinaryImage MImage;
};

std::vector<char>
serializeObjectSYCLBIN(const sycl::detail::RTDeviceBinaryImage &Img,
                       sycl::detail::device_impl &Dev) {
  sycl::detail::SYCLBIN::ImageInput Input;
  Input.Image = &Img;
  Input.Devices = {&Dev};
  return sycl::detail::SYCLBIN::serializeImages(
      {Input}, static_cast<uint8_t>(sycl::bundle_state::object));
}

} // namespace

TEST(SYCLBINAOTLink, AOTOnlyLinkSkipsJITLink) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  setupRuntimeLinkingMock();

  sycl::context Ctx{sycl::platform().get_devices()[0]};
  std::vector<sycl::device> Devs = Ctx.get_devices();
  sycl::detail::device_impl &DevImpl = *sycl::detail::getSyclObjImpl(Devs[0]);

  // Exporter: export-only native library (classifies executable, surfaced for
  // object via the export-only fallback). Importer: native image importing the
  // exported symbol (classifies object). Distinct magic bytes so the linked
  // programs can be told apart.
  NativeImageHolder Exporter{"SYCLBINAOTLinkExporter", /*Magic=*/41,
                             /*ExportedSymbols=*/{"SYCLBINAOTLinkSym"},
                             /*ImportedSymbols=*/{}};
  NativeImageHolder Importer{"SYCLBINAOTLinkKernel", /*Magic=*/43,
                             /*ExportedSymbols=*/{},
                             /*ImportedSymbols=*/{"SYCLBINAOTLinkSym"}};

  std::vector<char> ExportBytes =
      serializeObjectSYCLBIN(Exporter.image(), DevImpl);
  std::vector<char> ImportBytes =
      serializeObjectSYCLBIN(Importer.image(), DevImpl);

  auto ExportKB = sycl::ext::oneapi::experimental::get_kernel_bundle<
      sycl::bundle_state::object>(Ctx, Devs, sycl::span<char>{ExportBytes});
  auto ImportKB = sycl::ext::oneapi::experimental::get_kernel_bundle<
      sycl::bundle_state::object>(Ctx, Devs, sycl::span<char>{ImportBytes});

  CapturedLinkingData.clear();

  sycl::kernel_bundle LinkedKB = sycl::link({ExportKB, ImportKB});

  // No SPIR-V/JIT image is present: the JIT link entry point must never be
  // reached (previously an empty program list segfaulted the L0 adapter).
  EXPECT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 0u);
  // Each AOT image is built independently...
  EXPECT_EQ(CapturedLinkingData.NumOfUrProgramCreateWithBinaryCalls, 2u);
  // ...and the cross-image references resolved via a single dynamic link.
  EXPECT_EQ(CapturedLinkingData.NumOfUrProgramDynamicLinkCalls, 1u);
  EXPECT_TRUE(CapturedLinkingData.LinkedProgramsContains({41u, 43u}));
}
