//==------------------- AOTBinaryTarget.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the native-AOT image classifiers added alongside the
// object-state SYCLBIN load fix (see program_manager.cpp):
//   - ProgramManager::isAOTBinaryTarget
//   - ProgramManager::getBinImageState
// Both are pure, static, and hardware-independent, so they can be exercised
// directly without a UR mock. getBinImageState is what the SYCLBIN selector
// and the kernel_bundle AOT partition rely on to agree on an image's state.
//
//===----------------------------------------------------------------------===//

#include <detail/compiler.hpp>
#include <detail/device_binary_image.hpp>
#include <detail/program_manager/program_manager.hpp>

#include <helpers/MockDeviceImage.hpp>

#include <gtest/gtest.h>

using namespace sycl;

namespace {

// Build a MockPropertySet carrying an optional "imported symbols" set (its
// presence marks a native-AOT image as still needing a link).
unittest::MockPropertySet
makePropSet(const std::vector<std::string> &ImportedSymbols) {
  unittest::MockPropertySet PropSet;
  if (!ImportedSymbols.empty()) {
    std::vector<unittest::MockProperty> Props;
    for (const std::string &Sym : ImportedSymbols) {
      std::vector<char> Storage(sizeof(uint32_t), 0);
      Props.emplace_back(Sym, Storage, SYCL_PROPERTY_TYPE_UINT32);
    }
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS, std::move(Props));
  }
  return PropSet;
}

// Build a real RTDeviceBinaryImage from a MockDeviceImage so we can query
// getBinImageState/getImportedSymbols on it.
//
// MockDeviceImage's native handle stores raw pointers into the mock's own
// string members, so the mock must not be moved after construction (a move
// would leave those pointers dangling). We therefore construct the mock in
// place inside the holder and rely on guaranteed copy elision (C++17 prvalue
// return) so the holder itself is never moved after MImage captures &MStruct.
class ImageHolder {
public:
  ImageHolder(const char *TargetSpec, sycl::detail::ur::DeviceBinaryType Format,
              const std::vector<std::string> &ImportedSymbols)
      : MMock(static_cast<uint8_t>(Format), TargetSpec,
              /*CompileOptions=*/"", /*LinkOptions=*/"",
              std::vector<unsigned char>{0},
              unittest::makeEmptyKernels({"AOTBinaryTargetKernel"}),
              makePropSet(ImportedSymbols)),
        MStruct(MMock.convertToNativeType()), MImage(&MStruct) {}

  ImageHolder(ImageHolder &&) = delete;
  ImageHolder(const ImageHolder &) = delete;

  const detail::RTDeviceBinaryImage &image() const { return MImage; }

private:
  unittest::MockDeviceImage MMock;
  sycl_device_binary_struct MStruct;
  detail::RTDeviceBinaryImage MImage;
};

} // namespace

TEST(AOTBinaryTarget, IsAOTBinaryTarget) {
  // Null spec must not dereference.
  EXPECT_FALSE(detail::ProgramManager::isAOTBinaryTarget(nullptr));

  // The two native-AOT targets.
  EXPECT_TRUE(detail::ProgramManager::isAOTBinaryTarget(
      __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN));
  EXPECT_TRUE(detail::ProgramManager::isAOTBinaryTarget(
      __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64));

  // JIT SPIR-V and every non-native-AOT target must be excluded, otherwise the
  // kernel_bundle partition would route SPIR-V through the AOT dynamic-link
  // path and break urProgramLinkExp.
  EXPECT_FALSE(detail::ProgramManager::isAOTBinaryTarget(
      __SYCL_DEVICE_BINARY_TARGET_SPIRV64));
  EXPECT_FALSE(detail::ProgramManager::isAOTBinaryTarget(
      __SYCL_DEVICE_BINARY_TARGET_SPIRV32));
  EXPECT_FALSE(detail::ProgramManager::isAOTBinaryTarget(
      __SYCL_DEVICE_BINARY_TARGET_NVPTX64));
  EXPECT_FALSE(detail::ProgramManager::isAOTBinaryTarget(
      __SYCL_DEVICE_BINARY_TARGET_AMDGCN));
  EXPECT_FALSE(detail::ProgramManager::isAOTBinaryTarget(
      __SYCL_DEVICE_BINARY_TARGET_NATIVE_CPU));
  EXPECT_FALSE(detail::ProgramManager::isAOTBinaryTarget(
      __SYCL_DEVICE_BINARY_TARGET_UNKNOWN));
}

TEST(AOTBinaryTarget, GetBinImageStateNonAOT) {
  // A JIT SPIR-V image is always in the input state regardless of symbols.
  ImageHolder JIT{__SYCL_DEVICE_BINARY_TARGET_SPIRV64,
                  SYCL_DEVICE_BINARY_TYPE_SPIRV, /*ImportedSymbols=*/{"Dep"}};
  EXPECT_EQ(detail::ProgramManager::getBinImageState(&JIT.image()),
            bundle_state::input);
}

TEST(AOTBinaryTarget, GetBinImageStateAOTNoImports) {
  // A native-AOT image with no imported symbols is already executable.
  ImageHolder AOT{__SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
                  SYCL_DEVICE_BINARY_TYPE_NATIVE, /*ImportedSymbols=*/{}};
  EXPECT_EQ(detail::ProgramManager::getBinImageState(&AOT.image()),
            bundle_state::executable);
}

TEST(AOTBinaryTarget, GetBinImageStateAOTWithImports) {
  // A native-AOT image carrying imported symbols still needs a link, so it is
  // classified as object. This is the exact case the SYCLBIN selector fix
  // relies on: an AOT-only object SYCLBIN loaded as bundle_state::object.
  ImageHolder AOT{__SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
                  SYCL_DEVICE_BINARY_TYPE_NATIVE, /*ImportedSymbols=*/{"Dep"}};
  EXPECT_EQ(detail::ProgramManager::getBinImageState(&AOT.image()),
            bundle_state::object);
}
