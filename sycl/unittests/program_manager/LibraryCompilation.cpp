//==------------------- LibraryCompilation.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the "-library-compilation" compile-option injection added to
// ProgramManager::compile (see program_manager.cpp). When an input image was
// produced with -fsycl-allow-device-image-dependencies it carries SYCL
// exported/imported symbol property sets; on Level Zero, symbol visibility
// must be kept through compile (via -library-compilation) so cross-image
// resolution at sycl::link time (zeModuleDynamicLink) can see those symbols.
// The flag is only plumbed for Level Zero (see the backend guard in
// ProgramManager::compile); every other backend must not receive it. Cross-
// image AOT link on other backends (e.g. OpenCL CPU) is out of scope and not
// verified here.
//
// These tests capture the option string passed to urProgramCompileExp and
// assert the flag is present iff (backend == L0 && image has cross-image
// symbols).
//
//===----------------------------------------------------------------------===//

#include <detail/kernel_bundle_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>

// Two kernels: one whose image carries cross-image symbols (an exported
// symbol), one whose image carries none. MOCK_INTEGRATION_HEADER provides both
// the KernelInfo specialization and getKernelNameHelper, which the
// get_kernel_bundle<KernelName, state> template requires.
class LibCompileKernel;
class PlainKernel;
MOCK_INTEGRATION_HEADER(LibCompileKernel)
MOCK_INTEGRATION_HEADER(PlainKernel)

namespace {

std::vector<sycl::unittest::MockProperty>
makeSymbolProps(const std::vector<std::string> &Symbols) {
  std::vector<sycl::unittest::MockProperty> Props;
  for (const std::string &Symbol : Symbols) {
    std::vector<char> Storage(sizeof(uint32_t), 0);
    uint32_t Val = 1;
    std::memcpy(Storage.data(), &Val, sizeof(uint32_t));
    Props.emplace_back(Symbol, Storage, SYCL_PROPERTY_TYPE_UINT32);
  }
  return Props;
}

// A JIT SPIR-V image (bundle_state::input) for the given kernel, optionally
// carrying an exported symbol so that ProgramManager::compile sees it as a
// cross-image-symbol image. An exported symbol (rather than an imported one)
// is used so the image is self-contained: it has cross-image symbols but no
// unresolved dependency to satisfy during compile.
sycl::unittest::MockDeviceImage
makeImage(const std::string &KernelName,
          const std::vector<std::string> &ExportedSymbols) {
  sycl::unittest::MockPropertySet PropSet;
  if (!ExportedSymbols.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                   makeSymbolProps(ExportedSymbols));

  std::vector<unsigned char> Bin{0};
  return sycl::unittest::MockDeviceImage{
      SYCL_DEVICE_BINARY_TYPE_SPIRV,
      __SYCL_DEVICE_BINARY_TARGET_SPIRV64,
      /*CompileOptions=*/"",
      /*LinkOptions=*/"",
      std::move(Bin),
      sycl::unittest::makeEmptyKernels({KernelName}),
      std::move(PropSet)};
}

sycl::unittest::MockDeviceImage Imgs[] = {
    makeImage("LibCompileKernel", /*ExportedSymbols=*/{"LibCompileKernelSym"}),
    makeImage("PlainKernel", /*ExportedSymbols=*/{})};
sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

std::string CapturedCompileOptions;

ur_result_t captureCompileOptions(void *pParams) {
  auto Params = *static_cast<ur_program_compile_exp_params_t *>(pParams);
  if (*Params.ppOptions)
    CapturedCompileOptions = std::string(*Params.ppOptions);
  return UR_RESULT_SUCCESS;
}

// Compile the input bundle for KernelName and return the option string that
// reached urProgramCompileExp.
template <typename KernelName>
std::string compileAndCaptureOptions(sycl::context &Ctx) {
  CapturedCompileOptions.clear();
  mock::getCallbacks().set_replace_callback("urProgramCompileExp",
                                            &captureCompileOptions);

  sycl::kernel_bundle InputKB =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::input>(Ctx);
  sycl::kernel_bundle ObjectKB = sycl::compile(InputKB);
  (void)ObjectKB;
  return CapturedCompileOptions;
}

bool hasLibraryCompilationFlag(const std::string &Options) {
  return Options.find("-library-compilation") != std::string::npos;
}

} // namespace

// On Level Zero, an image carrying cross-image symbols must be compiled with
// -library-compilation so the symbols remain visible for the later dynamic
// link.
TEST(LibraryCompilation, L0WithSymbolsAddsFlag) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  sycl::context Ctx{sycl::platform().get_devices()[0]};

  std::string Options = compileAndCaptureOptions<LibCompileKernel>(Ctx);
  EXPECT_TRUE(hasLibraryCompilationFlag(Options))
      << "Expected -library-compilation for an L0 image with cross-image "
         "symbols, got: '"
      << Options << "'";
}

// The flag is L0-specific: an OpenCL context must not receive it even when the
// image carries cross-image symbols. The backend guard in
// ProgramManager::compile only injects the flag for Level Zero; the OpenCL
// cross-image AOT path is out of scope.
TEST(LibraryCompilation, OpenCLWithSymbolsNoFlag) {
  sycl::unittest::UrMock<sycl::backend::opencl> Mock;
  sycl::context Ctx{sycl::platform().get_devices()[0]};

  std::string Options = compileAndCaptureOptions<LibCompileKernel>(Ctx);
  EXPECT_FALSE(hasLibraryCompilationFlag(Options))
      << "Did not expect -library-compilation for a non-L0 backend, got: '"
      << Options << "'";
}

// An L0 image with no cross-image symbols is an ordinary translation unit and
// must not receive the library-compilation flag.
TEST(LibraryCompilation, L0NoSymbolsNoFlag) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  sycl::context Ctx{sycl::platform().get_devices()[0]};

  std::string Options = compileAndCaptureOptions<PlainKernel>(Ctx);
  EXPECT_FALSE(hasLibraryCompilationFlag(Options))
      << "Did not expect -library-compilation for an image without "
         "cross-image symbols, got: '"
      << Options << "'";
}
