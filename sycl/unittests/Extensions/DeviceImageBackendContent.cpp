//==--- DeviceImageBackendContent.cpp - backend_content extension tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// sycl_ext_oneapi_device_image_backend_content: an executable image built from
// IR hands out the binary the backend built, while an image stored as native
// code comes back as stored.

#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

class BackendContentJITKernel;
class BackendContentNativeKernel;
MOCK_INTEGRATION_HEADER(BackendContentJITKernel)
MOCK_INTEGRATION_HEADER(BackendContentNativeKernel)

namespace {

const std::vector<unsigned char> StoredSPIRV{0x03, 0x02, 0x23,
                                             0x07, 0xAA, 0xBB};
const std::vector<unsigned char> StoredNative{0x7F, 'E',  'L', 'F',
                                              0x11, 0x22, 0x33};
const std::vector<unsigned char> BuiltNative{0x7F, 'E',  'L',  'F',  0xDE,
                                             0xAD, 0xBE, 0xEF, 0x01, 0x02};

sycl::unittest::MockDeviceImage makeImage(uint8_t Format,
                                          const std::string &Kernel,
                                          std::vector<unsigned char> Bin) {
  using namespace sycl::unittest;
  return MockDeviceImage{Format,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64,
                         /*CompileOptions=*/"",
                         /*LinkOptions=*/"",
                         std::move(Bin),
                         makeEmptyKernels({Kernel}),
                         MockPropertySet{}};
}

sycl::unittest::MockDeviceImage Imgs[] = {
    makeImage(SYCL_DEVICE_BINARY_TYPE_SPIRV, "BackendContentJITKernel",
              StoredSPIRV),
    makeImage(SYCL_DEVICE_BINARY_TYPE_NATIVE, "BackendContentNativeKernel",
              StoredNative)};
sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

int BinariesQueries = 0;

// Pretend the backend built BuiltNative for the program's single device.
ur_result_t redefinedProgramGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_program_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_PROGRAM_INFO_BINARY_SIZES:
    if (*params.ppPropValue)
      *static_cast<size_t *>(*params.ppPropValue) = BuiltNative.size();
    break;
  case UR_PROGRAM_INFO_BINARIES:
    if (*params.ppPropValue) {
      ++BinariesQueries;
      unsigned char *Dest = *static_cast<unsigned char **>(*params.ppPropValue);
      std::copy(BuiltNative.begin(), BuiltNative.end(), Dest);
    }
    break;
  default:
    break;
  }
  return UR_RESULT_SUCCESS;
}

template <typename KernelT>
sycl::device_image<sycl::bundle_state::executable>
getImage(const sycl::context &Ctx, const sycl::device &Dev) {
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, {Dev}, {sycl::get_kernel_id<KernelT>()});
  EXPECT_EQ(std::distance(KB.begin(), KB.end()), 1);
  return *KB.begin();
}

std::vector<unsigned char> toUChar(const std::vector<std::byte> &Bytes) {
  std::vector<unsigned char> Out(Bytes.size());
  std::transform(Bytes.begin(), Bytes.end(), Out.begin(),
                 [](std::byte B) { return static_cast<unsigned char>(B); });
  return Out;
}

} // namespace

TEST(DeviceImageBackendContent, JITImageReturnsBuiltNativeBinary) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedProgramGetInfoAfter);
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  BinariesQueries = 0;

  auto Img = getImage<BackendContentJITKernel>(Ctx, Dev);
  EXPECT_EQ(toUChar(Img.ext_oneapi_get_backend_content()), BuiltNative);
  EXPECT_EQ(BinariesQueries, 1);

  // Fetched once per image and then served from the image.
  EXPECT_EQ(toUChar(Img.ext_oneapi_get_backend_content()), BuiltNative);
  EXPECT_EQ(BinariesQueries, 1);
#ifdef __cpp_lib_span
  auto View1 = Img.ext_oneapi_get_backend_content_view();
  auto View2 = Img.ext_oneapi_get_backend_content_view();
  EXPECT_EQ(View1.data(), View2.data());
  EXPECT_EQ(View1.size(), BuiltNative.size());
#endif
}

TEST(DeviceImageBackendContent, NativeImageReturnsStoredBytes) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedProgramGetInfoAfter);
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  BinariesQueries = 0;

  auto Img = getImage<BackendContentNativeKernel>(Ctx, Dev);
  EXPECT_EQ(toUChar(Img.ext_oneapi_get_backend_content()), StoredNative);
  EXPECT_EQ(BinariesQueries, 0);
}

// Selector-built executable bundles hold images that were never built; the
// query must obtain the program itself.
TEST(DeviceImageBackendContent, SelectorBundleImageReturnsBuiltNativeBinary) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedProgramGetInfoAfter);
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  BinariesQueries = 0;

  sycl::kernel_id KID = sycl::get_kernel_id<BackendContentJITKernel>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, {Dev},
      [&](const sycl::device_image<sycl::bundle_state::executable> &Img) {
        return Img.has_kernel(KID);
      });
  ASSERT_EQ(std::distance(KB.begin(), KB.end()), 1);
  EXPECT_EQ(toUChar(KB.begin()->ext_oneapi_get_backend_content()), BuiltNative);
  EXPECT_EQ(BinariesQueries, 1);
}
