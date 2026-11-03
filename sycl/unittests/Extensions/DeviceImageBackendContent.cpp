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
#include <cstring>
#include <vector>

class BackendContentJITKernel;
class BackendContentNativeKernel;
class BackendContentCudaKernel;
class BackendContentGpuOnlyKernel;
MOCK_INTEGRATION_HEADER(BackendContentJITKernel)
MOCK_INTEGRATION_HEADER(BackendContentNativeKernel)
MOCK_INTEGRATION_HEADER(BackendContentCudaKernel)
MOCK_INTEGRATION_HEADER(BackendContentGpuOnlyKernel)

namespace {

const std::vector<unsigned char> StoredSPIRV{0x03, 0x02, 0x23,
                                             0x07, 0xAA, 0xBB};
const std::vector<unsigned char> StoredNative{0x7F, 'E',  'L', 'F',
                                              0x11, 0x22, 0x33};
const std::vector<unsigned char> BuiltNative{0x7F, 'E',  'L',  'F',  0xDE,
                                             0xAD, 0xBE, 0xEF, 0x01, 0x02};
const std::vector<unsigned char> StoredPTX{'/', '/', 'p', 't', 'x', 0x00};

sycl::unittest::MockDeviceImage
makeImage(uint8_t Format, const char *Target, const std::string &Kernel,
          std::vector<unsigned char> Bin,
          sycl::unittest::MockPropertySet PropSet = {}) {
  using namespace sycl::unittest;
  return MockDeviceImage{Format,
                         Target,
                         /*CompileOptions=*/"",
                         /*LinkOptions=*/"",
                         std::move(Bin),
                         makeEmptyKernels({Kernel}),
                         std::move(PropSet)};
}

// [SYCL/device requirements] with a single required aspect.
sycl::unittest::MockPropertySet makeAspectRequirement(sycl::aspect Aspect) {
  using namespace sycl::unittest;
  // 8-byte size prefix, then one 32-bit aspect id.
  std::vector<char> Data(8 + sizeof(uint32_t));
  const uint32_t Id = static_cast<uint32_t>(Aspect);
  std::memcpy(Data.data() + 8, &Id, sizeof(Id));
  MockPropertySet PropSet;
  PropSet.insert(
      __SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS,
      MockProperty("aspects", std::move(Data), SYCL_PROPERTY_TYPE_BYTE_ARRAY));
  return PropSet;
}

sycl::unittest::MockDeviceImage Imgs[] = {
    makeImage(SYCL_DEVICE_BINARY_TYPE_SPIRV,
              __SYCL_DEVICE_BINARY_TARGET_SPIRV64, "BackendContentJITKernel",
              StoredSPIRV),
    makeImage(SYCL_DEVICE_BINARY_TYPE_NATIVE,
              __SYCL_DEVICE_BINARY_TARGET_SPIRV64, "BackendContentNativeKernel",
              StoredNative),
    makeImage(SYCL_DEVICE_BINARY_TYPE_NONE, __SYCL_DEVICE_BINARY_TARGET_NVPTX64,
              "BackendContentCudaKernel", StoredPTX),
    makeImage(SYCL_DEVICE_BINARY_TYPE_SPIRV,
              __SYCL_DEVICE_BINARY_TARGET_SPIRV64,
              "BackendContentGpuOnlyKernel", StoredSPIRV,
              makeAspectRequirement(sycl::aspect::gpu))};
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

// Two mock devices: handle 1 is a CPU, handle 2 a GPU.
constexpr uintptr_t CpuDevice = 1, GpuDevice = 2;
ur_device_handle_t BuiltForDevice = nullptr;

ur_result_t redefinedDeviceGetTwoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices && *params.pNumEntries >= 2)
    for (uintptr_t I = 0; I < 2; ++I)
      (*params.pphDevices)[I] = reinterpret_cast<ur_device_handle_t>(I + 1);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDeviceGetInfoTwoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_TYPE && *params.ppPropValue) {
    const bool IsCpu =
        reinterpret_cast<uintptr_t>(*params.phDevice) == CpuDevice;
    *static_cast<ur_device_type_t *>(*params.ppPropValue) =
        IsCpu ? UR_DEVICE_TYPE_CPU : UR_DEVICE_TYPE_GPU;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedProgramBuildExpAfter(void *pParams) {
  auto params = *static_cast<ur_program_build_exp_params_t *>(pParams);
  if (*params.pnumDevices > 0)
    BuiltForDevice = (*params.pphDevices)[0];
  return UR_RESULT_SUCCESS;
}

// Like redefinedProgramGetInfoAfter, but the program belongs to whichever
// device it was built for.
ur_result_t redefinedProgramGetInfoTwoAfter(void *pParams) {
  auto params = *static_cast<ur_program_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_PROGRAM_INFO_DEVICES && *params.ppPropValue) {
    *static_cast<ur_device_handle_t *>(*params.ppPropValue) = BuiltForDevice;
    return UR_RESULT_SUCCESS;
  }
  return redefinedProgramGetInfoAfter(pParams);
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

// Backends whose binary query is not trusted keep the stored bytes.
TEST(DeviceImageBackendContent, OtherBackendReturnsStoredBytes) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_cuda> Mock;
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedProgramGetInfoAfter);
  sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  BinariesQueries = 0;

  auto Img = getImage<BackendContentCudaKernel>(Ctx, Dev);
  EXPECT_EQ(toUChar(Img.ext_oneapi_get_backend_content()), StoredPTX);
  EXPECT_EQ(BinariesQueries, 0);
}

// A selector bundle over a CPU and a GPU keeps both devices on an image that
// only the GPU can run; the lazy build must target the GPU.
TEST(DeviceImageBackendContent, SelectorBundleBuildsForCompatibleDevice) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGet",
                                          &redefinedDeviceGetTwoAfter);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoTwoAfter);
  mock::getCallbacks().set_after_callback("urProgramBuildExp",
                                          &redefinedProgramBuildExpAfter);
  mock::getCallbacks().set_after_callback("urProgramGetInfo",
                                          &redefinedProgramGetInfoTwoAfter);
  BinariesQueries = 0;
  BuiltForDevice = nullptr;

  std::vector<sycl::device> Devs = sycl::platform().get_devices();
  ASSERT_EQ(Devs.size(), 2u);
  ASSERT_TRUE(Devs[0].is_cpu());
  ASSERT_TRUE(Devs[1].is_gpu());
  sycl::context Ctx{Devs};

  sycl::kernel_id KID = sycl::get_kernel_id<BackendContentGpuOnlyKernel>();
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, Devs,
      [&](const sycl::device_image<sycl::bundle_state::executable> &Img) {
        return Img.has_kernel(KID);
      });
  ASSERT_EQ(std::distance(KB.begin(), KB.end()), 1);
  EXPECT_EQ(toUChar(KB.begin()->ext_oneapi_get_backend_content()), BuiltNative);
  EXPECT_EQ(BuiltForDevice, reinterpret_cast<ur_device_handle_t>(GpuDevice));
  EXPECT_EQ(BinariesQueries, 1);
}
