//==----------------------- USMMemcpy2D.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <detail/queue_impl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

constexpr const char *USMFillHelperKernelNameLong = "__usmfill2d_long";
constexpr const char *USMFillHelperKernelNameChar = "__usmfill2d_char";
constexpr const char *USMMemcpyHelperKernelNameLong = "__usmmemcpy2d_long";
constexpr const char *USMMemcpyHelperKernelNameChar = "__usmmemcpy2d_char";

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<class __usmfill2d<long>>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return USMFillHelperKernelNameLong; }
  static constexpr unsigned getNumParams() { return 7; }
  static const kernel_param_desc_t &getParamDesc(int Idx) {
    // Actual signature does not matter, but we need entries for each param.
    static constexpr const kernel_param_desc_t DummySignature[] = {
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
    };
    return DummySignature[Idx];
  }
  static constexpr int64_t getKernelSize() {
    return 2 * sizeof(void *) + 2 * sizeof(sycl::id<2>) + 3 * sizeof(size_t);
  }
};

template <>
struct KernelInfo<class __usmfill2d<unsigned char>>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return USMFillHelperKernelNameChar; }
  static constexpr unsigned getNumParams() { return 7; }
  static const kernel_param_desc_t &getParamDesc(int Idx) {
    // Actual signature does not matter, but we need entries for each param.
    static constexpr const kernel_param_desc_t DummySignature[] = {
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
    };
    return DummySignature[Idx];
  }
  static constexpr int64_t getKernelSize() {
    return 2 * sizeof(void *) + 2 * sizeof(sycl::id<2>) + 3 * sizeof(size_t);
  }
};

template <>
struct KernelInfo<class __usmmemcpy2d<long>>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return USMMemcpyHelperKernelNameLong;
  }
  static constexpr unsigned getNumParams() { return 8; }
  static const kernel_param_desc_t &getParamDesc(int Idx) {
    // Actual signature does not matter, but we need entries for each param.
    static constexpr const kernel_param_desc_t DummySignature[] = {
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
    };
    return DummySignature[Idx];
  }
  static constexpr int64_t getKernelSize() {
    return 2 * sizeof(void *) + 2 * sizeof(sycl::id<2>) + 4 * sizeof(size_t);
  }
};

template <>
struct KernelInfo<class __usmmemcpy2d<unsigned char>>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return USMMemcpyHelperKernelNameChar;
  }
  static constexpr unsigned getNumParams() { return 8; }
  static const kernel_param_desc_t &getParamDesc(int Idx) {
    // Actual signature does not matter, but we need entries for each param.
    static constexpr const kernel_param_desc_t DummySignature[] = {
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
        {kernel_param_kind_t::kind_std_layout, 4, 0},
    };
    return DummySignature[Idx];
  }
  static constexpr int64_t getKernelSize() {
    return 2 * sizeof(void *) + 2 * sizeof(sycl::id<2>) + 4 * sizeof(size_t);
  }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace {
sycl::unittest::MockDeviceImage Imgs[] = {sycl::unittest::generateDefaultImage(
    {USMFillHelperKernelNameLong, USMFillHelperKernelNameChar,
     USMMemcpyHelperKernelNameLong, USMMemcpyHelperKernelNameChar})};
sycl::unittest::MockDeviceImageArray<1> ImgArray{Imgs};

ur_context_info_t LastMemopsQuery = UR_CONTEXT_INFO_NUM_DEVICES;

struct Fill2dParams {
  ur_queue_handle_t hQueue;
  void *pMem;
  size_t pitch;
  size_t patternSize;
  std::vector<char> pattern;
  size_t width;
  size_t height;
} LastFill2D;

struct Memcpy2dParams {
  ur_queue_handle_t hQueue;
  void *pDst;
  size_t dstPitch;
  const void *pSrc;
  size_t srcPitch;
  size_t width;
  size_t height;
} LastMemcpy2D;

std::map<ur_kernel_handle_t, std::string> KernelToNameMap;

template <bool MemfillSupported, bool MemsetSupported, bool MemcpySupported>
ur_result_t after_urContextGetInfo(void *pParams) {
  auto params = *static_cast<ur_context_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_CONTEXT_INFO_USM_FILL2D_SUPPORT:
    LastMemopsQuery = *params.ppropName;
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = MemfillSupported;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    return UR_RESULT_SUCCESS;
  case UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    LastMemopsQuery = *params.ppropName;
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = MemcpySupported;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    return UR_RESULT_SUCCESS;
  default:;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES:
    if (*params.ppPropValue) {
      assert(*params.ppropSize == 3 * sizeof(size_t));
      size_t *Ptr = static_cast<size_t *>(*params.ppPropValue);
      Ptr[0] = 32;
      Ptr[1] = 32;
      Ptr[2] = 32;
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = 3 * sizeof(size_t);
    return UR_RESULT_SUCCESS;
  case UR_DEVICE_INFO_MAX_COMPUTE_UNITS:
    if (*params.ppPropValue) {
      assert(*params.ppropSize == sizeof(uint32_t));
      *static_cast<uint32_t *>(*params.ppPropValue) = 256;
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = 3 * sizeof(size_t);
    return UR_RESULT_SUCCESS;
  default:;
  }

  return UR_RESULT_SUCCESS;
}

template <ur_usm_type_t USMType>
ur_result_t after_urUSMGetMemAllocInfo(void *pParams) {
  auto params = *static_cast<ur_usm_get_mem_alloc_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_USM_ALLOC_INFO_TYPE: {
    if (*params.ppPropValue) {
      assert(*params.ppropSize == sizeof(ur_usm_type_t));
      *static_cast<ur_usm_type_t *>(*params.ppPropValue) = USMType;
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_usm_type_t);
    return UR_RESULT_SUCCESS;
  }
  default:;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t redefine_urEnqueueUSMFill2D(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_fill_2d_params_t *>(pParams);
  LastFill2D = Fill2dParams{*params.phQueue,
                            *params.ppMem,
                            *params.ppitch,
                            *params.ppatternSize,
                            std::vector<char>(*params.ppatternSize),
                            *params.pwidth,
                            *params.pheight};
  std::memcpy(LastFill2D.pattern.data(), *params.ppPattern,
              *params.ppatternSize);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefine_urEnqueueUSMMemcpy2D(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_memcpy_2d_params_t *>(pParams);
  LastMemcpy2D = Memcpy2dParams{
      *params.phQueue,   *params.ppDst,  *params.pdstPitch, *params.ppSrc,
      *params.psrcPitch, *params.pwidth, *params.pheight};
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urKernelCreate(void *pParams) {
  auto params = *static_cast<ur_kernel_create_params_t *>(pParams);
  KernelToNameMap[**params.pphKernel] = *params.ppKernelName;
  return UR_RESULT_SUCCESS;
}

std::string LastEnqueuedKernel;

ur_result_t after_urEnqueueKernelLaunch(void *pParams) {
  auto params = *static_cast<ur_enqueue_kernel_launch_params_t *>(pParams);
  auto KernelIt = KernelToNameMap.find(*params.phKernel);
  EXPECT_TRUE(KernelIt != KernelToNameMap.end());
  LastEnqueuedKernel = KernelIt->second;
  return UR_RESULT_SUCCESS;
}
} // namespace

// Tests that the right APIs are called when they are reported as supported
// natively.
TEST(USMMemcpy2DTest, USMMemops2DSupported) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue Q{Plt.get_devices()[0]};

  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(Q);

  mock::getCallbacks().set_after_callback(
      "urContextGetInfo", &after_urContextGetInfo<true, true, true>);
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill2D",
                                            &redefine_urEnqueueUSMFill2D);
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy2D",
                                            &redefine_urEnqueueUSMMemcpy2D);
  mock::getCallbacks().set_after_callback(
      "urUSMGetMemAllocInfo", &after_urUSMGetMemAllocInfo<UR_USM_TYPE_DEVICE>);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastFill2D.hQueue, (ur_queue_handle_t)QueueImpl->getHandleRef());
  EXPECT_EQ(LastFill2D.pMem, (void *)Ptr1);
  EXPECT_EQ(LastFill2D.pitch, (size_t)5);
  EXPECT_EQ(LastFill2D.patternSize, sizeof(long));
  EXPECT_EQ(LastFill2D.width, (size_t)4);
  EXPECT_EQ(LastFill2D.height, (size_t)2);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastFill2D.hQueue, (ur_queue_handle_t)QueueImpl->getHandleRef());
  EXPECT_EQ(LastFill2D.pMem, (void *)Ptr1);
  EXPECT_EQ(LastFill2D.pitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastFill2D.pattern[0], 123);
  EXPECT_EQ(LastFill2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastFill2D.height, (size_t)2);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastMemcpy2D.hQueue, (ur_queue_handle_t)QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemcpy2D.pDst, (void *)Ptr1);
  EXPECT_EQ(LastMemcpy2D.dstPitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.pSrc, (void *)Ptr2);
  EXPECT_EQ(LastMemcpy2D.srcPitch, (size_t)8 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.height, (size_t)2);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastMemcpy2D.hQueue, (ur_queue_handle_t)QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemcpy2D.pDst, (void *)Ptr2);
  EXPECT_EQ(LastMemcpy2D.dstPitch, (size_t)8 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.pSrc, (void *)Ptr1);
  EXPECT_EQ(LastMemcpy2D.srcPitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.height, (size_t)2);
}

// Tests that the right fallback kernels are called when a backend does not
// support the APIs natively.
TEST(USMMemcpy2DTest, USMMemops2DUnsupported) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue Q{Plt.get_devices()[0]};

  mock::getCallbacks().set_after_callback(
      "urContextGetInfo", &after_urContextGetInfo<false, false, false>);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);
  mock::getCallbacks().set_after_callback("urKernelCreate",
                                          &after_urKernelCreate);
  mock::getCallbacks().set_after_callback("urEnqueueKernelLaunch",
                                          &after_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback(
      "urUSMGetMemAllocInfo", &after_urUSMGetMemAllocInfo<UR_USM_TYPE_DEVICE>);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameLong);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameChar);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameChar);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameLong);
}

// Tests that the right paths are taken when the backend only supports native
// USM fill.
TEST(USMMemcpy2DTest, USMFillSupportedOnly) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue Q{Plt.get_devices()[0]};

  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(Q);

  mock::getCallbacks().set_after_callback(
      "urContextGetInfo", &after_urContextGetInfo<true, false, false>);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);
  mock::getCallbacks().set_after_callback("urKernelCreate",
                                          &after_urKernelCreate);
  mock::getCallbacks().set_after_callback("urEnqueueKernelLaunch",
                                          &after_urEnqueueKernelLaunch);
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill2D",
                                            &redefine_urEnqueueUSMFill2D);
  mock::getCallbacks().set_after_callback(
      "urUSMGetMemAllocInfo", &after_urUSMGetMemAllocInfo<UR_USM_TYPE_DEVICE>);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastFill2D.hQueue, QueueImpl->getHandleRef());
  EXPECT_EQ(LastFill2D.pMem, (void *)Ptr1);
  EXPECT_EQ(LastFill2D.pitch, (size_t)5);
  EXPECT_EQ(LastFill2D.patternSize, sizeof(long));
  EXPECT_EQ(LastFill2D.width, (size_t)4);
  EXPECT_EQ(LastFill2D.height, (size_t)2);
  EXPECT_NE(LastEnqueuedKernel, USMFillHelperKernelNameLong);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameChar);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameLong);
}

// Tests that the right paths are taken when the backend only supports native
// USM memset.
TEST(USMMemcpy2DTest, USMMemsetSupportedOnly) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue Q{Plt.get_devices()[0]};

  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(Q);

  // Enable fill + set, they are implemented with the same entry point in the
  // backend so supporting one means supporting both.
  mock::getCallbacks().set_after_callback(
      "urContextGetInfo", &after_urContextGetInfo<true, true, false>);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);
  mock::getCallbacks().set_after_callback("urKernelCreate",
                                          &after_urKernelCreate);
  mock::getCallbacks().set_after_callback("urEnqueueKernelLaunch",
                                          &after_urEnqueueKernelLaunch);
  mock::getCallbacks().set_after_callback(
      "urUSMGetMemAllocInfo", &after_urUSMGetMemAllocInfo<UR_USM_TYPE_DEVICE>);
  mock::getCallbacks().set_replace_callback("urEnqueueUSMFill2D",
                                            &redefine_urEnqueueUSMFill2D);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastFill2D.hQueue, QueueImpl->getHandleRef());
  EXPECT_EQ(LastFill2D.pMem, (void *)Ptr1);
  EXPECT_EQ(LastFill2D.pitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastFill2D.pattern[0], 123);
  EXPECT_EQ(LastFill2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastFill2D.height, (size_t)2);
  EXPECT_NE(LastEnqueuedKernel, USMFillHelperKernelNameChar);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameChar);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameLong);
}

// Tests that the right paths are taken when the backend only supports native
// USM memcpy.
TEST(USMMemcpy2DTest, USMMemcpySupportedOnly) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue Q{Plt.get_devices()[0]};

  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(Q);

  mock::getCallbacks().set_after_callback(
      "urContextGetInfo", &after_urContextGetInfo<false, false, true>);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);
  mock::getCallbacks().set_after_callback("urKernelCreate",
                                          &after_urKernelCreate);
  mock::getCallbacks().set_after_callback("urEnqueueKernelLaunch",
                                          &after_urEnqueueKernelLaunch);
  mock::getCallbacks().set_replace_callback("urEnqueueUSMMemcpy2D",
                                            &redefine_urEnqueueUSMMemcpy2D);
  mock::getCallbacks().set_after_callback(
      "urUSMGetMemAllocInfo", &after_urUSMGetMemAllocInfo<UR_USM_TYPE_DEVICE>);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameLong);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameChar);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastMemcpy2D.hQueue, QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemcpy2D.pDst, (void *)Ptr1);
  EXPECT_EQ(LastMemcpy2D.dstPitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.pSrc, (void *)Ptr2);
  EXPECT_EQ(LastMemcpy2D.srcPitch, (size_t)8 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.height, (size_t)2);
  EXPECT_NE(LastEnqueuedKernel, USMMemcpyHelperKernelNameChar);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == UR_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastMemcpy2D.hQueue, QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemcpy2D.pDst, (void *)Ptr2);
  EXPECT_EQ(LastMemcpy2D.dstPitch, (size_t)8 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.pSrc, (void *)Ptr1);
  EXPECT_EQ(LastMemcpy2D.srcPitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.height, (size_t)2);
  EXPECT_NE(LastEnqueuedKernel, USMMemcpyHelperKernelNameLong);
}

// Negative tests for cases where USM 2D memory operations are expected to throw
// exceptions.
TEST(USMMemcpy2DTest, NegativeUSM2DOps) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  sycl::queue Q{Plt.get_devices()[0]};

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  try {
    Q.ext_oneapi_fill2d(Ptr1, 3, 42l, 4, 2);
    FAIL() << "No exception thrown with invalid pitch.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid))
        << "Unexpected error code for ext_oneapi_fill2d with invalid pitch.";
  }

  try {
    Q.ext_oneapi_memset2d(Ptr1, 3 * sizeof(long), 123, 4 * sizeof(long), 2);
    FAIL() << "No exception thrown with invalid pitch.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid))
        << "Unexpected error code for ext_oneapi_memset2d with invalid pitch.";
  }

  try {
    Q.ext_oneapi_memcpy2d(Ptr1, 3 * sizeof(long), Ptr2, 8 * sizeof(long),
                          4 * sizeof(long), 2);
    FAIL() << "No exception thrown with invalid source pitch.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid))
        << "Unexpected error code for ext_oneapi_memcpy2d with invalid "
           "destination pitch.";
  }

  try {
    Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 3 * sizeof(long),
                          4 * sizeof(long), 2);
    FAIL() << "No exception thrown with invalid destination pitch.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid))
        << "Unexpected error code for ext_oneapi_memcpy2d with invalid source "
           "pitch.";
  }

  try {
    Q.ext_oneapi_copy2d(Ptr1, 3, Ptr2, 8, 4, 2);
    FAIL() << "No exception thrown with invalid source pitch.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid))
        << "Unexpected error code for ext_oneapi_copy2d with invalid source "
           "pitch.";
  }

  try {
    Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 3, 4, 2);
    FAIL() << "No exception thrown with invalid destination pitch.";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid))
        << "Unexpected error code for ext_oneapi_copy2d with invalid "
           "destination pitch.";
  }
}
