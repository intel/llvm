//==----------------------- USMMemcpy2D.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <detail/queue_impl.hpp>

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

constexpr const char *USMFillHelperKernelNameLong = "__usmfill2d_long";
constexpr const char *USMFillHelperKernelNameChar = "__usmfill2d_char";
constexpr const char *USMMemcpyHelperKernelNameLong = "__usmmemcpy2d_long";
constexpr const char *USMMemcpyHelperKernelNameChar = "__usmmemcpy2d_char";

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<class __usmfill2d<long>> {
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
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() {
    return 2 * sizeof(void *) + 2 * sizeof(sycl::id<2>) + 3 * sizeof(size_t);
  }
};

template <> struct KernelInfo<class __usmfill2d<unsigned char>> {
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
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() {
    return 2 * sizeof(void *) + 2 * sizeof(sycl::id<2>) + 3 * sizeof(size_t);
  }
};

template <> struct KernelInfo<class __usmmemcpy2d<long>> {
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
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() {
    return 2 * sizeof(void *) + 2 * sizeof(sycl::id<2>) + 4 * sizeof(size_t);
  }
};

template <> struct KernelInfo<class __usmmemcpy2d<unsigned char>> {
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
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() {
    return 2 * sizeof(void *) + 2 * sizeof(sycl::id<2>) + 4 * sizeof(size_t);
  }
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

static sycl::unittest::PiImage generateMemopsImage() {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

  std::vector<unsigned char> Bin{10, 11, 12, 13, 14, 15}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels(
      {USMFillHelperKernelNameLong, USMFillHelperKernelNameChar,
       USMMemcpyHelperKernelNameLong, USMMemcpyHelperKernelNameChar});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

namespace {
sycl::unittest::PiImage Imgs[] = {generateMemopsImage()};
sycl::unittest::PiImageArray<1> ImgArray{Imgs};

size_t LastMemopsQuery = 0;

struct Fill2DStruct {
  pi_queue queue;
  void *ptr;
  size_t pitch;
  size_t pattern_size;
  const void *pattern;
  size_t width;
  size_t height;
  pi_uint32 num_events_in_waitlist;
  const pi_event *events_waitlist;
  pi_event *event;
} LastFill2D;

struct Memset2DStruct {
  pi_queue queue;
  void *ptr;
  size_t pitch;
  int value;
  size_t width;
  size_t height;
  pi_uint32 num_events_in_waitlist;
  const pi_event *events_waitlist;
  pi_event *event;
} LastMemset2D;

struct Memcpy2DStruct {
  pi_queue queue;
  pi_bool blocking;
  void *dst_ptr;
  size_t dst_pitch;
  const void *src_ptr;
  size_t src_pitch;
  size_t width;
  size_t height;
  pi_uint32 num_events_in_waitlist;
  const pi_event *events_waitlist;
  pi_event *event;
} LastMemcpy2D;

std::map<pi_kernel, std::string> KernelToNameMap;

template <bool MemfillSupported, bool MemsetSupported, bool MemcpySupported>
pi_result after_piContextGetInfo(pi_context context, pi_context_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT:
    LastMemopsQuery = param_name;
    if (param_value)
      *static_cast<pi_bool *>(param_value) = MemfillSupported;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_bool);
    return PI_SUCCESS;
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT:
    LastMemopsQuery = param_name;
    if (param_value)
      *static_cast<pi_bool *>(param_value) = MemsetSupported;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_bool);
    return PI_SUCCESS;
  case PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT:
    LastMemopsQuery = param_name;
    if (param_value)
      *static_cast<pi_bool *>(param_value) = MemcpySupported;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_bool);
    return PI_SUCCESS;
  default:;
  }

  return PI_SUCCESS;
}

pi_result after_piDeviceGetInfo(pi_device device, pi_device_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES:
    if (param_value) {
      assert(param_value_size == 3 * sizeof(size_t));
      size_t *Ptr = static_cast<size_t *>(param_value);
      Ptr[0] = 32;
      Ptr[1] = 32;
      Ptr[2] = 32;
    }
    if (param_value_size_ret)
      *param_value_size_ret = 3 * sizeof(size_t);
    return PI_SUCCESS;
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS:
    if (param_value) {
      assert(param_value_size == sizeof(pi_uint32));
      *static_cast<pi_uint32 *>(param_value) = 256;
    }
    if (param_value_size_ret)
      *param_value_size_ret = 3 * sizeof(size_t);
    return PI_SUCCESS;
  default:;
  }

  return PI_SUCCESS;
}

pi_result redefine_piextUSMEnqueueFill2D(pi_queue queue, void *ptr,
                                         size_t pitch, size_t pattern_size,
                                         const void *pattern, size_t width,
                                         size_t height,
                                         pi_uint32 num_events_in_waitlist,
                                         const pi_event *events_waitlist,
                                         pi_event *event) {
  LastFill2D =
      Fill2DStruct{queue,           ptr,   pitch,  pattern_size,
                   pattern,         width, height, num_events_in_waitlist,
                   events_waitlist, event};
  return PI_SUCCESS;
}

pi_result redefine_piextUSMEnqueueMemset2D(pi_queue queue, void *ptr,
                                           size_t pitch, int value,
                                           size_t width, size_t height,
                                           pi_uint32 num_events_in_waitlist,
                                           const pi_event *events_waitlist,
                                           pi_event *event) {
  LastMemset2D = Memset2DStruct{queue,
                                ptr,
                                pitch,
                                value,
                                width,
                                height,
                                num_events_in_waitlist,
                                events_waitlist,
                                event};
  return PI_SUCCESS;
}

pi_result redefine_piextUSMEnqueueMemcpy2D(
    pi_queue queue, pi_bool blocking, void *dst_ptr, size_t dst_pitch,
    const void *src_ptr, size_t src_pitch, size_t width, size_t height,
    pi_uint32 num_events_in_waitlist, const pi_event *events_waitlist,
    pi_event *event) {
  LastMemcpy2D =
      Memcpy2DStruct{queue,           blocking, dst_ptr,
                     dst_pitch,       src_ptr,  src_pitch,
                     width,           height,   num_events_in_waitlist,
                     events_waitlist, event};
  return PI_SUCCESS;
}

pi_result after_piKernelCreate(pi_program, const char *kernel_name,
                               pi_kernel *ret_kernel) {
  KernelToNameMap[*ret_kernel] = kernel_name;
  return PI_SUCCESS;
}

std::string LastEnqueuedKernel;

pi_result after_piEnqueueKernelLaunch(pi_queue, pi_kernel kernel, pi_uint32,
                                      const size_t *, const size_t *,
                                      const size_t *, pi_uint32,
                                      const pi_event *, pi_event *) {
  auto KernelIt = KernelToNameMap.find(kernel);
  EXPECT_TRUE(KernelIt != KernelToNameMap.end());
  LastEnqueuedKernel = KernelIt->second;
  return PI_SUCCESS;
}
} // namespace

// Tests that the right APIs are called when they are reported as supported
// natively.
TEST(USMMemcpy2DTest, USMMemops2DSupported) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q{Plt.get_devices()[0]};

  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(Q);

  Mock.redefineAfter<sycl::detail::PiApiKind::piContextGetInfo>(
      after_piContextGetInfo<true, true, true>);
  Mock.redefine<sycl::detail::PiApiKind::piextUSMEnqueueFill2D>(
      redefine_piextUSMEnqueueFill2D);
  Mock.redefine<sycl::detail::PiApiKind::piextUSMEnqueueMemset2D>(
      redefine_piextUSMEnqueueMemset2D);
  Mock.redefine<sycl::detail::PiApiKind::piextUSMEnqueueMemcpy2D>(
      redefine_piextUSMEnqueueMemcpy2D);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastFill2D.queue, (pi_queue)QueueImpl->getHandleRef());
  EXPECT_EQ(LastFill2D.ptr, (void *)Ptr1);
  EXPECT_EQ(LastFill2D.pitch, (size_t)5);
  EXPECT_EQ(LastFill2D.pattern_size, sizeof(long));
  EXPECT_EQ(LastFill2D.width, (size_t)4);
  EXPECT_EQ(LastFill2D.height, (size_t)2);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT);
  EXPECT_EQ(LastMemset2D.queue, (pi_queue)QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemset2D.ptr, (void *)Ptr1);
  EXPECT_EQ(LastMemset2D.pitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemset2D.value, 123);
  EXPECT_EQ(LastMemset2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemset2D.height, (size_t)2);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastMemcpy2D.queue, (pi_queue)QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemcpy2D.dst_ptr, (void *)Ptr1);
  EXPECT_EQ(LastMemcpy2D.dst_pitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.src_ptr, (void *)Ptr2);
  EXPECT_EQ(LastMemcpy2D.src_pitch, (size_t)8 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.height, (size_t)2);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastMemcpy2D.queue, (pi_queue)QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemcpy2D.dst_ptr, (void *)Ptr2);
  EXPECT_EQ(LastMemcpy2D.dst_pitch, (size_t)8 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.src_ptr, (void *)Ptr1);
  EXPECT_EQ(LastMemcpy2D.src_pitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.height, (size_t)2);
}

// Tests that the right fallback kernels are called when a backend does not
// support the APIs natively.
TEST(USMMemcpy2DTest, USMMemops2DUnsupported) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q{Plt.get_devices()[0]};

  Mock.redefineAfter<sycl::detail::PiApiKind::piContextGetInfo>(
      after_piContextGetInfo<false, false, false>);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piKernelCreate>(
      after_piKernelCreate);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueKernelLaunch>(
      after_piEnqueueKernelLaunch);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameLong);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameChar);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameChar);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameLong);
}

// Tests that the right paths are taken when the backend only supports native
// USM fill.
TEST(USMMemcpy2DTest, USMFillSupportedOnly) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q{Plt.get_devices()[0]};

  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(Q);

  Mock.redefineAfter<sycl::detail::PiApiKind::piContextGetInfo>(
      after_piContextGetInfo<true, false, false>);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piKernelCreate>(
      after_piKernelCreate);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueKernelLaunch>(
      after_piEnqueueKernelLaunch);
  Mock.redefine<sycl::detail::PiApiKind::piextUSMEnqueueFill2D>(
      redefine_piextUSMEnqueueFill2D);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastFill2D.queue, (pi_queue)QueueImpl->getHandleRef());
  EXPECT_EQ(LastFill2D.ptr, (void *)Ptr1);
  EXPECT_EQ(LastFill2D.pitch, (size_t)5);
  EXPECT_EQ(LastFill2D.pattern_size, sizeof(long));
  EXPECT_EQ(LastFill2D.width, (size_t)4);
  EXPECT_EQ(LastFill2D.height, (size_t)2);
  EXPECT_NE(LastEnqueuedKernel, USMFillHelperKernelNameLong);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameChar);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameChar);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameLong);
}

// Tests that the right paths are taken when the backend only supports native
// USM memset.
TEST(USMMemcpy2DTest, USMMemsetSupportedOnly) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q{Plt.get_devices()[0]};

  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(Q);

  Mock.redefineAfter<sycl::detail::PiApiKind::piContextGetInfo>(
      after_piContextGetInfo<false, true, false>);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piKernelCreate>(
      after_piKernelCreate);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueKernelLaunch>(
      after_piEnqueueKernelLaunch);
  Mock.redefine<sycl::detail::PiApiKind::piextUSMEnqueueMemset2D>(
      redefine_piextUSMEnqueueMemset2D);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameLong);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT);
  EXPECT_EQ(LastMemset2D.queue, (pi_queue)QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemset2D.ptr, (void *)Ptr1);
  EXPECT_EQ(LastMemset2D.pitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemset2D.value, 123);
  EXPECT_EQ(LastMemset2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemset2D.height, (size_t)2);
  EXPECT_NE(LastEnqueuedKernel, USMFillHelperKernelNameChar);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameChar);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMMemcpyHelperKernelNameLong);
}

// Tests that the right paths are taken when the backend only supports native
// USM memcpy.
TEST(USMMemcpy2DTest, USMMemcpySupportedOnly) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q{Plt.get_devices()[0]};

  std::shared_ptr<sycl::detail::queue_impl> QueueImpl =
      sycl::detail::getSyclObjImpl(Q);

  Mock.redefineAfter<sycl::detail::PiApiKind::piContextGetInfo>(
      after_piContextGetInfo<false, false, true>);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piKernelCreate>(
      after_piKernelCreate);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueKernelLaunch>(
      after_piEnqueueKernelLaunch);
  Mock.redefine<sycl::detail::PiApiKind::piextUSMEnqueueMemcpy2D>(
      redefine_piextUSMEnqueueMemcpy2D);

  long *Ptr1 = sycl::malloc_device<long>(10, Q);
  long *Ptr2 = sycl::malloc_device<long>(16, Q);

  Q.ext_oneapi_fill2d(Ptr1, 5, 42l, 4, 2);
  EXPECT_TRUE(LastMemopsQuery == PI_EXT_ONEAPI_CONTEXT_INFO_USM_FILL2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameLong);

  Q.ext_oneapi_memset2d(Ptr1, 5 * sizeof(long), 123, 4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMSET2D_SUPPORT);
  EXPECT_EQ(LastEnqueuedKernel, USMFillHelperKernelNameChar);

  Q.ext_oneapi_memcpy2d(Ptr1, 5 * sizeof(long), Ptr2, 8 * sizeof(long),
                        4 * sizeof(long), 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastMemcpy2D.queue, (pi_queue)QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemcpy2D.dst_ptr, (void *)Ptr1);
  EXPECT_EQ(LastMemcpy2D.dst_pitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.src_ptr, (void *)Ptr2);
  EXPECT_EQ(LastMemcpy2D.src_pitch, (size_t)8 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.height, (size_t)2);
  EXPECT_NE(LastEnqueuedKernel, USMMemcpyHelperKernelNameChar);

  Q.ext_oneapi_copy2d(Ptr1, 5, Ptr2, 8, 4, 2);
  EXPECT_TRUE(LastMemopsQuery ==
              PI_EXT_ONEAPI_CONTEXT_INFO_USM_MEMCPY2D_SUPPORT);
  EXPECT_EQ(LastMemcpy2D.queue, (pi_queue)QueueImpl->getHandleRef());
  EXPECT_EQ(LastMemcpy2D.dst_ptr, (void *)Ptr2);
  EXPECT_EQ(LastMemcpy2D.dst_pitch, (size_t)8 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.src_ptr, (void *)Ptr1);
  EXPECT_EQ(LastMemcpy2D.src_pitch, (size_t)5 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.width, (size_t)4 * sizeof(long));
  EXPECT_EQ(LastMemcpy2D.height, (size_t)2);
  EXPECT_NE(LastEnqueuedKernel, USMMemcpyHelperKernelNameLong);
}

// Negative tests for cases where USM 2D memory operations are expected to throw
// exceptions.
TEST(USMMemcpy2DTest, NegativeUSM2DOps) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
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
