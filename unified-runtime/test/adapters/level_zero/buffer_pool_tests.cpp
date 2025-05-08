// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

static inline bool isWithinRange(const void *ptr, const void *startPtr,
                                 size_t size) {
  uintptr_t ptrAddr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t startAddr = reinterpret_cast<uintptr_t>(startPtr);
  uintptr_t endAddr = startAddr + size;

  return ptrAddr >= startAddr && ptrAddr < endAddr;
}

static ur_result_t urUSMHostAllocWrapped(ur_context_handle_t Context,
                                         ur_device_handle_t,
                                         const ur_usm_desc_t *USMDesc,
                                         ur_usm_pool_handle_t Pool, size_t Size,
                                         void **RetMem) {
  return urUSMHostAlloc(Context, USMDesc, Pool, Size, RetMem);
}

using AllocFunc = ur_result_t (*)(ur_context_handle_t, ur_device_handle_t,
                                  const ur_usm_desc_t *, ur_usm_pool_handle_t,
                                  size_t, void **);
using CheckSupportFunc = ur_result_t (*)(
    ur_device_handle_t, ur_device_usm_access_capability_flags_t &);

struct BufferPoolTestParam {
  AllocFunc USMAllocFunc;
  CheckSupportFunc checkUSMSupportFunc;
};

std::ostream &operator<<(std::ostream &os, BufferPoolTestParam param) {
  if (param.USMAllocFunc == urUSMHostAllocWrapped) {
    os << " urUSMHostAlloc";
  } else if (param.USMAllocFunc == urUSMDeviceAlloc) {
    os << " urUSMDeviceAlloc";
  } else if (param.USMAllocFunc == urUSMSharedAlloc) {
    os << " urUSMSharedAlloc";
  }

  return os;
}

struct urL0BufferPoolTest : uur::urQueueTestWithParam<BufferPoolTestParam> {
  void ValidateAlloc(void *ptr, size_t size) {
    ur_event_handle_t fillEvent = nullptr;
    uint8_t fillPattern = 0xAF;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(fillPattern),
                                    &fillPattern, size, 0, nullptr,
                                    &fillEvent));
    ASSERT_SUCCESS(urQueueFlush(queue));
    ASSERT_SUCCESS(urEventWait(1, &fillEvent));
    EXPECT_SUCCESS(urEventRelease(fillEvent));
  }
};

static std::unordered_map<AllocFunc, ur_usm_type_t> AllocFuncsToUSMType = {
    {urUSMHostAllocWrapped, UR_USM_TYPE_HOST},
    {urUSMSharedAlloc, UR_USM_TYPE_SHARED},
    {urUSMDeviceAlloc, UR_USM_TYPE_DEVICE},
};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urL0BufferPoolTest,
    ::testing::ValuesIn({BufferPoolTestParam{urUSMHostAllocWrapped,
                                             uur::GetDeviceUSMHostSupport},
                         BufferPoolTestParam{
                             urUSMSharedAlloc,
                             uur::GetDeviceUSMSingleSharedSupport},
                         BufferPoolTestParam{urUSMDeviceAlloc,
                                             uur::GetDeviceUSMDeviceSupport}}),
    uur::deviceTestWithParamPrinter<BufferPoolTestParam>);

TEST_P(urL0BufferPoolTest, SuccessRepeat) {
  const auto USMAllocFunc = std::get<1>(this->GetParam()).USMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  ur_device_handle_t hDevice =
      (USMAllocFunc == urUSMHostAllocWrapped) ? nullptr : device;

  void *bufPtr = nullptr;
  size_t bufSize = 1 * 1024 * 1024; // 1MB
  size_t allocSize = sizeof(int);

  ASSERT_SUCCESS(
      USMAllocFunc(context, hDevice, nullptr, nullptr, bufSize, &bufPtr));
  ASSERT_NE(bufPtr, nullptr);

  // Set buffer pool descriptor buffer to the USM allocation
  ur_usm_pool_buffer_desc_t bufferPoolDesc{};
  bufferPoolDesc.stype = UR_STRUCTURE_TYPE_USM_POOL_BUFFER_DESC;
  bufferPoolDesc.pNext = nullptr;
  bufferPoolDesc.pMem = bufPtr;
  bufferPoolDesc.size = bufSize;
  bufferPoolDesc.memType = AllocFuncsToUSMType[USMAllocFunc];
  bufferPoolDesc.device = hDevice;

  ur_usm_pool_desc_t poolDesc{};
  poolDesc.stype = UR_STRUCTURE_TYPE_USM_POOL_DESC;
  poolDesc.pNext = &bufferPoolDesc;
  poolDesc.flags = 0;

  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_SUCCESS(urUSMPoolCreate(context, &poolDesc, &pool));
  ASSERT_NE(pool, nullptr);

  void *ptr1 = nullptr;
  ASSERT_SUCCESS(
      USMAllocFunc(context, hDevice, nullptr, pool, allocSize, &ptr1));
  ASSERT_NE(ptr1, nullptr);
  ValidateAlloc(ptr1, allocSize);
  ASSERT_TRUE(isWithinRange(ptr1, bufPtr, bufSize));

  void *ptr2 = nullptr;
  ASSERT_SUCCESS(
      USMAllocFunc(context, hDevice, nullptr, pool, allocSize, &ptr2));
  ASSERT_NE(ptr2, nullptr);
  ValidateAlloc(ptr2, allocSize);
  ASSERT_TRUE(isWithinRange(ptr2, bufPtr, bufSize));

  ASSERT_SUCCESS(urUSMFree(context, ptr2));
  ASSERT_SUCCESS(urUSMFree(context, ptr1));

  ASSERT_SUCCESS(urUSMPoolRelease(pool));
  ASSERT_SUCCESS(urUSMFree(context, bufPtr));
}

TEST_P(urL0BufferPoolTest, FailSize) {
  const auto USMAllocFunc = std::get<1>(this->GetParam()).USMAllocFunc;
  const auto checkUSMSupportFunc =
      std::get<1>(this->GetParam()).checkUSMSupportFunc;

  ur_device_usm_access_capability_flags_t USMSupport = 0;
  ASSERT_SUCCESS(checkUSMSupportFunc(device, USMSupport));
  if (!(USMSupport & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
    GTEST_SKIP() << "Selected USM type is not supported.";
  }

  ur_device_handle_t hDevice =
      (USMAllocFunc == urUSMHostAllocWrapped) ? nullptr : device;

  void *bufPtr = nullptr;
  size_t bufSize = 1 * 1024 * 1024; // 1MB
  size_t allocSize = bufSize + 1;

  ASSERT_SUCCESS(
      USMAllocFunc(context, hDevice, nullptr, nullptr, bufSize, &bufPtr));
  ASSERT_NE(bufPtr, nullptr);

  // Set buffer pool descriptor buffer to the USM allocation
  ur_usm_pool_buffer_desc_t bufferPoolDesc{};
  bufferPoolDesc.stype = UR_STRUCTURE_TYPE_USM_POOL_BUFFER_DESC;
  bufferPoolDesc.pNext = nullptr;
  bufferPoolDesc.pMem = bufPtr;
  bufferPoolDesc.size = bufSize;
  bufferPoolDesc.memType = AllocFuncsToUSMType[USMAllocFunc];
  bufferPoolDesc.device = hDevice;

  ur_usm_pool_desc_t poolDesc{};
  poolDesc.stype = UR_STRUCTURE_TYPE_USM_POOL_DESC;
  poolDesc.pNext = &bufferPoolDesc;
  poolDesc.flags = 0;

  ur_usm_pool_handle_t pool = nullptr;
  ASSERT_SUCCESS(urUSMPoolCreate(context, &poolDesc, &pool));
  ASSERT_NE(pool, nullptr);

  void *ptr = nullptr;
  ASSERT_EQ(USMAllocFunc(context, hDevice, nullptr, pool, allocSize, &ptr),
            UR_RESULT_ERROR_OUT_OF_HOST_MEMORY);

  ASSERT_SUCCESS(urUSMPoolRelease(pool));
  ASSERT_SUCCESS(urUSMFree(context, bufPtr));
}
