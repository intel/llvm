// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/utils.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

// Closes a DMA-BUF file descriptor when it goes out of scope.
struct FdGuard {
  explicit FdGuard(int fd) : fd(fd) {}
  FdGuard(const FdGuard &) = delete;
  FdGuard &operator=(const FdGuard &) = delete;
  ~FdGuard() {
#ifndef _WIN32
    if (fd >= 0) {
      close(fd);
    }
#endif
  }
  int fd;
};

size_t hostPageSize() {
#ifndef _WIN32
  return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#else
  return 4096;
#endif
}

bool supportsDmaBuf(ur_device_handle_t device) {
  ur_exp_address_range_export_type_flags_t types = 0;
  if (urDeviceGetInfo(device, UR_DEVICE_INFO_ADDRESS_RANGE_EXPORT_TYPES_EXP,
                      sizeof(types), &types, nullptr) != UR_RESULT_SUCCESS) {
    return false;
  }
  return types & UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_FLAG_DMA_BUF_FD;
}

ur_result_t getHandle(ur_context_handle_t context, const void *pMem,
                      size_t size, ur_exp_address_range_export_t *pExport) {
  return urUSMGetHandleForAddressRangeExp(
      context, pMem, size, UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD, nullptr,
      pExport);
}

// Checks the invariants that hold for any successful export of a range
// starting at pMem.
void checkLocation(const void *pMem, const ur_exp_address_range_export_t &exp) {
  ASSERT_EQ(exp.type, UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD);
  ASSERT_GE(exp.handle.fd, 0);
  const auto mem = reinterpret_cast<uintptr_t>(pMem);
  const auto base = reinterpret_cast<uintptr_t>(exp.base);
  ASSERT_LE(base, mem);
  ASSERT_EQ(base + exp.offset, mem);
  ASSERT_GT(exp.size, exp.offset);
  ASSERT_EQ(base % hostPageSize(), 0u);
}

// Additionally checks that [pMem, pMem + size), which lies within one USM
// allocation, is covered in full.
void checkFullCoverage(const void *pMem, size_t size,
                       const ur_exp_address_range_export_t &exp) {
  UUR_RETURN_ON_FATAL_FAILURE(checkLocation(pMem, exp));
  ASSERT_GE(exp.size - exp.offset, size);
}

} // namespace

struct urUSMGetHandleForAddressRangeExpTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

    if (!supportsDmaBuf(device)) {
      GTEST_SKIP() << "DMA-BUF address range handles are not supported.";
    }

    ur_device_usm_access_capability_flags_t deviceUSM = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceUSM));
    if (!deviceUSM) {
      GTEST_SKIP() << "Device USM is not supported.";
    }

    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, device, nullptr, nullptr, allocSize, &ptr));
    ASSERT_NE(ptr, nullptr);
  }

  void TearDown() override {
    if (ptr) {
      EXPECT_SUCCESS(urUSMFree(context, ptr));
    }
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
  }

  ur_result_t getHandle(const void *pMem, size_t size,
                        ur_exp_address_range_export_t *pExport) {
    return ::getHandle(context, pMem, size, pExport);
  }

  static constexpr size_t allocSize = 64 * 1024;
  void *ptr = nullptr;
  ur_exp_address_range_export_t exported{};
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUSMGetHandleForAddressRangeExpTest);

TEST_P(urUSMGetHandleForAddressRangeExpTest, SuccessWholeAllocation) {
  ASSERT_SUCCESS(getHandle(ptr, allocSize, &exported));
  FdGuard guard(exported.handle.fd);
  checkFullCoverage(ptr, allocSize, exported);
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, SuccessUnalignedInteriorRange) {
  const void *interior = static_cast<const uint8_t *>(ptr) + 100;
  constexpr size_t size = 1000;
  ASSERT_SUCCESS(getHandle(interior, size, &exported));
  FdGuard guard(exported.handle.fd);
  checkFullCoverage(interior, size, exported);
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, SuccessTwoAllocations) {
  // Small allocations are likely to be sub-allocated from the same object,
  // so the second one is expected at a non-zero offset of that object.
  constexpr size_t smallSize = 4096 + 123;
  void *other = nullptr;
  ASSERT_SUCCESS(
      urUSMDeviceAlloc(context, device, nullptr, nullptr, smallSize, &other));
  ASSERT_NE(other, nullptr);

  ur_exp_address_range_export_t first{};
  ur_exp_address_range_export_t second{};
  ur_result_t firstResult = getHandle(ptr, smallSize, &first);
  FdGuard firstGuard(firstResult == UR_RESULT_SUCCESS ? first.handle.fd : -1);
  ur_result_t secondResult = getHandle(other, smallSize, &second);
  FdGuard secondGuard(secondResult == UR_RESULT_SUCCESS ? second.handle.fd
                                                        : -1);
  EXPECT_SUCCESS(urUSMFree(context, other));

  ASSERT_SUCCESS(firstResult);
  ASSERT_SUCCESS(secondResult);
  UUR_RETURN_ON_FATAL_FAILURE(checkFullCoverage(ptr, smallSize, first));
  checkFullCoverage(other, smallSize, second);
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, SuccessWithProperties) {
  ur_exp_address_range_handle_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ADDRESS_RANGE_HANDLE_PROPERTIES, nullptr, 0};
  ASSERT_SUCCESS(urUSMGetHandleForAddressRangeExp(
      context, ptr, allocSize, UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD,
      &props, &exported));
  FdGuard guard(exported.handle.fd);
  checkFullCoverage(ptr, allocSize, exported);
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, SuccessHandleOutlivesAllocation) {
  ASSERT_SUCCESS(getHandle(ptr, allocSize, &exported));
  FdGuard guard(exported.handle.fd);
  ASSERT_GE(exported.handle.fd, 0);
  void *freed = ptr;
  ptr = nullptr;
  ASSERT_SUCCESS(urUSMFree(context, freed));
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, SuccessReadBack) {
#ifdef _WIN32
  GTEST_SKIP() << "DMA-BUF file descriptors are only available on Linux.";
#else
  constexpr uint8_t pattern = 0x5A;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(pattern), &pattern,
                                  allocSize, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_SUCCESS(getHandle(ptr, allocSize, &exported));
  FdGuard guard(exported.handle.fd);
  UUR_RETURN_ON_FATAL_FAILURE(checkFullCoverage(ptr, allocSize, exported));

  void *map = mmap(nullptr, exported.size, PROT_READ, MAP_SHARED,
                   exported.handle.fd, 0);
  if (map == MAP_FAILED) {
    GTEST_SKIP() << "The DMA-BUF cannot be mapped on the host: "
                 << std::strerror(errno);
  }
  const auto *bytes = static_cast<const uint8_t *>(map) + exported.offset;
  size_t mismatches = 0;
  for (size_t i = 0; i < allocSize; ++i) {
    mismatches += bytes[i] != pattern;
  }
  munmap(map, exported.size);
  ASSERT_EQ(mismatches, 0u);
#endif
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidNullHandleContext) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urUSMGetHandleForAddressRangeExp(
                       nullptr, ptr, allocSize,
                       UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD, nullptr,
                       &exported));
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidNullPointerMem) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   getHandle(nullptr, allocSize, &exported));
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidNullPointerExport) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   getHandle(ptr, allocSize, nullptr));
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidEnumerationExportType) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urUSMGetHandleForAddressRangeExp(
                       context, ptr, allocSize,
                       UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_FORCE_UINT32, nullptr,
                       &exported));
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidEnumerationFlags) {
  ur_exp_address_range_handle_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ADDRESS_RANGE_HANDLE_PROPERTIES, nullptr,
      UR_EXP_ADDRESS_RANGE_HANDLE_FLAG_FORCE_UINT32};
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ENUMERATION,
                   urUSMGetHandleForAddressRangeExp(
                       context, ptr, allocSize,
                       UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD, &props,
                       &exported));
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, UnsupportedFeatureReservedFlag) {
  ur_exp_address_range_handle_properties_t props{
      UR_STRUCTURE_TYPE_EXP_ADDRESS_RANGE_HANDLE_PROPERTIES, nullptr,
      UR_EXP_ADDRESS_RANGE_HANDLE_FLAG_TBD};
  ur_result_t result = urUSMGetHandleForAddressRangeExp(
      context, ptr, allocSize, UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD,
      &props, &exported);
  FdGuard guard(result == UR_RESULT_SUCCESS ? exported.handle.fd : -1);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_UNSUPPORTED_FEATURE, result);
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidSize) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE, getHandle(ptr, 0, &exported));
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidArgumentHostPointer) {
  uint8_t hostBuffer[256] = {};
  ur_result_t result = getHandle(hostBuffer, sizeof(hostBuffer), &exported);
  FdGuard guard(result == UR_RESULT_SUCCESS ? exported.handle.fd : -1);
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ARGUMENT, result);
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidArgumentHostMemory) {
  ur_device_usm_access_capability_flags_t hostUSM = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, hostUSM));
  if (!hostUSM) {
    GTEST_SKIP() << "Host USM is not supported.";
  }

  void *hostPtr = nullptr;
  ASSERT_SUCCESS(
      urUSMHostAlloc(context, nullptr, nullptr, allocSize, &hostPtr));
  ur_result_t result = getHandle(hostPtr, allocSize, &exported);
  FdGuard guard(result == UR_RESULT_SUCCESS ? exported.handle.fd : -1);
  EXPECT_SUCCESS(urUSMFree(context, hostPtr));
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ARGUMENT, result);
}

TEST_P(urUSMGetHandleForAddressRangeExpTest, InvalidArgumentSharedMemory) {
  ur_device_usm_access_capability_flags_t sharedUSM = 0;
  ASSERT_SUCCESS(uur::GetDeviceUSMSingleSharedSupport(device, sharedUSM));
  if (!sharedUSM) {
    GTEST_SKIP() << "Shared USM is not supported.";
  }

  void *sharedPtr = nullptr;
  ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr, allocSize,
                                  &sharedPtr));
  ur_result_t result = getHandle(sharedPtr, allocSize, &exported);
  FdGuard guard(result == UR_RESULT_SUCCESS ? exported.handle.fd : -1);
  EXPECT_SUCCESS(urUSMFree(context, sharedPtr));
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_ARGUMENT, result);
}

struct urUSMGetHandleForAddressRangeExpVirtualMemTest
    : uur::urVirtualMemMappedTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urVirtualMemMappedTest::SetUp());
    if (!supportsDmaBuf(device)) {
      GTEST_SKIP() << "DMA-BUF address range handles are not supported.";
    }
  }
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE(
    urUSMGetHandleForAddressRangeExpVirtualMemTest);

TEST_P(urUSMGetHandleForAddressRangeExpVirtualMemTest, SuccessMappedRange) {
  // A mapped range may be backed by several objects, so a handle may cover
  // only a prefix of it. Walk the range until it is covered.
  size_t done = 0;
  while (done < size) {
    const void *p = static_cast<const uint8_t *>(virtual_ptr) + done;
    ur_exp_address_range_export_t exported{};
    ASSERT_SUCCESS(getHandle(context, p, size - done, &exported));
    FdGuard guard(exported.handle.fd);
    UUR_RETURN_ON_FATAL_FAILURE(checkLocation(p, exported));
    done += std::min(size - done,
                     static_cast<size_t>(exported.size - exported.offset));
  }
  ASSERT_EQ(done, size);
}
