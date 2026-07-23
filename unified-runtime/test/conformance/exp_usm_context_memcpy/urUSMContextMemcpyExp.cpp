// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/utils.h"
#include <uur/fixtures.h>

#include <atomic>
#include <thread>
#include <vector>

struct urUSMContextMemcpyExpTest : uur::urMultiQueueTypeTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(uur::urMultiQueueTypeTest::SetUp());

    bool context_memcpy_support = false;
    ASSERT_SUCCESS(
        uur::GetUSMContextMemcpyExpSupport(device, context_memcpy_support));
    if (!context_memcpy_support) {
      GTEST_SKIP() << "urUSMContextMemcpyExp is not supported";
    }
  }

  void TearDown() override {
    if (src_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, src_ptr));
    }
    if (dst_ptr) {
      EXPECT_SUCCESS(urUSMFree(context, dst_ptr));
    }

    UUR_RETURN_ON_FATAL_FAILURE(uur::urMultiQueueTypeTest::TearDown());
  }

  void initAllocations() {
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, src_ptr, sizeof(memset_src_value),
                                    &memset_src_value, allocation_size, 0,
                                    nullptr, nullptr));
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, dst_ptr, sizeof(memset_dst_value),
                                    &memset_dst_value, allocation_size, 0,
                                    nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  void verifyData() {
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, &host_mem, dst_ptr,
                                      allocation_size, 0, nullptr, nullptr));
    ASSERT_EQ(host_mem, memset_src_value);
  }

  static constexpr size_t memset_src_value = 42;
  static constexpr uint8_t memset_dst_value = 0;
  static constexpr uint32_t allocation_size = sizeof(memset_src_value);
  size_t host_mem = 0;

  void *src_ptr{nullptr};
  void *dst_ptr{nullptr};
};

struct urUSMContextMemcpyExpTestDevice : urUSMContextMemcpyExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urUSMContextMemcpyExpTest::SetUp());

    ur_device_usm_access_capability_flags_t device_usm = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_usm));
    if (!device_usm) {
      GTEST_SKIP() << "Device USM is not supported";
    }

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size,
                                    reinterpret_cast<void **>(&src_ptr)));
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size,
                                    reinterpret_cast<void **>(&dst_ptr)));
    initAllocations();
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_MULTI_QUEUE(urUSMContextMemcpyExpTestDevice);

TEST_P(urUSMContextMemcpyExpTestDevice, Success) {
  ASSERT_SUCCESS(
      urUSMContextMemcpyExp(context, dst_ptr, src_ptr, allocation_size));
  verifyData();
}

// Arbitrarily do the negative tests with device allocations. These are mostly a
// test of the loader and validation layer anyway so no big deal if they don't
// run on all devices due to lack of support.
TEST_P(urUSMContextMemcpyExpTestDevice, InvalidNullContext) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urUSMContextMemcpyExp(nullptr, dst_ptr, src_ptr, allocation_size));
}

TEST_P(urUSMContextMemcpyExpTestDevice, InvalidNullPtrs) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMContextMemcpyExp(context, nullptr, src_ptr, allocation_size));
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_POINTER,
      urUSMContextMemcpyExp(context, dst_ptr, nullptr, allocation_size));
}

TEST_P(urUSMContextMemcpyExpTestDevice, InvalidZeroSize) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                   urUSMContextMemcpyExp(context, dst_ptr, src_ptr, 0));
}

struct urUSMContextMemcpyExpTestHost : urUSMContextMemcpyExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urUSMContextMemcpyExpTest::SetUp());

    ur_device_usm_access_capability_flags_t host_usm = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, host_usm));
    if (!host_usm) {
      GTEST_SKIP() << "Host USM is not supported";
    }

    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                  reinterpret_cast<void **>(&src_ptr)));
    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                  reinterpret_cast<void **>(&dst_ptr)));
    initAllocations();
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUSMContextMemcpyExpTestHost);

TEST_P(urUSMContextMemcpyExpTestHost, Success) {
  ASSERT_SUCCESS(
      urUSMContextMemcpyExp(context, dst_ptr, src_ptr, allocation_size));
  verifyData();
}

struct urUSMContextMemcpyExpTestShared : urUSMContextMemcpyExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urUSMContextMemcpyExpTest::SetUp());

    ur_device_usm_access_capability_flags_t shared_usm_single = 0;

    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_single));

    if (!shared_usm_single) {
      GTEST_SKIP() << "Shared USM is not supported by the device.";
    }

    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size,
                                    reinterpret_cast<void **>(&src_ptr)));
    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size,
                                    reinterpret_cast<void **>(&dst_ptr)));
    initAllocations();
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urUSMContextMemcpyExpTestShared);

TEST_P(urUSMContextMemcpyExpTestShared, Success) {
  ASSERT_SUCCESS(
      urUSMContextMemcpyExp(context, dst_ptr, src_ptr, allocation_size));
  verifyData();
}

TEST_P(urUSMContextMemcpyExpTestDevice, LargeAllocation) {
  constexpr size_t large_size = 64 * 1024 * 1024;
  void *large_src = nullptr;
  void *large_dst = nullptr;

  ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr, large_size,
                                  &large_src));
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr, large_size,
                                  &large_dst));

  constexpr uint8_t pattern = 0xAB;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, large_src, sizeof(pattern), &pattern,
                                  large_size, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_SUCCESS(
      urUSMContextMemcpyExp(context, large_dst, large_src, large_size));

  uint8_t first = 0, last = 0;
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, &first, large_dst, 1, 0,
                                    nullptr, nullptr));
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(
      queue, true, &last, static_cast<char *>(large_dst) + large_size - 1, 1, 0,
      nullptr, nullptr));
  ASSERT_EQ(first, pattern);
  ASSERT_EQ(last, pattern);

  EXPECT_SUCCESS(urUSMFree(context, large_src));
  EXPECT_SUCCESS(urUSMFree(context, large_dst));
}

TEST_P(urUSMContextMemcpyExpTestDevice, ConcurrentCopies) {
  constexpr int num_threads = 4;
  constexpr size_t size_per_thread = 1024;

  struct ThreadData {
    void *src;
    void *dst;
    uint8_t pattern;
  };

  std::vector<ThreadData> thread_data(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    size_per_thread, &thread_data[i].src));
    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    size_per_thread, &thread_data[i].dst));

    thread_data[i].pattern = static_cast<uint8_t>(i + 1);
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, thread_data[i].src, 1,
                                    &thread_data[i].pattern, size_per_thread, 0,
                                    nullptr, nullptr));
  }
  ASSERT_SUCCESS(urQueueFinish(queue));

  std::vector<std::thread> threads;
  std::atomic<int> errors{0};

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      auto result = urUSMContextMemcpyExp(context, thread_data[i].dst,
                                          thread_data[i].src, size_per_thread);
      if (result != UR_RESULT_SUCCESS) {
        errors++;
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  ASSERT_EQ(errors.load(), 0) << "Some concurrent copies failed";

  for (int i = 0; i < num_threads; ++i) {
    std::vector<uint8_t> result(size_per_thread);
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, result.data(),
                                      thread_data[i].dst, size_per_thread, 0,
                                      nullptr, nullptr));

    for (auto byte : result) {
      ASSERT_EQ(byte, thread_data[i].pattern)
          << "Thread " << i << " data corrupted";
    }

    EXPECT_SUCCESS(urUSMFree(context, thread_data[i].src));
    EXPECT_SUCCESS(urUSMFree(context, thread_data[i].dst));
  }
}

TEST_P(urUSMContextMemcpyExpTestDevice, MultiThreadedSequential) {
  constexpr int num_threads = 4;
  constexpr size_t size_per_thread = 512;

  std::atomic<int> errors{0};
  std::vector<std::thread> threads;

  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back([&, thread_id]() {
      void *src = nullptr;
      void *dst = nullptr;

      auto cleanup = [&]() {
        if (src)
          urUSMFree(context, src);
        if (dst)
          urUSMFree(context, dst);
      };

      if (urUSMDeviceAlloc(context, device, nullptr, nullptr, size_per_thread,
                           &src) != UR_RESULT_SUCCESS) {
        errors++;
        return;
      }
      if (urUSMDeviceAlloc(context, device, nullptr, nullptr, size_per_thread,
                           &dst) != UR_RESULT_SUCCESS) {
        cleanup();
        errors++;
        return;
      }

      uint8_t pattern = static_cast<uint8_t>(thread_id + 1);
      if (urEnqueueUSMFill(queue, src, 1, &pattern, size_per_thread, 0, nullptr,
                           nullptr) != UR_RESULT_SUCCESS) {
        cleanup();
        errors++;
        return;
      }
      if (urQueueFinish(queue) != UR_RESULT_SUCCESS) {
        cleanup();
        errors++;
        return;
      }

      if (urUSMContextMemcpyExp(context, dst, src, size_per_thread) !=
          UR_RESULT_SUCCESS) {
        cleanup();
        errors++;
        return;
      }

      std::vector<uint8_t> verify(size_per_thread);
      if (urEnqueueUSMMemcpy(queue, true, verify.data(), dst, size_per_thread,
                             0, nullptr, nullptr) != UR_RESULT_SUCCESS) {
        cleanup();
        errors++;
        return;
      }

      for (auto byte : verify) {
        if (byte != pattern) {
          errors++;
          break;
        }
      }

      cleanup();
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  ASSERT_EQ(errors.load(), 0) << "Multi-threaded sequential test failed";
}

TEST_P(urUSMContextMemcpyExpTestDevice, UnalignedPointers) {
  constexpr size_t base_size = 1024;
  constexpr size_t offset = 7;
  void *src_base = nullptr;
  void *dst_base = nullptr;

  ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr, base_size,
                                  &src_base));
  ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr, base_size,
                                  &dst_base));

  void *src_unaligned = static_cast<char *>(src_base) + offset;
  void *dst_unaligned = static_cast<char *>(dst_base) + offset;
  size_t copy_size = base_size - offset;

  constexpr uint8_t pattern = 0xCD;
  ASSERT_SUCCESS(urEnqueueUSMFill(queue, src_base, sizeof(pattern), &pattern,
                                  base_size, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));

  ASSERT_SUCCESS(
      urUSMContextMemcpyExp(context, dst_unaligned, src_unaligned, copy_size));

  std::vector<uint8_t> verify(copy_size);
  ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, verify.data(), dst_unaligned,
                                    copy_size, 0, nullptr, nullptr));

  for (auto byte : verify) {
    ASSERT_EQ(byte, pattern);
  }

  EXPECT_SUCCESS(urUSMFree(context, src_base));
  EXPECT_SUCCESS(urUSMFree(context, dst_base));
}
