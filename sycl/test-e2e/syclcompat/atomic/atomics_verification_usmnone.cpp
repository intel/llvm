// ====------ atomics_noneusm_verification.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define SYCLCOMPAT_USM_LEVEL_NONE
#include <sycl/detail/core.hpp>
#include <cstdio>
#include <ctime>
#include <syclcompat/syclcompat.hpp>
#include <math.h>
#include <stdint.h>

#define min(a, b) (a) < (b) ? (a) : (b)
#define max(a, b) (a) > (b) ? (a) : (b)

#define LOOP_NUM 5

void atomicKernel(int *atom_arr, sycl::nd_item<3> item_ct1) {
  unsigned int tid = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                     item_ct1.get_local_id(2);

  for (int i = 0; i < LOOP_NUM; i++) {
    // Atomic addition
    syclcompat::atomic_fetch_add(&atom_arr[0], 10);

    // Atomic exchange
    syclcompat::atomic_exchange(&atom_arr[1], (int)tid);

    // Atomic maximum
    syclcompat::atomic_fetch_max(&atom_arr[2], (int)tid);

    // Atomic minimum
    syclcompat::atomic_fetch_min(&atom_arr[3], (int)tid);

    // Atomic increment (modulo 17+1)
    syclcompat::atomic_fetch_compare_inc((unsigned int *)&atom_arr[4],
                                   (unsigned int)17);

    // Atomic compare-and-swap
    syclcompat::atomic_compare_exchange_strong(&atom_arr[6], (int)(tid - 1),
                                         (int)tid);

    // Bitwise atomic instructions

    // Atomic AND
    syclcompat::atomic_fetch_and(&atom_arr[7], (int)(2 * tid + 7));

    // Atomic OR
    syclcompat::atomic_fetch_or(&atom_arr[8], 1 << tid);

    // Atomic XOR
    syclcompat::atomic_fetch_xor(&atom_arr[9], (int)tid);
  }
}

void atomicKernel_CPU(int *atom_arr, int no_of_threads) {

  for (int i = no_of_threads; i < 2 * no_of_threads; i++) {

    for (int j = 0; j < LOOP_NUM; j++) {
      // Atomic addition
      __sync_fetch_and_add(&atom_arr[0], 10);

      // Atomic exchange
      __sync_lock_test_and_set(&atom_arr[1], i);

      // Atomic maximum
      int old, expected;
      do {
        expected = atom_arr[2];
        old = __sync_val_compare_and_swap(&atom_arr[2], expected,
                                          max(expected, i));
      } while (old != expected);

      // Atomic minimum
      do {
        expected = atom_arr[3];
        old = __sync_val_compare_and_swap(&atom_arr[3], expected,
                                          min(expected, i));
      } while (old != expected);

      // Atomic increment (modulo 17+1)
      int limit = 17;
      do {
        expected = atom_arr[4];
        old = __sync_val_compare_and_swap(
            &atom_arr[4], expected, (expected >= limit) ? 0 : expected + 1);
      } while (old != expected);

      // Atomic decrement
      limit = 137;
      do {
        expected = atom_arr[5];
        old = __sync_val_compare_and_swap(
            &atom_arr[5], expected,
            ((expected == 0) || (expected > limit)) ? limit : expected - 1);
      } while (old != expected);

      // Atomic compare-and-swap
      __sync_val_compare_and_swap(&atom_arr[6], i - 1, i);

      // Bitwise atomic instructions

      // Atomic AND
      __sync_fetch_and_and(&atom_arr[7], 2 * i + 7);

      // Atomic OR
      __sync_fetch_and_or(&atom_arr[8], 1 << i);

      // Atomic XOR
      // 11th element should be 0xff
      __sync_fetch_and_xor(&atom_arr[9], i);
    }
  }
}

int verify(int *testData, const int len) {
  int val = 0;

  for (int i = 0; i < len * LOOP_NUM; ++i) {
    val += 10;
  }

  if (val != testData[0]) {
    printf("atomicAdd failed val = %d testData = %d\n", val, testData[0]);
    return false;
  }

  val = 0;

  bool found = false;

  for (int i = 0; i < len; ++i) {
    // second element should be a member of [0, len)
    if (i == testData[1]) {
      found = true;
      break;
    }
  }

  if (!found) {
    printf("atomicExch failed\n");
    return false;
  }

  val = -(1 << 8);

  for (int i = 0; i < len; ++i) {
    // third element should be len-1
    val = max(val, i);
  }

  if (val != testData[2]) {
    printf("atomicMax failed\n");
    return false;
  }

  val = 1 << 8;

  for (int i = 0; i < len; ++i) {
    val = min(val, i);
  }

  if (val != testData[3]) {
    printf("atomicMin failed\n");
    return false;
  }

  int limit = 17;
  val = 0;

  for (int i = 0; i < len * LOOP_NUM; ++i) {
    val = (val >= limit) ? 0 : val + 1;
  }

  if (val != testData[4]) {
    printf("atomicInc failed\n");
    return false;
  }

  limit = 137;
  val = 0;

  for (int i = 0; i < len * LOOP_NUM; ++i) {
    val = ((val == 0) || (val > limit)) ? limit : val - 1;
  }

  found = false;

  for (int i = 0; i < len; ++i) {
    // seventh element should be a member of [0, len)
    if (i == testData[6]) {
      found = true;
      break;
    }
  }

  if (!found) {
    printf("atomicCAS failed\n");
    return false;
  }

  val = 0xff;

  for (int i = 0; i < len; ++i) {
    // 8th element should be 1
    val &= (2 * i + 7);
  }

  if (val != testData[7]) {
    printf("atomicAnd failed\n");
    return false;
  }

  val = 0;

  for (int i = 0; i < len; ++i) {
    // 9th element should be 0xff
    val |= (1 << i);
  }

  if (val != testData[8]) {
    printf("atomicOr failed\n");
    return false;
  }

  val = 0xff;

  for (int i = 0; i < len; ++i) {
    // 11th element should be 0xff
    val ^= i;
  }

  if (val != testData[9]) {
    printf("atomicXor failed\n");
    return false;
  }

  return true;
}

int main(int argc, char **argv) {

  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 10;

  int *atom_arr;

  atom_arr = (int *)syclcompat::malloc(sizeof(int) * numData);

  for (unsigned int i = 0; i < numData; i++) {
    *syclcompat::get_host_ptr<unsigned int>(atom_arr + i) = 0;
  }

  // To make the AND and XOR tests generate something other than 0...
  *syclcompat::get_host_ptr<unsigned int>(atom_arr + 7) =
      *syclcompat::get_host_ptr<unsigned int>(atom_arr + 9) = 0xff;

  std::cout << "Selected device: "
            << syclcompat::get_default_queue()
                   .get_device()
                   .get_info<sycl::info::device::name>()
            << "\n";

  {
    std::pair<syclcompat::buffer_t, size_t> atom_arr_buf_ct0 =
        syclcompat::get_buffer_and_offset(atom_arr);
    size_t atom_arr_offset_ct0 = atom_arr_buf_ct0.second;
    syclcompat::get_default_queue().submit([&](sycl::handler &cgh) {
      auto atom_arr_acc_ct0 =
          atom_arr_buf_ct0.first.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                                             sycl::range<3>(1, 1, numThreads),
                                         sycl::range<3>(1, 1, numThreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         int *atom_arr_ct0 = (int *)(&atom_arr_acc_ct0[0] +
                                                     atom_arr_offset_ct0);
                         atomicKernel(atom_arr_ct0, item_ct1);
                       });
    });
  }

  atomicKernel_CPU(syclcompat::get_host_ptr<int>(atom_arr), numBlocks * numThreads);

  syclcompat::get_current_device().queues_wait_and_throw();

  // Compute & verify reference solution
  int testResult =
      verify(syclcompat::get_host_ptr<int>(atom_arr), 2 * numThreads * numBlocks);

  syclcompat::free(atom_arr);

  printf("Atomics test completed, returned %s \n",
         testResult ? "OK" : "ERROR!");
  exit(testResult ? 0 : -1);
}
