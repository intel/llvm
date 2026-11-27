//===------------------assert_in_simultaneous_kernels.hpp------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests device-side assertion handling in distinct kernels submitted
/// concurrently from multiple host threads, each using its own queue.
///
//===----------------------------------------------------------------------===//
#include <cassert>
#include <cstdio>
#include <iostream>
#include <latch>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <thread>

using namespace sycl;
using namespace sycl::access;

static constexpr size_t NUM_THREADS = 4;
static constexpr size_t RANGE_SIZE = 16;

void assertFunc() { assert(false && "from assert statement"); }

template <class kernel_name> void assertTest(queue *Q, std::latch *Sync) {
  Sync->arrive_and_wait();

  Q->parallel_for<kernel_name>(RANGE_SIZE, [=](sycl::id<1> wiID) {
     if (wiID == RANGE_SIZE - 1)
       assertFunc();
   }).wait();
}

void runTestForTid(queue *Q, size_t Tid) {
  // Passed to kernel functions to have better chance
  // of running them simultaneously.
  static std::latch Sync(NUM_THREADS);

  // Select a distinct kernel name for each host thread to test
  // concurrent assertion reporting from multiple unique kernels.
  switch (Tid % 4) {
  case 0: {
    assertTest<class the_kernel1>(Q, &Sync);
    break;
  }
  case 1: {
    assertTest<class the_kernel2>(Q, &Sync);
    break;
  }
  case 2: {
    assertTest<class the_kernel3>(Q, &Sync);
    break;
  }
  case 3: {
    assertTest<class the_kernel4>(Q, &Sync);
    break;
  }
  }
}

int main() {
  // On windows stderr output becomes messed up if several thread
  // output simultaneously. Hence, setting explicit line buffering here.
#ifndef __SYCL_DEVICE_ONLY__
  if (setvbuf(stderr, nullptr, _IOLBF, BUFSIZ)) {
    std::cerr << "Can't set line-buffering mode fo stderr\n";
    return 1;
  }
#endif

  std::vector<std::thread> threadPool;
  threadPool.reserve(NUM_THREADS);

  std::vector<std::unique_ptr<queue>> Queues;
  for (size_t i = 0; i < NUM_THREADS; ++i) {
    Queues.push_back(std::make_unique<queue>());
  }

  for (size_t tid = 0; tid < NUM_THREADS; ++tid) {
    threadPool.push_back(std::thread(runTestForTid, Queues[tid].get(), tid));
  }

  for (auto &currentThread : threadPool) {
    currentThread.join();
  }

  std::cout << "The test ended." << std::endl;
  return 0;
}
