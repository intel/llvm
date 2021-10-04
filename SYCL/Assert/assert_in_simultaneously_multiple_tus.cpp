// FIXME unsupported on CUDA and HIP until fallback libdevice becomes available
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %S/Inputs %s %S/Inputs/kernels_in_file2.cpp -o %t.out %threads_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// CHECK:      {{this message from file1|this message from file2}}
// CHECK-NOT:  The test ended.

#include "Inputs/kernels_in_file2.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <thread>

#ifdef DEFINE_NDEBUG_INFILE1
#define NDEBUG
#else
#undef NDEBUG
#endif

#include <cassert>

using namespace cl::sycl;
using namespace cl::sycl::access;

static constexpr size_t NUM_THREADS = 4;
static constexpr size_t BUFFER_SIZE = 10;

template <class kernel_name> void enqueueKernel(queue *Q) {
  cl::sycl::range<1> numOfItems{BUFFER_SIZE};
  cl::sycl::buffer<int, 1> Buf(numOfItems);

  Q->submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<kernel_name>(numOfItems, [=](cl::sycl::id<1> wiID) {
      Acc[wiID] = 0;
      if (wiID == 5)
        assert(false && "this message from file1");
    });
  });
}

void runTestForTid(queue *Q, size_t Tid) {
  switch (Tid % 4) {
  case 0: {
    enqueueKernel<class kernel_name1>(Q);
    Q->wait();
    break;
  }
  case 1: {
    enqueueKernel<class kernel_name2>(Q);
    Q->wait();
    break;
  }
  case 2: {
    enqueueKernel_1_fromFile2(Q);
    Q->wait();
    break;
  }
  case 3: {
    enqueueKernel_2_fromFile2(Q);
    Q->wait();
    break;
  }
  }
}

int main(int Argc, const char *Argv[]) {
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
