// REQUIRES: gpu, level_zero

// RUN: %{build} -o %t.ooo.out
// RUN: %{build} -DUSING_INORDER -o %t.ino.out
// UNSUPPORTED: ze_debug, level_zero_v2_adapter

// To test batching on out-of-order queue:
// Set batching to 4 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=4 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ooo.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB4 %s

// Set batching to 1 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ooo.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB1 %s

// Set batching to 3 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=3 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ooo.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB3 %s

// Set batching to 5 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=5 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ooo.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB5 %s

// Set batching to 7 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=7 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ooo.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB7 %s

// Set batching to 8 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=8 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ooo.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB8 %s

// Set batching to 9 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=9 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ooo.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB9 %s

// To test batching on in-order queue:
// Set batching to 4 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=4 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ino.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB4 %s

// Set batching to 1 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ino.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB1 %s

// Set batching to 3 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=3 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ino.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB3 %s

// Set batching to 5 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=5 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ino.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB5 %s

// Set batching to 7 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=7 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ino.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB7 %s

// Set batching to 8 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=8 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ino.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB8 %s

// Set batching to 9 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=9 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_UR_TRACE=2 UR_L0_DEBUG=1 %{run} %t.ino.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB9 %s

// level_zero_batch_test.cpp
//
// This tests the level zero adapter's kernel batching code.  The default
// batching is 4, and exact batch size can be controlled with environment
// variable SYCL_PI_LEVEL_ZEOR+BATCH_SIZE=N.
// This test enqueues 8 kernels and then does a wait. And it does this 3 times.
// Expected output is that for batching =1 you will see zeCommandListClose,
// and zeCommandQueueExecuteCommandLists after every
// urEnqueueKernelLaunchWithArgsExp. For batching=3 you will see that after 3rd
// and 6th enqueues, and then after urQueueFinish. For 5, after 5th urEnqueue,
// and then after urQueueFinish.  For 4 you will see these after 4th and 8th
// Enqueue, and for 8, only after the 8th enqueue.  And lastly for 9, you will
// see the Close and Execute calls only after the urQueueFinish. Since the test
// does this 3 times, this pattern will repeat 2 more times, and then the test
// will print Test Passed 8 times, once for each kernel validation check.
// Pattern starts first set of kernel executions.
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB4:  zeCommandListClose(
// CKB4:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB5:  zeCommandListClose(
// CKB5:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB7:  zeCommandListClose(
// CKB7:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB4:  zeCommandListClose(
// CKB4:  zeCommandQueueExecuteCommandLists(
// CKB8:  zeCommandListClose(
// CKB8:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urQueueFinish
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKB5:  zeCommandListClose(
// CKB5:  zeCommandQueueExecuteCommandLists(
// CKB7:  zeCommandListClose(
// CKB7:  zeCommandQueueExecuteCommandLists(
// CKB9:  zeCommandListClose(
// CKB9:  zeCommandQueueExecuteCommandLists(
// Pattern starts 2nd set of kernel executions
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB4:  zeCommandListClose(
// CKB4:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB5:  zeCommandListClose(
// CKB5:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB7:  zeCommandListClose(
// CKB7:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB4:  zeCommandListClose(
// CKB4:  zeCommandQueueExecuteCommandLists(
// CKB8:  zeCommandListClose(
// CKB8:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urQueueFinish
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKB5:  zeCommandListClose(
// CKB5:  zeCommandQueueExecuteCommandLists(
// CKB7:  zeCommandListClose(
// CKB7:  zeCommandQueueExecuteCommandLists(
// CKB9:  zeCommandListClose(
// CKB9:  zeCommandQueueExecuteCommandLists(
// Pattern starts 3rd set of kernel executions
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB4:  zeCommandListClose(
// CKB4:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB5:  zeCommandListClose(
// CKB5:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB7:  zeCommandListClose(
// CKB7:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urEnqueueKernelLaunchWithArgsExp
// CKALL: zeCommandListAppendLaunchKernel(
// CKB1:  zeCommandListClose(
// CKB1:  zeCommandQueueExecuteCommandLists(
// CKB4:  zeCommandListClose(
// CKB4:  zeCommandQueueExecuteCommandLists(
// CKB8:  zeCommandListClose(
// CKB8:  zeCommandQueueExecuteCommandLists(
// CKALL: ---> urQueueFinish
// CKB3:  zeCommandListClose(
// CKB3:  zeCommandQueueExecuteCommandLists(
// CKB5:  zeCommandListClose(
// CKB5:  zeCommandQueueExecuteCommandLists(
// CKB7:  zeCommandListClose(
// CKB7:  zeCommandQueueExecuteCommandLists(
// CKB9:  zeCommandListClose(
// CKB9:  zeCommandQueueExecuteCommandLists(
// Now just check for 8 Test Pass kernel validations.
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass

#include <chrono>
#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

void validate(uint32_t *result, uint32_t *expect, size_t n) {
  int error = 0;
  for (int i = 0; i < n; i++) {
    if (result[i] != expect[i]) {
      error++;
      if (error < 10) {
        printf("Error: %d, expect: %d\n", result[i], expect[i]);
      }
    }
  }
  error > 0 ? printf("Error: %d\n", error) : printf("Test Pass\n");
}

int main(int argc, char *argv[]) {
  size_t M = 65536;
  size_t N = 512 / 4;
  size_t AL = M * N * sizeof(uint32_t);

#ifdef USING_INORDER
  sycl::property_list Props{sycl::property::queue::in_order{}};
#else
  sycl::property_list Props{};
#endif
  sycl::queue q(sycl::default_selector_v, Props);
  auto ctx = q.get_context();
  auto dev = q.get_device();

  uint32_t *Y1 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *Z1 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *Z2 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *Z3 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *Z4 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *Z5 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *Z6 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *Z7 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *Z8 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));

  for (size_t i = 0; i < M * N; i++) {
    Y1[i] = i % 255;
  }

  memset(Z1, '\0', AL);
  memset(Z2, '\0', AL);
  memset(Z3, '\0', AL);
  memset(Z4, '\0', AL);
  memset(Z5, '\0', AL);
  memset(Z6, '\0', AL);
  memset(Z7, '\0', AL);
  memset(Z8, '\0', AL);

  {
    for (size_t j = 0; j < 3; j++) {
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy1>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z1[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy2>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z2[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy3>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z3[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy4>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z4[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy5>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z5[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy6>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z6[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy7>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z7[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy8>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z8[m * N + n] = Y1[m * N + n];
                                        });
      });

      q.wait();
    }
  }
  validate(Y1, Z1, M * N);
  validate(Y1, Z2, M * N);
  validate(Y1, Z3, M * N);
  validate(Y1, Z4, M * N);
  validate(Y1, Z5, M * N);
  validate(Y1, Z6, M * N);
  validate(Y1, Z7, M * N);
  validate(Y1, Z8, M * N);

  sycl::free(Y1, ctx);
  sycl::free(Z1, ctx);
  sycl::free(Z2, ctx);
  sycl::free(Z3, ctx);
  sycl::free(Z4, ctx);
  sycl::free(Z5, ctx);
  sycl::free(Z6, ctx);
  sycl::free(Z7, ctx);
  sycl::free(Z8, ctx);

  return 0;
}
