// REQUIRES: gpu, level_zero

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out

// Set batching to 4 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=4 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB4 %s

// Set batching to 1 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB1 %s

// Set batching to 3 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=3 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB3 %s

// Set batching to 5 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=5 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB5 %s

// Set batching to 7 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=7 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB7 %s

// Set batching to 8 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=8 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB8 %s

// Set batching to 9 explicitly
// RUN: env SYCL_PI_LEVEL_ZERO_BATCH_SIZE=9 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 | FileCheck --check-prefixes=CKALL,CKB9 %s

// level_zero_batch_test_copy_with_compute.cpp
//
// This tests the level zero plugin's kernel batching code.  The default
// batching is 4, and exact batch size can be controlled with environment
// variable SYCL_PI_LEVEL_ZERO_{COPY_}BATCH_SIZE=N.
// This test enqueues 8 kernels and then does a wait. And it does this 3 times.
// Expected output is that for batching =1 you will see zeCommandListClose,
// and zeCommandQueueExecuteCommandLists after every piEnqueueKernelLaunch.
// For batching=3 you will see that after 3rd and 6th enqueues, and then after
// piEventsWait. For 5, after 5th piEnqueue, and then after piEventsWait.  For
// 4 you will see these after 4th and 8th Enqueue, and for 8, only after the
// 8th enqueue.  And lastly for 9, you will see the Close and Execute calls
// only after the piEventsWait.
// Since the test does this 3 times, this pattern will repeat 2 more times,
// and then the test will print Test Passed 8 times, once for each kernel
// validation check.
// Pattern starts first set of kernel executions.
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB4:  ZE ---> zeCommandListClose(
// CKB4:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB5:  ZE ---> zeCommandListClose(
// CKB5:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB7:  ZE ---> zeCommandListClose(
// CKB7:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB4:  ZE ---> zeCommandListClose(
// CKB4:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB8:  ZE ---> zeCommandListClose(
// CKB8:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piQueueFinish(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB5:  ZE ---> zeCommandListClose(
// CKB5:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB7:  ZE ---> zeCommandListClose(
// CKB7:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB9:  ZE ---> zeCommandListClose(
// CKB9:  ZE ---> zeCommandQueueExecuteCommandLists(
// Pattern starts 2nd set of kernel executions
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB4:  ZE ---> zeCommandListClose(
// CKB4:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB5:  ZE ---> zeCommandListClose(
// CKB5:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB7:  ZE ---> zeCommandListClose(
// CKB7:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB4:  ZE ---> zeCommandListClose(
// CKB4:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB8:  ZE ---> zeCommandListClose(
// CKB8:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piQueueFinish(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB5:  ZE ---> zeCommandListClose(
// CKB5:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB7:  ZE ---> zeCommandListClose(
// CKB7:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB9:  ZE ---> zeCommandListClose(
// CKB9:  ZE ---> zeCommandQueueExecuteCommandLists(
// Pattern starts 3rd set of kernel executions
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB4:  ZE ---> zeCommandListClose(
// CKB4:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB5:  ZE ---> zeCommandListClose(
// CKB5:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB7:  ZE ---> zeCommandListClose(
// CKB7:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piEnqueueKernelLaunch(
// CKALL: ZE ---> zeCommandListAppendLaunchKernel(
// CKB1:  ZE ---> zeCommandListClose(
// CKB1:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB4:  ZE ---> zeCommandListClose(
// CKB4:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB8:  ZE ---> zeCommandListClose(
// CKB8:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKALL: ---> piQueueFinish(
// CKB3:  ZE ---> zeCommandListClose(
// CKB3:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB5:  ZE ---> zeCommandListClose(
// CKB5:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB7:  ZE ---> zeCommandListClose(
// CKB7:  ZE ---> zeCommandQueueExecuteCommandLists(
// CKB9:  ZE ---> zeCommandListClose(
// CKB9:  ZE ---> zeCommandQueueExecuteCommandLists(
// Now just check for 16 Test Pass kernel validations.
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass

#include "CL/sycl.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

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
  size_t M = 16;
  size_t N = 4;
  size_t AL = M * N * sizeof(uint32_t);

  sycl::queue q(sycl::default_selector_v);
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

  uint32_t *X1 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *X2 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *X3 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *X4 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *X5 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *X6 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *X7 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));
  uint32_t *X8 = static_cast<uint32_t *>(sycl::malloc_shared(AL, dev, ctx));

  for (size_t i = 0; i < M * N; i++) {
    Y1[i] = i % 255;
    X1[i] = X2[i] = X3[i] = X4[i] = X5[i] = X6[i] = X7[i] = X8[i] = 0;
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
      q.memcpy(X1, Y1, sizeof(uint32_t) * M * N);

      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy2>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z2[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.memcpy(X2, Y1, sizeof(uint32_t) * M * N);

      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy3>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z3[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.memcpy(X3, Y1, sizeof(uint32_t) * M * N);

      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy4>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z4[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.memcpy(X4, Y1, sizeof(uint32_t) * M * N);

      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy5>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z5[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.memcpy(X5, Y1, sizeof(uint32_t) * M * N);

      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy6>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z6[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.memcpy(X6, Y1, sizeof(uint32_t) * M * N);

      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy7>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z7[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.memcpy(X7, Y1, sizeof(uint32_t) * M * N);

      q.submit([&](sycl::handler &h) {
        h.parallel_for<class u32_copy8>(sycl::range<2>{M, N},
                                        [=](sycl::id<2> it) {
                                          const int m = it[0];
                                          const int n = it[1];
                                          Z8[m * N + n] = Y1[m * N + n];
                                        });
      });
      q.memcpy(X8, Y1, sizeof(uint32_t) * M * N);

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

  validate(Y1, X1, M * N);
  validate(Y1, X2, M * N);
  validate(Y1, X3, M * N);
  validate(Y1, X4, M * N);
  validate(Y1, X5, M * N);
  validate(Y1, X6, M * N);
  validate(Y1, X7, M * N);
  validate(Y1, X8, M * N);

  sycl::free(Y1, ctx);
  sycl::free(Z1, ctx);
  sycl::free(Z2, ctx);
  sycl::free(Z3, ctx);
  sycl::free(Z4, ctx);
  sycl::free(Z5, ctx);
  sycl::free(Z6, ctx);
  sycl::free(Z7, ctx);
  sycl::free(Z8, ctx);

  sycl::free(X1, ctx);
  sycl::free(X2, ctx);
  sycl::free(X3, ctx);
  sycl::free(X4, ctx);
  sycl::free(X5, ctx);
  sycl::free(X6, ctx);
  sycl::free(X7, ctx);
  sycl::free(X8, ctx);

  return 0;
}
