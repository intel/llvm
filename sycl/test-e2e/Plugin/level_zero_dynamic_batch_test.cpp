// REQUIRES: gpu, level_zero
// UNSUPPORTED: ze_debug

// RUN: %{build} -o %t.ooo.out
// RUN: %{build} -DUSING_INORDER -o %t.ino.out
// RUN: %{build} -DUSING_DISCARD_EVENTS -o %t.discard_events.out

// Check that dynamic batching raises/lowers batch size
// RUN: env SYCL_PI_TRACE=2 UR_L0_DEBUG=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{run} %t.ooo.out 2>&1 | FileCheck --check-prefixes=CKALL,CKDYN %s
// RUN: env SYCL_PI_TRACE=2 UR_L0_DEBUG=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{run} %t.ino.out 2>&1 | FileCheck --check-prefixes=CKALL,CKDYN %s
// RUN: env SYCL_PI_TRACE=2 UR_L0_DEBUG=1 SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=2 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{run} %t.discard_events.out 2>&1 | FileCheck --check-prefixes=CKALL,CKDYN %s

// level_zero_dynamic_batch_test.cpp
//
// This tests the level zero plugin's kernel dynamic batch size adjustment
// code.
// It starts out by enqueing 40 kernels before it does a wait, and it does
// this 5 times.  That should cause the dynamic batch size adjustment to
// raise the batch size up 3 times.
//
// Then the test starts enqueueing only 4 kernels before doing a wait, and
// it does that 25 times.  That should cause the batch size to
// be lowered to be less than 4.
//
// CKDYN: Raising QueueBatchSize to 5
// CKDYN: Raising QueueBatchSize to 6
// CKDYN: Raising QueueBatchSize to 7
// CKDYN-NOT: Raising QueueBatchSize
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKALL: Test Pass
// CKDYN: Lowering QueueBatchSize to 3
// CKDYN-NOT: Lowering QueueBatchSize
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
#elif USING_DISCARD_EVENTS
  sycl::property_list Props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
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

  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++) {
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
    }
    q.wait();
  }

  validate(Y1, Z1, M * N);
  validate(Y1, Z2, M * N);
  validate(Y1, Z3, M * N);
  validate(Y1, Z4, M * N);
  validate(Y1, Z5, M * N);
  validate(Y1, Z6, M * N);
  validate(Y1, Z7, M * N);
  validate(Y1, Z8, M * N);

  for (size_t i = 0; i < 25; i++) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class u32_copy9>(sycl::range<2>{M, N},
                                      [=](sycl::id<2> it) {
                                        const int m = it[0];
                                        const int n = it[1];
                                        Z1[m * N + n] = Y1[m * N + n];
                                      });
    });
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class u32_copy10>(sycl::range<2>{M, N},
                                       [=](sycl::id<2> it) {
                                         const int m = it[0];
                                         const int n = it[1];
                                         Z2[m * N + n] = Y1[m * N + n];
                                       });
    });
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class u32_copy11>(sycl::range<2>{M, N},
                                       [=](sycl::id<2> it) {
                                         const int m = it[0];
                                         const int n = it[1];
                                         Z3[m * N + n] = Y1[m * N + n];
                                       });
    });
    q.submit([&](sycl::handler &h) {
      h.parallel_for<class u32_copy12>(sycl::range<2>{M, N},
                                       [=](sycl::id<2> it) {
                                         const int m = it[0];
                                         const int n = it[1];
                                         Z4[m * N + n] = Y1[m * N + n];
                                       });
    });
    q.wait();
  }
  validate(Y1, Z1, M * N);
  validate(Y1, Z2, M * N);
  validate(Y1, Z3, M * N);
  validate(Y1, Z4, M * N);

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
