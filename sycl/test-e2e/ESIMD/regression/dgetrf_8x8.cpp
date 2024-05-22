//==-------------- dgetrf.cpp  - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: aspect-fp64
// RUN: %{build} -I%S/.. -o %t.out
// RUN: %{run} %t.out 1
//
// Reduced version of dgetrf.cpp - M = 8, N = 8, single batch.
//
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "../esimd_test_utils.hpp"

#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define MIN(x, y) ((x) <= (y) ? (x) : (y))
#define MAX(x, y) ((x) >= (y) ? (x) : (y))
#define FP_RAND ((double)rand() / (double)RAND_MAX)

#define OUTN(text, ...) fprintf(stderr, text, ##__VA_ARGS__)
#define OUT(text, ...) OUTN(text "\n", ##__VA_ARGS__)

#define CHECK(cmd, status)                                                     \
  do {                                                                         \
    cmd;                                                                       \
    if (status) {                                                              \
      OUT(#cmd " status: %d", status);                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define FAILED(res, thresh) ((res) > (thresh) || (res) != (res))
#define CHECK_AND_REPORT(test_desc, test_id, fail_cond, res, fail_cnt)         \
  do {                                                                         \
    if (fail_cond)                                                             \
      fail_cnt++;                                                              \
    OUT("Test (%s): " test_desc ". Result: %f. %s", test_id, res,              \
        (fail_cond) ? "FAILED" : "PASSED");                                    \
  } while (0)

using namespace sycl;
using namespace std;
using namespace sycl::ext::intel::esimd;

ESIMD_PRIVATE ESIMD_REGISTER(384) simd<double, 3 * 32 * 4> GRF;

#define V(x, w, i) (x).template select<w, 1>(i)
#define V1(x, i) V(x, 1, i)
#define V8(x, i) V(x, 8, i)
#define BCAST8(x, i) (x).template replicate_w<8, 1>(i)

template <int M, int N, int K> ESIMD_INLINE void dgetrfnp_panel(int64_t *info) {
  auto a = V(GRF, M * N, 0);
  for (int kk = 0; kk < N; kk += 8) {
    simd_mask<8> mask = 1;
    for (int k = 0; k < 8 && kk + k < N; k++) {
      auto ak = V(a, M, (kk + k) * M);
      auto ak0 = V8(ak, kk + K);

      V1(mask, k) = 0;
      if (ak0[k] != 0.0) {
        // scal
        double temp = 1.0 / ak0[k];
        ak0.merge(ak0 * temp, mask);
        for (int i = 8 + K + kk; i < M; i += 8) {
          V8(ak, i) *= temp;
        }

        // update
        for (int j = kk + k + 1; j < N; j++) {
          auto aj = V(a, M, j * M);
          auto aj0 = V8(aj, kk + K);
          auto temp = BCAST8(aj0, k);
          aj0.merge(aj0 - temp * ak0, aj0, mask);
          for (int i = 8 + K + kk; i < M; i += 8) {
            V8(aj, i) -= temp * V8(ak, i);
          }
        }
      } else if (*info == 0) {
        *info = K + kk + k + 1;
      }
    }
  }
}

// A left-looking algorithm step
// M, N - a panel size to be updated and factorized (M * N <= 64 * 6), must fit
// into GRF K - an update rank P0=A[0:M,0:K] = column(F=A[0:K,0:K],
// L=A[K:M,0:K]) - panel to update with P1=A[0:M,K:K+N] = column(U=A[0:K,K:K+N],
// T=A[K:M,K:K+N]) - panel to be updated
template <int M, int N, int K>
ESIMD_INLINE void dgetrfnp_left_step(double *a, int64_t lda, int64_t *info) {
  auto p1 = V(GRF, M * N, 0);
  double *a1;
  int i, j, k;

  // load P1
  for (j = 0, a1 = a + K * lda; j < N; j++, a1 += lda)
    for (i = 0; i < M; i += 8) {
      simd<double, 8> data;
      data.copy_from(a1 + i);
      V8(p1, j * M + i) = data;
    }
  // (getrf) factorize T=P*L*U
  dgetrfnp_panel<M, N, K>(info);

  // store P1
  for (j = 0, a1 = a + K * lda; j < N; j++, a1 += lda)
    for (i = 0; i < M; i += 8) {
      simd<double, 8> vals = V8(p1, j * M + i);
      vals.copy_to(a1 + i);
    }
}

ESIMD_INLINE void dgetrfnp_esimd_8x8(double *a, int64_t lda, int64_t *ipiv,
                                     int64_t *info) {
  *info = 0;
  dgetrfnp_left_step<8, 8, 0>(a, lda, info);
}

void dgetrfnp_batch_strided_c(queue &queue, int64_t m, int64_t n, double *a,
                              int64_t lda, int64_t stride_a, int64_t *ipiv,
                              int64_t stride_ipiv, int64_t batch,
                              int64_t *info) {
  auto device = queue.get_device();
  auto context = queue.get_context();
  int status;

  CHECK(status = device.is_gpu(), !status);

  double *a_gpu;
  int64_t *ipiv_gpu;
  int64_t *info_gpu;
  CHECK(a_gpu = static_cast<double *>(
            malloc_shared(stride_a * batch * sizeof(double), device, context)),
        !a_gpu);
  CHECK(ipiv_gpu = static_cast<int64_t *>(malloc_shared(
            stride_ipiv * batch * sizeof(int64_t), device, context)),
        !ipiv_gpu);
  CHECK(info_gpu = static_cast<int64_t *>(
            malloc_shared(batch * sizeof(int64_t), device, context)),
        !info_gpu);

  memcpy(a_gpu, a, stride_a * batch * sizeof(double));

  sycl::nd_range<1> range(sycl::range<1>{static_cast<size_t>(batch)},
                          sycl::range<1>{1});
  try {
    auto event = queue.submit([&](handler &cgh) {
      cgh.parallel_for<class dgetrfnp_batch_strided>(
          range, [=](nd_item<1> id) SYCL_ESIMD_KERNEL {
            int i = id.get_global_id(0);
            dgetrfnp_esimd_8x8(&a_gpu[i * stride_a], lda,
                               &ipiv_gpu[i * stride_ipiv], &info_gpu[i]);
          });
    });
    event.wait();
  } catch (const sycl::exception &e) {
    std::cout << "*** EXCEPTION caught: " << e.what() << "\n";
    free(a_gpu, context);
    free(ipiv_gpu, context);
    free(info_gpu, context);
    return;
  }

  memcpy(a, a_gpu, stride_a * batch * sizeof(double));
  memcpy(ipiv, ipiv_gpu, stride_ipiv * batch * sizeof(int64_t));
  memcpy(info, info_gpu, batch * sizeof(int64_t));

  free(a_gpu, context);
  free(ipiv_gpu, context);
  free(info_gpu, context);
}

static void fp_init(int64_t m, int64_t n, double *a, int64_t lda) {
  int64_t i, j;
  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++)
      a[i + j * lda] = 2.0 * FP_RAND - 1.0;
}

static void fp_copy(int64_t m, int64_t n, double *a, int64_t lda, double *b,
                    int64_t ldb) {
  int64_t i, j;
  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++)
      b[i + j * ldb] = a[i + j * lda];
}

static double fp_norm1(int64_t m, int64_t n, double *a, int64_t lda) {
  double sum, value = 0.0;
  int64_t i, j;
  for (j = 0; j < n; j++) {
    sum = 0.0;
    for (i = 0; i < m; i++)
      sum += ABS(a[i + j * lda]);
    if (value < sum)
      value = sum;
  }
  return value;
}

static int dgetrfnp_batch_strided_check(int64_t m, int64_t n, double *a_in,
                                        double *a, int64_t lda,
                                        int64_t stride_a, int64_t *ipiv,
                                        int64_t stride_ipiv, int64_t batch,
                                        int64_t *info) {
  double thresh = 30.0;
  int fail = 0;
  int64_t i, j, k, l;
  char label[1024];
  unsigned char prec_b[] = {0, 0, 0, 0, 0, 0, 0xb0, 0x3c};
  double res = 0.0, nrm = 0.0, ulp = *(double *)prec_b;
  double *w = (double *)malloc(sizeof(double) * MAX(m * n, 1));

  sprintf(label, "m=%ld, n=%ld, lda=%ld, batch=%ld", m, n, lda, batch);

  for (k = 0; k < batch; k++) {
    /* info == 0 */
    CHECK_AND_REPORT("info == 0", label, info[k] != 0, (double)info[k], fail);

    if (m > 0 && n > 0) {
      /* | L U - A | / ( |A| n ulp ) */
      memset(w, 0, sizeof(double) * m * n);
      if (m < n) {
        for (j = 0; j < n; j++)
          for (i = 0; i <= j; i++)
            w[i + j * m] = a[i + j * lda + k * stride_a];
        for (i = m - 1; i >= 0; i--)
          for (j = 0; j < n; j++)
            for (l = 0; l < i; l++)
              w[i + j * m] += a[i + l * lda + k * stride_a] * w[l + j * m];
      } else {
        for (j = 0; j < n; j++)
          for (i = j; i < m; i++)
            w[i + j * m] = a[i + j * lda + k * stride_a];
        for (j = 0; j < n; j++)
          w[j + j * m] = 1.0;
        for (j = n - 1; j >= 0; j--)
          for (i = 0; i < m; i++) {
            w[i + j * m] *= a[j + j * lda + k * stride_a];
            for (l = 0; l < j; l++)
              w[i + j * m] += w[i + l * m] * a[l + j * lda + k * stride_a];
          }
      }
      for (j = 0; j < n; j++)
        for (i = 0; i < m; i++)
          w[i + j * m] -= a_in[k * stride_a + i + j * lda];
      res = fp_norm1(m, n, w, m);
      nrm = fp_norm1(m, n, &a_in[k * stride_a], lda);
      nrm *= (double)n * ulp;
      res /= nrm > 0.0 ? nrm : ulp;
      CHECK_AND_REPORT("| L U - A | / ( |A| n ulp )", label,
                       FAILED(res, thresh), res, fail);
    }
  }

  free(w);
  return fail;
}

int main(int argc, char *argv[]) {
  queue queue(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(queue);

  int exit_status = 0;
  constexpr int64_t m = 8, n = 8, lda = 8;
  int64_t stride_a = lda * n, stride_ipiv = n;

  srand(1);

  for (int i = 1; i < argc; i++) {
    int64_t batch = (int64_t)atoi(argv[i]);
    batch = MAX(batch, 0);
    int64_t a_count = MAX(stride_a * batch, 1);
    int64_t ipiv_count = MAX(stride_ipiv * batch, 1);
    int64_t info_count = MAX(batch, 1);
    double *a = NULL, *a_copy = NULL;
    int64_t *ipiv = NULL, *info = NULL;
    CHECK(a = (double *)malloc(sizeof(double) * a_count), !a);
    CHECK(a_copy = (double *)malloc(sizeof(double) * a_count), !a_copy);
    CHECK(ipiv = (int64_t *)malloc(sizeof(int64_t) * ipiv_count), !ipiv);
    CHECK(info = (int64_t *)malloc(sizeof(int64_t) * info_count), !info);

    /* Initialize input data */
    for (int64_t k = 0; k < batch; k++) {
      fp_init(m, n, &a_copy[k * stride_a], lda);
      fp_copy(m, n, &a_copy[k * stride_a], lda, &a[k * stride_a], lda);
    }

    /* Run the tested function */
    dgetrfnp_batch_strided_c(queue, m, n, a, lda, stride_a, ipiv, stride_ipiv,
                             batch, info);

    /* Check that the computation completed successfully */
    exit_status += dgetrfnp_batch_strided_check(m, n, a_copy, a, lda, stride_a,
                                                ipiv, stride_ipiv, batch, info);

    free(a);
    free(a_copy);
    free(ipiv);
    free(info);
  }
  return exit_status;
}
