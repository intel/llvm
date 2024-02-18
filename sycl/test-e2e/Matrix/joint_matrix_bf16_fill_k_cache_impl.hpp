//------------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

// number of test iterations
constexpr unsigned int testIterations = 100;
// start recording time after X iterations
constexpr unsigned int recordThresh = 10;

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 256
#endif

#ifndef tM
#define tM 8
#endif
#ifndef tN
#define tN TN
#endif
#ifndef tK
#define tK 16
#endif

#ifndef MCACHE1
#define MCACHE1 32
#endif
#ifndef NCACHE1
#define NCACHE1 (TN * 4)
#endif
#ifndef KCACHE1
#define KCACHE1 16
#endif

#ifndef MCACHE2
#define MCACHE2 256
#endif
#ifndef NCACHE2
#define NCACHE2 256
#endif
#ifndef KCACHE2
#define KCACHE2 32
#endif

#ifdef MANUAL_UNROLL
template <class T, T... inds, class F>
static constexpr void loop(std::integer_sequence<T, inds...>, F &&f) {
  (f(std::integral_constant<T, inds>{}), ...); // C++17 fold expression
}

template <class T, T count, class F>
static constexpr void manually_unroll_loop(F &&f) {
  loop(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}
#endif

template <unsigned int rowsA, unsigned int colsA, unsigned int rowsB,
          unsigned int colsB, unsigned int vnniFactor, typename TOperand,
          typename TResult, unsigned int sgSize = SG_SZ>
double joint_matmul(TOperand *A, TOperand *B, TResult *C, queue &q, int i) {
  range<2> global{rowsA / MCACHE1, (colsB / NCACHE1) * sgSize};
  range<2> cachelocal{MCACHE2 / MCACHE1, NCACHE2 / NCACHE1 * sgSize};

  // throw error if padding needed
  assert(colsA == rowsB);
  assert(rowsA % tM == 0);
  assert(colsA % tK == 0);
  assert(colsB % tN == 0);
  // submit main kernel
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();

  auto mk = q.submit([&](handler &h) {
    h.parallel_for( // cache layer#1
        nd_range<2>{global, cachelocal},
        // loop global
        // loop localrange
        [=](nd_item<2> it) [[intel::reqd_sub_group_size(sgSize)]] {
          auto pA =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::no>(A);
          auto pB =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::no>(B);
          auto pC =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::no>(C);
          auto m2 = it.get_group(0);
          auto n2 = it.get_group(1);
          auto m1 = it.get_local_id(0);
          auto n1 = it.get_local_id(1) / sgSize;
          auto sg = it.get_sub_group();
          joint_matrix<sub_group, TResult, use::accumulator, tM, tN>
              tC[MCACHE1 / tM][NCACHE1 / tN]
#ifdef INIT_LIST
              = {joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>(),
                 joint_matrix<sub_group, TResult, use::accumulator, tM, tN>()}
#endif
          ;
#ifdef MANUAL_UNROLL
          manually_unroll_loop<unsigned int, MCACHE1 / tM>([&](auto m) {
            manually_unroll_loop<unsigned int, NCACHE1 / tN>([&](auto n) {
#else
          for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
            for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
#endif
              joint_matrix_fill(sg, tC[m][n], 0);
#ifdef MANUAL_UNROLL
            });
          });
#else
            }
          }
#endif

          for (unsigned int k2 = 0; k2 < colsA / KCACHE2; k2++) {
            joint_matrix<sub_group, TOperand, use::a, tM, tK, layout::row_major>
                tA[MCACHE1 / tM][KCACHE2 / KCACHE1]
#ifdef INIT_LIST
                = {joint_matrix<sub_group, TOperand, use::a, tM, tK,
                                layout::row_major>(),
                   joint_matrix<sub_group, TOperand, use::a, tM, tK,
                                layout::row_major>(),
                   joint_matrix<sub_group, TOperand, use::a, tM, tK,
                                layout::row_major>(),
                   joint_matrix<sub_group, TOperand, use::a, tM, tK,
                                layout::row_major>(),
                   joint_matrix<sub_group, TOperand, use::a, tM, tK,
                                layout::row_major>(),
                   joint_matrix<sub_group, TOperand, use::a, tM, tK,
                                layout::row_major>(),
                   joint_matrix<sub_group, TOperand, use::a, tM, tK,
                                layout::row_major>(),
                   joint_matrix<sub_group, TOperand, use::a, tM, tK,
                                layout::row_major>()}
#endif
            ;

            joint_matrix<sub_group, TOperand, use::b, tK, tN,
                         layout::ext_intel_packed>
                tB[NCACHE1 / tN][KCACHE2 / KCACHE1]
#ifdef INIT_LIST
                =
                    {
                        joint_matrix<sub_group, TOperand, use::b, tK, tN,
                                     layout::ext_intel_packed>(),
                        joint_matrix<sub_group, TOperand, use::b, tK, tN,
                                     layout::ext_intel_packed>(),
                        joint_matrix<sub_group, TOperand, use::b, tK, tN,
                                     layout::ext_intel_packed>(),
                        joint_matrix<sub_group, TOperand, use::b, tK, tN,
                                     layout::ext_intel_packed>(),
                        joint_matrix<sub_group, TOperand, use::b, tK, tN,
                                     layout::ext_intel_packed>(),
                        joint_matrix<sub_group, TOperand, use::b, tK, tN,
                                     layout::ext_intel_packed>(),
                        joint_matrix<sub_group, TOperand, use::b, tK, tN,
                                     layout::ext_intel_packed>(),
                        joint_matrix<sub_group, TOperand, use::b, tK, tN,
                                     layout::ext_intel_packed>(),
                    }
#endif
            ;
#ifdef MANUAL_UNROLL
            manually_unroll_loop<unsigned int, KCACHE2 / KCACHE1>([&](auto k1) {
#else
            for (unsigned int k1 = 0; k1 < KCACHE2 / KCACHE1; k1++) {
#endif
              // physical layer
              unsigned int k = (k2 * KCACHE2 + k1 * KCACHE1) / tK;
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, MCACHE1 / tM>([&](auto m) {
#else
              for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
#endif
                joint_matrix_load(
                    sg, tA[m][k1],
                    pA + (m2 * MCACHE2 + m1 * MCACHE1 + m * tM) * colsA +
                        k * tK,
                    colsA);
#ifdef MANUAL_UNROLL
              }); // m
#else
              } // m
#endif
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, NCACHE1 / tN>([&](auto n) {
#else
              for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
#endif
                joint_matrix_load(
                    sg, tB[n][k1],
                    pB + (k * tK / vnniFactor) * (colsB * vnniFactor) +
                        (n2 * NCACHE2 + n1 * NCACHE1 + n * tN) * vnniFactor,
                    colsB * vnniFactor);
#ifdef MANUAL_UNROLL
              });
#else
              } // n
#endif
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, MCACHE1 / tM>([&](auto m) {
#else
              for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
#endif
#ifdef MANUAL_UNROLL
                manually_unroll_loop<unsigned int, NCACHE1 / tN>([&](auto n) {
#else
                for (unsigned int n = 0; n < NCACHE1 / tN; n++) {

#endif
                  joint_matrix_mad(sg, tC[m][n], tA[m][k1], tB[n][k1],
                                   tC[m][n]);
#ifdef MANUAL_UNROLL
                }); // n
              });   // m
            });     // for k1
#else
                } // n
              }   // m
            }     // k1
#endif
          } // for k2
#ifdef MANUAL_UNROLL
          manually_unroll_loop<unsigned int, MCACHE1 / tM>([&](auto m) {
#else
          for (unsigned int m = 0; m < MCACHE1 / tM; m++) {
#endif
#ifdef MANUAL_UNROLL
            manually_unroll_loop<unsigned int, NCACHE1 / tN>([&](auto n) {
#else
            for (unsigned int n = 0; n < NCACHE1 / tN; n++) {
#endif
              joint_matrix_store(
                  sg, tC[m][n],
                  pC + (m2 * MCACHE2 + m1 * MCACHE1 + m * tM) * colsB +
                      (n2 * NCACHE2 + n1 * NCACHE1 + n * tN),
                  colsB, layout::row_major);
#ifdef MANUAL_UNROLL
            }); // n
          });   // m
#else
            } // n
          }   // m
#endif
        }); // parallel_for
  });       // queue.submit
  if (i == testIterations - 1)
    q.wait();
  std::chrono::duration<double, std::milli> duration =
      std::chrono::high_resolution_clock::now() - start;

  return duration.count();
}

void fill_matrix(bfloat16 *M) {
  std::random_device dev;
  std::uniform_real_distribution<float> fdistr(-1.0, 1.0);
  for (unsigned int i = 0; i < MATRIX_SIZE; i++) {
    for (unsigned int j = 0; j < MATRIX_SIZE; j++) {
      M[i * MATRIX_SIZE + j] = bfloat16(fdistr(dev));
    }
  }
}

void native_matmul(bfloat16 *A, bfloat16 *B, float *C) {
  memset(C, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
  for (unsigned int i = 0; i < MATRIX_SIZE; i++) {
    for (unsigned int k = 0; k < MATRIX_SIZE; k++) {
      for (unsigned int j = 0; j < MATRIX_SIZE; j++) {
        C[i * MATRIX_SIZE + j] += make_fp32(A[i * MATRIX_SIZE + k]) *
                                  make_fp32(B[k * MATRIX_SIZE + j]);
      }
    }
  }
}

int main(void) {
  assert(MATRIX_SIZE >= tM && MATRIX_SIZE >= tK && MATRIX_SIZE >= tN &&
         "invalid matrix size");
  assert((MATRIX_SIZE % tM) == 0 && (MATRIX_SIZE % tN) == 0 &&
         (MATRIX_SIZE % tK) == 0 &&
         "invalid matrix size detected: not a multiple of <tM,tN,tK>");

  queue q;
  bfloat16 *A = malloc_shared<bfloat16>(MATRIX_SIZE * MATRIX_SIZE, q);
  bfloat16 *B = malloc_shared<bfloat16>(MATRIX_SIZE * MATRIX_SIZE, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(MATRIX_SIZE * MATRIX_SIZE, q);
  float *C = malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, q);
  float *refC = malloc_shared<float>(MATRIX_SIZE * MATRIX_SIZE, q);

  // Initialize; fill matrices
  fill_matrix(A);
  fill_matrix(B);
  matrix_vnni<bfloat16>(MATRIX_SIZE, MATRIX_SIZE, B, vnniB, 2);
  native_matmul(A, B, refC);

  // run testIterations time, aggregate and calculate average run time
  double totalDuration = 0;
  for (unsigned int i = 0; i < testIterations; i++) {
    double duration =
        joint_matmul<MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 2,
                     bfloat16, float>(A, vnniB, C, q, i);
    if (i >= recordThresh) {
      totalDuration += duration;
    }
  }

  bool result = matrix_compare(MATRIX_SIZE, MATRIX_SIZE, C, refC);

  double msecPerMatrixMul =
      totalDuration / static_cast<double>(testIterations - recordThresh);
  double gflops = (2.f * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE * 1.0e-9f) /
                  (msecPerMatrixMul / 1000.f);

  std::cout << "DONE for size " << MATRIX_SIZE << std::endl;
  std::cout << "GOPS is " << gflops << " Gop/s" << std::endl;

  free(A, q);
  free(B, q);
  free(vnniB, q);
  free(C, q);
  free(refC, q);

  return !result;
}
