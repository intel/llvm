//------------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

#include <random>
#include <sycl/usm.hpp>

// number of test iterations
constexpr unsigned int testIterations = 100;
// start recording time after X iterations
constexpr unsigned int recordThresh = 10;

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 256
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

template <size_t TM, size_t TN, size_t TK> class MatMul;

template <size_t rowsA, size_t colsA, size_t rowsB, size_t colsB, size_t VNNI,
          typename TOperand, typename TResult, size_t TM, size_t TN, size_t TK,
          size_t MCache1, size_t NCache1, size_t KCache1, size_t MCache2,
          size_t NCache2, size_t KCache2>
double joint_matmul(TOperand *A, TOperand *B, TResult *C, queue &q, int i) {
  size_t sgSize = get_sg_size<MatMul<TM, TN, TK>>(q);
  range<2> global{rowsA / MCache1, (colsB / NCache1) * sgSize};
  range<2> cachelocal{MCache2 / MCache1, NCache2 / NCache1 * sgSize};

  // throw error if padding needed
  assert(colsA == rowsB);
  assert(rowsA % TM == 0);
  assert(colsA % TK == 0);
  assert(colsB % TN == 0);
  // submit main kernel
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();

  q.submit([&](handler &h) {
    h.parallel_for<MatMul<TM, TN, TK>>( // cache layer#1
        nd_range<2>{global, cachelocal},
        // loop global
        // loop localrange
        [=](nd_item<2> it)
#ifdef SG_SZ
            [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
        {
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
          joint_matrix<sub_group, TResult, use::accumulator, TM, TN>
              tC[MCache1 / TM][NCache1 / TN]
#ifdef INIT_LIST
              = {}; // default initialization of all array elements
#else
              ; // no initialization
#endif

#ifdef MANUAL_UNROLL
          manually_unroll_loop<unsigned int, MCache1 / TM>([&](auto m) {
            manually_unroll_loop<unsigned int, NCache1 / TN>([&](auto n) {
#else
          for (unsigned int m = 0; m < MCache1 / TM; m++) {
            for (unsigned int n = 0; n < NCache1 / TN; n++) {
#endif
              joint_matrix_fill(sg, tC[m][n], 0);
#ifdef MANUAL_UNROLL
            });
          });
#else
            }
          }
#endif

          for (unsigned int k2 = 0; k2 < colsA / KCache2; k2++) {
            joint_matrix<sub_group, TOperand, use::a, TM, TK, layout::row_major>
                tA[MCache1 / TM][KCache2 / KCache1]
#ifdef INIT_LIST
                = {}; // default initialization of all array elements
#else
                ; // no initialization
#endif

            joint_matrix<sub_group, TOperand, use::b, TK, TN,
                         layout::ext_intel_packed>
                tB[NCache1 / TN][KCache2 / KCache1]
#ifdef INIT_LIST
                = {}; // default initialization of all array elements
#else
                ; // no initialization
#endif

#ifdef MANUAL_UNROLL
            manually_unroll_loop<unsigned int, KCache2 / KCache1>([&](auto k1) {
#else
            for (unsigned int k1 = 0; k1 < KCache2 / KCache1; k1++) {
#endif
              // physical layer
              unsigned int k = (k2 * KCache2 + k1 * KCache1) / TK;
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, MCache1 / TM>([&](auto m) {
#else
              for (unsigned int m = 0; m < MCache1 / TM; m++) {
#endif
#ifdef OOB
                ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, tA[m][k1], pA, colsA, rowsA, colsA,
                    m2 * MCache2 + m1 * MCache1 + m * TM, k * TK);
#else
                joint_matrix_load(
                    sg, tA[m][k1],
                    pA + (m2 * MCache2 + m1 * MCache1 + m * TM) * colsA +
                        k * TK,
                    colsA);
#endif
#ifdef MANUAL_UNROLL
              }); // m
#else
              } // m
#endif
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, NCache1 / TN>([&](auto n) {
#else
              for (unsigned int n = 0; n < NCache1 / TN; n++) {
#endif
#ifdef OOB
                ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, tB[n][k1], pB, colsB * VNNI, rowsB / VNNI, colsB * VNNI,
                    k * TK / VNNI,
                    (n2 * NCache2 + n1 * NCache1 + n * TN) * VNNI);
#else
                joint_matrix_load(sg, tB[n][k1],
                                  pB + (k * TK / VNNI) * (colsB * VNNI) +
                                      (n2 * NCache2 + n1 * NCache1 + n * TN) *
                                          VNNI,
                                  colsB * VNNI);
#endif
#ifdef MANUAL_UNROLL
              });
#else
              } // n
#endif
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, MCache1 / TM>([&](auto m) {
#else
              for (unsigned int m = 0; m < MCache1 / TM; m++) {
#endif
#ifdef MANUAL_UNROLL
                manually_unroll_loop<unsigned int, NCache1 / TN>([&](auto n) {
#else
                for (unsigned int n = 0; n < NCache1 / TN; n++) {

#endif
                  joint_matrix_mad(sg, tC[m][n], tA[m][k1], tB[n][k1],
                                   tC[m][n]);
#ifdef MANUAL_UNROLL
                }); // n
              });   // m
            });     // for k1
#else
                } // n
              } // m
            } // k1
#endif
          } // for k2
#ifdef MANUAL_UNROLL
          manually_unroll_loop<unsigned int, MCache1 / TM>([&](auto m) {
#else
          for (unsigned int m = 0; m < MCache1 / TM; m++) {
#endif
#ifdef MANUAL_UNROLL
            manually_unroll_loop<unsigned int, NCache1 / TN>([&](auto n) {
#else
            for (unsigned int n = 0; n < NCache1 / TN; n++) {
#endif
#ifdef OOB
              ext::intel::experimental::matrix::joint_matrix_store_checked(
                  sg, tC[m][n], pC, colsB, layout::row_major, rowsA, colsB,
                  m2 * MCache2 + m1 * MCache1 + m * TM,
                  n2 * NCache2 + n1 * NCache1 + n * TN);
#else
              joint_matrix_store(
                  sg, tC[m][n],
                  pC + (m2 * MCache2 + m1 * MCache1 + m * TM) * colsB +
                      (n2 * NCache2 + n1 * NCache1 + n * TN),
                  colsB, layout::row_major);
#endif
#ifdef MANUAL_UNROLL
            }); // n
          });   // m
#else
            } // n
          } // m
#endif
        }); // parallel_for
  });       // queue.submit

  if (i == testIterations - 1)
    q.wait();
  std::chrono::duration<double, std::milli> duration =
      std::chrono::high_resolution_clock::now() - start;

  return duration.count();
}

template <typename T, typename TResult, size_t VNNI, size_t TM, size_t TN,
          size_t TK, size_t MCache1, size_t NCache1, size_t KCache1,
          size_t MCache2, size_t NCache2, size_t KCache2>
void test() {
  assert(MATRIX_SIZE >= TM && MATRIX_SIZE >= TK && MATRIX_SIZE >= TN &&
         "invalid matrix size");
  assert((MATRIX_SIZE % TM) == 0 && (MATRIX_SIZE % TN) == 0 &&
         (MATRIX_SIZE % TK) == 0 &&
         "invalid matrix size detected: not a multiple of <TM,TN,TK>");

  std::cout << "Testing: " << TM << " x " << TN << " x " << TK
            << " [TM x TN x TK]" << std::endl;

  queue q;
  T *A = malloc_shared<T>(MATRIX_SIZE * MATRIX_SIZE, q);
  T *B = malloc_shared<T>(MATRIX_SIZE * MATRIX_SIZE, q);
  T *vnniB = malloc_shared<T>(MATRIX_SIZE * MATRIX_SIZE, q);
  TResult *C = malloc_shared<TResult>(MATRIX_SIZE * MATRIX_SIZE, q);
  TResult *refC = malloc_shared<TResult>(MATRIX_SIZE * MATRIX_SIZE, q);

  matrix_rand<T>(MATRIX_SIZE, MATRIX_SIZE, A, T(1));
  matrix_rand<T>(MATRIX_SIZE, MATRIX_SIZE, B, T(1));
  matrix_vnni<T>(MATRIX_SIZE, MATRIX_SIZE, B, vnniB, VNNI);

  matrix_multiply_ref<T, T, TResult, 1>(A, B, refC, MATRIX_SIZE, MATRIX_SIZE,
                                        MATRIX_SIZE);

  // run testIterations time, aggregate and calculate average run time
  double totalDuration = 0;
  for (unsigned int i = 0; i < testIterations; i++) {
    double duration =
        joint_matmul<MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, VNNI,
                     T, TResult, TM, TN, TK, MCache1, NCache1, KCache1, MCache2,
                     NCache2, KCache2>(A, vnniB, C, q, i);
    if (i >= recordThresh) {
      totalDuration += duration;
    }
  }

  assert(matrix_compare(MATRIX_SIZE, MATRIX_SIZE, C, refC));

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
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  constexpr size_t MCache1 = 32;
  constexpr size_t MCache2 = 256;
  constexpr size_t NCache2 = 256;
  constexpr size_t KCache2 = 32;

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      constexpr size_t NCache1 = 32;
      constexpr size_t KCache1 = 32;

      test<bfloat16, float, 2, /*TM*/ 16, /*TN*/ 16, /*TK*/ 32, MCache1,
           NCache1, KCache1, MCache2, NCache2, KCache2>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      constexpr size_t NCache1 = 4 * /*TN*/ 16;
      constexpr size_t KCache1 = 16;

      test<bfloat16, float, 2, /*TM*/ 8, /*TN*/ 16, /*TK*/ 16, MCache1, NCache1,
           KCache1, MCache2, NCache2, KCache2>();
#if (!defined(SG_SZ) || SG_SZ != 32)
      // These combination are not currently supported for subgroup size = 32 in
      // IGC
      test<bfloat16, float, 2, /*TM*/ 16, /*TN*/ 16, /*TK*/ 16, MCache1,
           NCache1, KCache1, MCache2, NCache2, KCache2>();
      test<bfloat16, float, 2, /*TM*/ 32, /*TN*/ 64, /*TK*/ 16, MCache1,
           NCache1, KCache1, MCache2, NCache2, KCache2>();
#endif
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      constexpr size_t NCache1 = 4 * /*TN*/ 8;
      constexpr size_t KCache1 = 16;

      test<bfloat16, float, 2, /*TM*/ 8, /*TN*/ 8, /*TK*/ 16, MCache1, NCache1,
           KCache1, MCache2, NCache2, KCache2>();
      break;
    }
  }
  return 0;
}
