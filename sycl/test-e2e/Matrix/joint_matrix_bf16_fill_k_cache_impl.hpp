//------------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

#include <random>
#include <sycl/usm.hpp>

#ifdef SLM
#include "slm_utils.hpp"
#endif

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

template <
#if !defined(ARG_DIM) && !defined(RUNTIME_DIM)
          size_t rowsA, size_t colsA, size_t rowsB, size_t colsB,
#endif // ARG_DIM, RUNTIME_DIM
          size_t vnniFactor, typename TOperand, typename TResult, size_t TM,
          size_t TN, size_t TK, size_t MCache1, size_t NCache1, size_t KCache1,
          size_t MCache2, size_t NCache2, size_t KCache2>
double joint_matmul(TOperand *A, TOperand *B, TResult *C, queue &q, int i
#if defined(ARG_DIM) || defined(RUNTIME_DIM)
       , size_t rowsA, size_t colsA, size_t rowsB, size_t colsB
#endif // ARG_DIM, RUNTIME_DIM
       ) {

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
#ifdef SLM
    local_accessor<TOperand, 2> tileA{{MCache2, KCache2}, h};
    local_accessor<TOperand, 2> tileB{
        {KCache2 / vnniFactor, NCache2 * vnniFactor}, h};
#endif

    h.parallel_for<MatMul<TM, TN, TK>>( // cache layer#1
        nd_range<2>{global, cachelocal},
        // loop global
        // loop localrange
        [=](nd_item<2> it)
#ifdef SG_SZ
            [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif // SG_SZ
        {
          // sg::load and sg::store expect decorations to be ON
          auto pA =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::yes>(A);
          auto pB =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::yes>(B);
          auto pC =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::yes>(C);
          auto m2 = it.get_group(0);
          auto n2 = it.get_group(1);
          auto m1 = it.get_local_id(0);
          auto n1 = it.get_local_id(1) / sgSize;
          auto sg = it.get_sub_group();
#ifdef PREFETCH
          size_t sgId = sg.get_group_id()[0];
          // There are MCache2/MCache1 x NCache2/NCache1 subgroups: NumSGs
          // PVC case: this is 8x4 subgroups
          // BKM for PVC is to use prefetch of 8x32 for each subgroup
          constexpr size_t prefRow = 8;
          constexpr size_t prefCol = 32;
      // All the SGs of one workgroup prefetch MCache2xKCache2 of A
      // All the SGs of one workgroup prefetch KCache2xNCache2 of B
      // PVC case: 256x32 of A and 32x256 of B
      // For both A and B: each subgroup performs a prefetch of
      // prefRow rows and prefCol cols at a time
      // For A, the subgroups are distributed along the row dimension:
      // PVC: A layed as MCache2/prefRow (256/32)
      // For B: the subgroups are distributed along the column dimension
      // PVC: NCache2/prefCol = 256/32 = 8 SGs on the column dimension and
      // KCache2/prefRow = 32/8 = 4 SGs on the row dimension
#ifdef VNNI
          // In the VNNI case, each subgroup still gets prefRow x prefCol
          // In the PVC case: subgroups distribution become
          // (NCache2*2)/prefCol = 512/32 = 16 SGs on the column dimension and
          // (KCache2/2)/prefRow = 16/8 = 2 SGs on the row dimension
          // pm1B and pn1B are used to identify the distribution of subgroups
          // along the workgroup prefetch for B matrix. For A matrix, sgId is
          // enough.
          size_t pm1B = sgId / 16;   // prefetch m1 (sgId/16)
          size_t pn1B = sgId & 0x15; // prefetch n1 (sgId%16)
#else                                // VNNI
          size_t pm1B = sgId / 8;   // prefetch m1 (sgId/8)
          size_t pn1B = sgId & 0x7; // prefetch n1 (sgId%8)
#endif                               // VNNI
          constexpr size_t prefDistance = 3;
          for (int p = 0; p < prefDistance; p++)
            joint_matrix_prefetch<prefRow, prefCol>(
                sg, A + (m2 * MCache2 + sgId * prefRow) * colsA + p * prefCol,
                colsA, layout::row_major,
                syclex::properties{syclex::prefetch_hint_L1});

          for (int p = 0; p < prefDistance; p++)
            joint_matrix_prefetch<prefRow, prefCol>(
                sg,
                B +
                    (p * (KCache2 / vnniFactor) + pm1B * prefRow) * colsB *
                        vnniFactor +
                    (n2 * NCache2 * vnniFactor + pn1B * prefCol),
                colsB * vnniFactor, layout::row_major,
                syclex::properties{syclex::prefetch_hint_L1});
#endif // PREFETCH

          joint_matrix<sub_group, TResult, use::accumulator, TM, TN>
              tC[MCache1 / TM][NCache1 / TN]
#ifdef INIT_LIST
              = {}; // default initialization of all array elements
#else               // INIT_LIST
              ; // no initialization
#endif              // INIT_LIST

#ifdef MANUAL_UNROLL
          manually_unroll_loop<unsigned int, MCache1 / TM>([&](auto m) {
            manually_unroll_loop<unsigned int, NCache1 / TN>([&](auto n) {
#else  // MANUAL_UNROLL
          for (unsigned int m = 0; m < MCache1 / TM; m++) {
            for (unsigned int n = 0; n < NCache1 / TN; n++) {
#endif // MANUAL_UNROLL
              joint_matrix_fill(sg, tC[m][n], 0);
#ifdef MANUAL_UNROLL
            });
          });
#else  // MANUAL_UNROLL
            }
          }
#endif // MANUAL_UNROLL

#ifdef SLM
          constexpr unsigned int SGs =
              (MCache2 / MCache1) * (NCache2 / NCache1);
#endif // SLM
          for (unsigned int k2 = 0; k2 < colsA / KCache2; k2++) {
#ifdef SLM
            slm_read_write<colsA, colsB, MCache2, NCache2, KCache2, vnniFactor,
                           SGs>(pA, pB, tileA, tileB, sg, k2, m2, n2, sgSize);
            it.barrier(access::fence_space::local_space);
#endif // SLM
            joint_matrix<sub_group, TOperand, use::a, TM, TK, layout::row_major>
                tA[MCache1 / TM][KCache2 / KCache1]
#ifdef INIT_LIST
                = {}; // default initialization of all array elements
#else                 // INIT_LIST
                ; // no initialization
#endif                // INIT_LIST
#ifdef VNNI
            joint_matrix<sub_group, TOperand, use::b, TK, TN,
                         layout::ext_intel_packed>
                tB[NCache1 / TN][KCache2 / KCache1]
#else  // VNNI
            joint_matrix<sub_group, TOperand, use::b, TK, TN,
                         layout::row_major>
                tB[NCache1 / TN][KCache2 / KCache1]
#endif // VNNI
#ifdef INIT_LIST
                = {}; // default initialization of all array elements
#else                 // INIT_LIST
                ; // no initialization
#endif                // INIT_LIST

#ifdef MANUAL_UNROLL
            manually_unroll_loop<unsigned int, KCache2 / KCache1>([&](auto k1) {
#else  // MANUAL_UNROLL
            for (unsigned int k1 = 0; k1 < KCache2 / KCache1; k1++) {
#endif // MANUAL_UNROLL
       // physical layer
              unsigned int k = (k2 * KCache2 + k1 * KCache1) / TK;
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, MCache1 / TM>([&](auto m) {
#else  // MANUAL_UNROLL
              for (unsigned int m = 0; m < MCache1 / TM; m++) {
#endif // MANUAL_UNROLL
#ifdef SLM
                joint_matrix_load(sg, tA[m][k1],
                                  tileA.template get_multi_ptr<
                                      sycl::access::decorated::no>() +
                                      (m1 * MCache1 + m * TM) * KCache2 +
                                      k1 * TK,
                                  KCache2);
#else // SLM
#ifdef OOB
                ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, tA[m][k1], pA, colsA, rowsA, colsA,
                    m2 * MCache2 + m1 * MCache1 + m * TM, k * TK);
#else  // OOB
                joint_matrix_load(
                    sg, tA[m][k1],
                    pA + (m2 * MCache2 + m1 * MCache1 + m * TM) * colsA +
                        k * TK,
                    colsA);
#endif // OOB
#endif // SLM
#ifdef MANUAL_UNROLL
              }); // m
#else             // MANUAL_UNROLL
              } // m
#endif            // MANUAL_UNROLL
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, NCache1 / TN>([&](auto n) {
#else  // MANUAL_UNROLL
              for (unsigned int n = 0; n < NCache1 / TN; n++) {
#endif // MANUAL_UNROLL
#ifdef SLM
                joint_matrix_load(sg, tB[n][k1],
                                  tileB.template get_multi_ptr<
                                      sycl::access::decorated::no>() +
                                      (k1 * TK / vnniFactor) *
                                          (NCache2 * vnniFactor) +
                                      (n1 * NCache1 + n * TN) * vnniFactor,
                                  NCache2 * vnniFactor);
#else // SLM
#ifdef OOB
                ext::intel::experimental::matrix::joint_matrix_load_checked(
                    sg, tB[n][k1], pB, colsB * vnniFactor, rowsB / vnniFactor,
                    colsB * vnniFactor, k * TK / vnniFactor,
                    (n2 * NCache2 + n1 * NCache1 + n * TN) * vnniFactor);
#else  // OOB
                joint_matrix_load(
                    sg, tB[n][k1],
                    pB + (k * TK / vnniFactor) * (colsB * vnniFactor) +
                        (n2 * NCache2 + n1 * NCache1 + n * TN) * vnniFactor,
                    colsB * vnniFactor);
#endif // OOB
#endif // SLM
#ifdef MANUAL_UNROLL
              }); // n
#else             // MANUAL_UNROLL
              } // n
#endif            // MANUAL_UNROLL
#ifdef MANUAL_UNROLL
              manually_unroll_loop<unsigned int, MCache1 / TM>([&](auto m) {
#else  // MANUAL_UNROLL
              for (unsigned int m = 0; m < MCache1 / TM; m++) {
#endif // MANUAL_UNROLL
#ifdef MANUAL_UNROLL
                manually_unroll_loop<unsigned int, NCache1 / TN>([&](auto n) {
#else // MANUAL_UNROLL
                for (unsigned int n = 0; n < NCache1 / TN; n++) {

#endif // MANUAL_UNROLL
                  joint_matrix_mad(sg, tC[m][n], tA[m][k1], tB[n][k1],
                                   tC[m][n]);
#ifdef MANUAL_UNROLL
                }); // n
              });   // m
            });     // for k1
#else               // MANUAL_UNROLL
                } // n
              } // m
            } // k1
#endif              // MANUAL_UNROLL
#ifdef SLM
            it.barrier(access::fence_space::local_space);
#endif // SLM
#ifdef PREFETCH
            auto prefetch_offsetA = (m2 * MCache2 + sgId * prefRow) * colsA +
                                    (k2 + prefDistance) * prefCol;
            if ((prefetch_offsetA + (prefRow * colsA) + prefCol) <
                (rowsA * colsA))
              joint_matrix_prefetch<prefRow, prefCol>(
                  sg, A + prefetch_offsetA, colsA, layout::row_major,
                  syclex::properties{syclex::prefetch_hint_L1});

            auto prefetch_offsetB =
                ((k2 + prefDistance) * (KCache2 / vnniFactor) +
                 pm1B * prefRow) *
                    (colsB)*vnniFactor +
                (n2 * NCache2 * vnniFactor + pn1B * prefCol);
            if ((prefetch_offsetB + (prefRow * colsB * vnniFactor) +
                 prefCol) < (rowsB * colsB))
              joint_matrix_prefetch<prefRow, prefCol>(
                  sg, B + prefetch_offsetB, colsB * vnniFactor,
                  layout::row_major,
                  syclex::properties{syclex::prefetch_hint_L1});
#endif // PREFETCH
          } // for k2
#ifdef MANUAL_UNROLL
          manually_unroll_loop<unsigned int, MCache1 / TM>([&](auto m) {
#else  // MANUAL_UNROLL
          for (unsigned int m = 0; m < MCache1 / TM; m++) {
#endif // MANUAL_UNROLL
#ifdef MANUAL_UNROLL
            manually_unroll_loop<unsigned int, NCache1 / TN>([&](auto n) {
#else  // MANUAL_UNROLL
            for (unsigned int n = 0; n < NCache1 / TN; n++) {
#endif // MANUAL_UNROLL
#ifdef OOB
              ext::intel::experimental::matrix::joint_matrix_store_checked(
                  sg, tC[m][n], pC, colsB, layout::row_major, rowsA, colsB,
                  m2 * MCache2 + m1 * MCache1 + m * TM,
                  n2 * NCache2 + n1 * NCache1 + n * TN);
#else  // OOB
              joint_matrix_store(
                  sg, tC[m][n],
                  pC + (m2 * MCache2 + m1 * MCache1 + m * TM) * colsB +
                      (n2 * NCache2 + n1 * NCache1 + n * TN),
                  colsB, layout::row_major);
#endif // OOB
#ifdef MANUAL_UNROLL
            }); // n
          });   // m
#else           // MANUAL_UNROLL
            } // n
          } // m
#endif          // MANUAL_UNROLL
        });     // parallel_for
  });           // queue.submit

  if (i == testIterations - 1)
    q.wait();
  std::chrono::duration<double, std::milli> duration =
      std::chrono::high_resolution_clock::now() - start;

  return duration.count();
}

template <typename T, typename TResult, size_t vnniFactor, size_t TM, size_t TN,
          size_t TK, size_t MCache1, size_t NCache1, size_t KCache1,
          size_t MCache2, size_t NCache2, size_t KCache2>
void test(size_t matrix_size_input) {
#ifdef RUNTIME_DIM
  size_t matrix_size = matrix_size_input;
#else
  constexpr size_t matrix_size = MATRIX_SIZE;
#endif // RUNTIME_DIM

  assert(matrix_size >= TM && matrix_size >= TK && matrix_size >= TN &&
         "invalid matrix size");
  assert((matrix_size % TM) == 0 && (matrix_size % TN) == 0 &&
         (matrix_size % TK) == 0 &&
         "invalid matrix size detected: not a multiple of <TM,TN,TK>");

  std::cout << "Testing: " << TM << " x " << TN << " x " << TK
            << " [TM x TN x TK]" << std::endl;

  queue q;
  T *A = malloc_shared<T>(matrix_size * matrix_size, q);
  T *B = malloc_shared<T>(matrix_size * matrix_size, q);
  TResult *C = malloc_shared<TResult>(matrix_size * matrix_size, q);
  TResult *refC = malloc_shared<TResult>(matrix_size * matrix_size, q);

  matrix_rand<T>(matrix_size, matrix_size, A, T(1));
  matrix_rand<T>(matrix_size, matrix_size, B, T(1));

  matrix_multiply_ref<T, T, TResult, 1>(A, B, refC, matrix_size, matrix_size,
                                        matrix_size);

#ifdef VNNI
  T *vnniB = malloc_shared<T>(matrix_size * matrix_size, q);
  matrix_vnni<T>(matrix_size, matrix_size, B, vnniB, vnniFactor);
  free(B, q);
  B = vnniB;
#endif

  // run testIterations time, aggregate and calculate average run time
  double totalDuration = 0;
  for (unsigned int i = 0; i < testIterations; i++) {
    double duration =
            joint_matmul<
#if !defined(ARG_DIM) && !defined(RUNTIME_DIM)
                    matrix_size, matrix_size, matrix_size, matrix_size,
#endif // ARG_DIM, RUNTIME_DIM
                    vnniFactor, T, TResult, TM, TN, TK, MCache1, NCache1,
                    KCache1, MCache2, NCache2, KCache2>
                    (A, B, C, q, i
#if defined(ARG_DIM) || defined(RUNTIME_DIM)
                    , matrix_size, matrix_size, matrix_size, matrix_size
#endif // ARG_DIM, RUNTIME_DIM
                    );

    if (i >= recordThresh) {
      totalDuration += duration;
    }
  }

  assert(matrix_compare(matrix_size, matrix_size, C, refC));

  double msecPerMatrixMul =
      totalDuration / static_cast<double>(testIterations - recordThresh);
  double gflops = (2.f * matrix_size * matrix_size * matrix_size * 1.0e-9f) /
                  (msecPerMatrixMul / 1000.f);

  std::cout << "DONE for size " << matrix_size << std::endl;
  std::cout << "GOPS is " << gflops << " Gop/s" << std::endl;

  free(A, q);
  free(B, q);
  free(C, q);
  free(refC, q);
}

int main(
#ifdef RUNTIME_DIM
  int argc, char *argv[]
#endif //RUNTIME_DIM
  ) {

size_t matrix_size = -1;
#ifdef RUNTIME_DIM
  if (argc == 2) {
    matrix_size = std::stoul(argv[1]);
  } else {
    std::cerr << "Usage: ./program matrix_size\n";
    return 1; // Error if no argument
  }
#endif //RUNTIME_DIM

  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  constexpr size_t MCache1 = 32;
  constexpr size_t MCache2 = 256;
  constexpr size_t NCache2 = 256;
  constexpr size_t KCache2 = 32;

#ifdef VNNI
  constexpr unsigned int VnniFactor = 2;
#else  // VNNI
  constexpr unsigned int VnniFactor = 1;
#endif // VNNI

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      constexpr size_t NCache1 = 32;
      constexpr size_t KCache1 = 32;
      test<bfloat16, float, VnniFactor, /*TM*/ 16, /*TN*/ 16, /*TK*/ 32,
           MCache1, NCache1, KCache1, MCache2, NCache2, KCache2>(matrix_size);
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      constexpr size_t NCache1 = 4 * /*TN*/ 16;
      constexpr size_t KCache1 = 16;
      test<bfloat16, float, VnniFactor, /*TM*/ 8, /*TN*/ 16, /*TK*/ 16, MCache1,
           NCache1, KCache1, MCache2, NCache2, KCache2>(matrix_size);
#if (!defined(SG_SZ) || SG_SZ != 32)
      // These combination are not currently supported for subgroup size = 32 in
      // IGC
      test<bfloat16, float, VnniFactor, /*TM*/ 16, /*TN*/ 16, /*TK*/ 16,
           MCache1, NCache1, KCache1, MCache2, NCache2, KCache2>(matrix_size);
      test<bfloat16, float, VnniFactor, /*TM*/ 32, /*TN*/ 64, /*TK*/ 16,
           MCache1, NCache1, KCache1, MCache2, NCache2, KCache2>(matrix_size);
#endif
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      constexpr size_t NCache1 = 4 * /*TN*/ 8;
      constexpr size_t KCache1 = 16;

      test<bfloat16, float, VnniFactor, /*TM*/ 8, /*TN*/ 8, /*TK*/ 16, MCache1,
           NCache1, KCache1, MCache2, NCache2, KCache2>(matrix_size);
      // test<bfloat16, float, VnniFactor, /*TM*/ 32, /*TN*/ 32, /*TK*/ 16, MCache1,
      //      NCache1, KCache1, MCache2, NCache2, KCache2>(matrix_size);
      break;
    }
  }
  return 0;
}
