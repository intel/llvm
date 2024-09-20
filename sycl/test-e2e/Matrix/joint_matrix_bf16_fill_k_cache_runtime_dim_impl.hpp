//------------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

#include <random>
#include <sycl/usm.hpp>
template <size_t TM, size_t TN, size_t TK> class MatMul;

template <typename T, typename TResult, size_t vnniFactor, size_t TM, size_t TN,
          size_t TK, size_t MCache1, size_t NCache1, size_t KCache1,
          size_t MCache2, size_t NCache2, size_t KCache2>
void test(size_t matrix_size) {
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
        joint_matmul<vnniFactor, T, TResult, TM, TN, TK, MCache1, NCache1,
                    KCache1, MCache2, NCache2, KCache2>
                    (A, B, C, q, i, 
                    matrix_size, matrix_size, matrix_size, matrix_size);

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

int main(int argc, char *argv[]) {
  size_t matrix_size;

  // Check for command line argument
  if (argc == 2) {
    matrix_size = std::stoul(argv[1]);
  } else {
    std::cerr << "Usage: ./program matrix_size\n";
    return 1; // Error if no argument
  }

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
      // test<bfloat16, float, VnniFactor, /*TM*/ 32, /*TN*/ 32, /*TK*/ 16,
      // MCache1,
      //      NCache1, KCache1, MCache2, NCache2, KCache2>(matrix_size);
      break;
    }
  }
  return 0;
}
