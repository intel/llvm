// REQUIRES: matrix-xmx8

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define N_THREADS_PER_MATRIX_OP 8

#include "../joint_matrix_gemm.hpp"


int main() {

queue Q;
/////// Variations that do work
test<bfloat16, float, float, SUB_TILES_M, SUB_TILES_K, SUB_TILES_N, 8, 16, 8>(
        Q);
}
