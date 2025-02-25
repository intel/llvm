//------------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

#include <sycl/usm.hpp>

template <typename Tb, unsigned rows, unsigned cols, unsigned HW_MAX_COL_SIZE>
void joint_B_rowmajor_pair_load_store(Tb *B, Tb *OutB, queue &q) {

  range<1> global{1};
  range<1> local{1};

  q.submit([&](handler &h) {
    h.parallel_for<class Load>(
        nd_range<1>{global, local}, [=](nd_item<1> it)
#ifdef SG_SZ
                                        [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
        {
          auto pB =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::no>(B);
          auto pOutB =
              address_space_cast<sycl::access::address_space::global_space,
                                 sycl::access::decorated::no>(OutB);

          auto sg = it.get_sub_group();

          joint_matrix<sub_group, Tb, use::b, rows, HW_MAX_COL_SIZE,
                       layout::row_major>
              tB[2];

          joint_matrix_load(sg, tB[0], pB, cols);
          joint_matrix_load(sg, tB[1], pB + HW_MAX_COL_SIZE, cols);
          ext::intel::experimental::matrix::joint_matrix_store(sg, tB[0], pOutB,
                                                               cols);
          ext::intel::experimental::matrix::joint_matrix_store(
              sg, tB[1], pOutB + HW_MAX_COL_SIZE, cols);
        }); // parallel_for
  });       // queue.submit

  q.wait();
}

template <typename Tb, size_t ROW_SIZE, size_t COL_SIZE, size_t HW_MAX_COL_SIZE>
void test(queue &q) {
  Tb *B = malloc_shared<Tb>(ROW_SIZE * COL_SIZE, q);
  Tb *outB = malloc_shared<Tb>(ROW_SIZE * COL_SIZE, q);

  matrix_fill(ROW_SIZE, COL_SIZE, B, [](int i, int j) { return i + j; });

  joint_B_rowmajor_pair_load_store<Tb, ROW_SIZE, COL_SIZE, HW_MAX_COL_SIZE>(
      B, outB, q);

  assert(matrix_compare(ROW_SIZE, COL_SIZE, outB, B));

  free(B, q);
  free(outB, q);
}

int main(void) {
  queue q;

  test<bfloat16, 8, 32, 16>(q);
  return 0;
}
