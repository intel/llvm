//------------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

#include <sycl/usm.hpp>

template <typename Tb, unsigned int rows, unsigned int cols>
void joint_B_rowmajor_load_store(Tb *B, Tb *OutB, queue &q) {

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

          joint_matrix<sub_group, Tb, use::b, rows, cols, layout::row_major> tB;

          joint_matrix_load(sg, tB, pB, cols);
          ext::intel::experimental::matrix::joint_matrix_store(sg, tB, pOutB,
                                                               cols);
        }); // parallel_for
  });       // queue.submit

  q.wait();
}

template <typename Tb, size_t ROW_SIZE, size_t COL_SIZE> void test(queue &q) {
  Tb *B = malloc_shared<Tb>(ROW_SIZE * COL_SIZE, q);
  Tb *outB = malloc_shared<Tb>(ROW_SIZE * COL_SIZE, q);

  matrix_fill(ROW_SIZE, COL_SIZE, B, [](int i, int j) { return i + j; });

  joint_B_rowmajor_load_store<Tb, ROW_SIZE, COL_SIZE>(B, outB, q);

  assert(matrix_compare(ROW_SIZE, COL_SIZE, outB, B));

  free(B, q);
  free(outB, q);
}

int main(void) {
  queue q;

  test<bfloat16, 8, 16>(q);

  return 0;
}
