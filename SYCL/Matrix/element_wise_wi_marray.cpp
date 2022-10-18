//==----------- element_wise_wi_marray.cpp  - DPC++ joint_matrix------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: cuda
// Temp xfail: test was merged early.
// XFAIL: cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 32

template <typename T, size_t M, size_t K> void verify_wi_marray(queue q) {
  int err = 0;
  {
    buffer<int> err_buf(&err, 1);
    q.submit([&](handler &cgh) {
       accessor<int, 1, access::mode::write, target::device> ERR(err_buf, cgh);

       cgh.parallel_for<class marray_kernel>(
           nd_range<2>({1, 1 * SG_SZ}, {1, 1 * SG_SZ}),
           [ERR](nd_item<2> spmd_item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
             auto sg = spmd_item.get_sub_group();

             joint_matrix<T, use::a, M, K, layout::row_major> sub_a;
             joint_matrix<T, use::a, M, K, layout::row_major> sub_a_2;

             joint_matrix_fill(sg, sub_a, -1);
             joint_matrix_fill(sg, sub_a_2, -1);

             auto wi_slice_a = sub_a.get_wi_data();
             for (int i = 0; i < wi_slice_a.length(); i++) {
               wi_slice_a[i] = fabs(wi_slice_a[i]);
             }
             sub_a_2.wi_marray = fabs(sub_a_2.wi_marray);

             for (int i = 0; i < sub_a_2.wi_marray.size(); i++) {
               if (sub_a_2.wi_marray[i] != wi_slice_a[i]) {
                 ERR[0] = 1;
               }
             }
           }); // parallel for
     }).wait();
  }
  assert(err == 0);
}

int main() {

  queue q;
  auto computeCapability =
      std::stof(q.get_device().get_info<sycl::info::device::backend_version>());

  if (computeCapability >= 8.0) {
    verify_wi_marray<bfloat16, 16, 16>(q);
  }

  return 0;
}
