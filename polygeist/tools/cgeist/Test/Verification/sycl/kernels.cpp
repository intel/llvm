// Copyright (C) Codeplay Software Limited

//===--- kernels.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: sycl-clang.py %s -S | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK-NOT: module

// CHECK: !sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK: !sycl_id_1_ = !sycl.id<1>
// CHECK: !sycl_item_1_1_ = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>
// CHECK: !sycl_range_1_ = !sycl.range<1>

// CHECK: gpu.module @device_functions
// CHECK: gpu.func @_ZTS8kernel_1(%arg0: memref<?xi32>, %arg1: !sycl_range_1_, %arg2: !sycl_range_1_, %arg3: !sycl_id_1_)
// CHECK-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]}
// CHECK-NOT: gpu.func kernel

class kernel_1 {
 sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write> A;

public:
	kernel_1(sycl::accessor<sycl::cl_int, 1, sycl::access::mode::read_write> A) : A(A) {}

  void operator()(sycl::id<1> id) const {
   A[id] = 42;
 }
};

void host_1() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
      auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
      auto ker =  kernel_1{A};
      cgh.parallel_for<kernel_1>(range, ker);
    });
  }
}

// CHECK: gpu.func @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2(%arg0: memref<?xi32>, %arg1: !sycl_range_1_, %arg2: !sycl_range_1_, %arg3: !sycl_id_1_)
// CHECK-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, passthrough = ["norecurse", "nounwind", "convergent", "mustprogress"]}
// CHECK-NOT: gpu.func kernel

void host_2() {
  auto q = sycl::queue{};
  auto range = sycl::range<1>{1};

  {
    auto buf = sycl::buffer<int, 1>{nullptr, range};
    q.submit([&](sycl::handler &cgh) {
      auto A = buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class kernel_2>(range, [=](sycl::id<1> id) {
        A[id] = 42;
      });
    });
  }
}

// CHECK-NOT: SYCLKernel =
SYCL_EXTERNAL void function_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}
