// Copyright (C) Codeplay Software Limited

//===--- kernels.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
// RUN: clang++ -fsycl -fsycl-device-only -emit-mlir %s 2> /dev/null| FileCheck %s --check-prefix=CHECK-MLIR
// RUN: clang++ -fsycl -fsycl-device-only -S -emit-llvm -fsycl-targets=spir64-unknown-unknown-syclmlir %s | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

// CHECK-MLIR: !sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>, !llvm.struct<(ptr<i32, 1>)>)>
// CHECK-MLIR: !sycl_id_1_ = !sycl.id<1>
// CHECK-MLIR: !sycl_item_1_1_ = !sycl.item<[1, true], (!sycl.item_base<[1, true], (!sycl.range<1>, !sycl.id<1>, !sycl.id<1>)>)>
// CHECK-MLIR: !sycl_range_1_ = !sycl.range<1>

// CHECK-LLVM-DAG: %"class.sycl::_V1::accessor.1" = type { %"class.sycl::_V1::detail::AccessorImplDevice.1", { i32 addrspace(1)* } }
// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::AccessorImplDevice.1" = type { %"class.sycl::_V1::id.1", %"class.sycl::_V1::range.1", %"class.sycl::_V1::range.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::id.1" = type { %"class.sycl::_V1::detail::array.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::detail::array.1" = type { [1 x i64] }
// CHECK-LLVM-DAG: %"class.sycl::_V1::range.1" = type { %"class.sycl::_V1::detail::array.1" }
// CHECK-LLVM-DAG: %"class.sycl::_V1::item.1.true" = type { %"struct.sycl::_V1::detail::ItemBase.1.true" }
// CHECK-LLVM-DAG: %"struct.sycl::_V1::detail::ItemBase.1.true" = type { %"class.sycl::_V1::range.1", %"class.sycl::_V1::id.1", %"class.sycl::_V1::id.1" }

// CHECK-MLIR: gpu.module @device_functions
// CHECK-MLIR: gpu.func @_ZTS8kernel_1(%arg0: memref<?xi32, 1>, %arg1: memref<?x!sycl_range_1_> {llvm.byval}, %arg2: memref<?x!sycl_range_1_> {llvm.byval}, %arg3: memref<?x!sycl_id_1_> {llvm.byval})
// CHECK-MLIR-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>,
// CHECK-MLIR-SAME: [[PASSTHROUGH:passthrough = \[\["sycl-module-id", ".*/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp"\], "norecurse", "nounwind", "convergent", "mustprogress"\]]]} {
// CHECK-MLIR-NOT: gpu.func kernel

// CHECK-LLVM: define weak_odr spir_kernel void @_ZTS8kernel_1(i32 addrspace(1)* {{.*}}, [[RANGE_TY:%"class.sycl::_V1::range.1"]]* byval([[RANGE_TY]]) {{.*}}, [[RANGE_TY]]* byval([[RANGE_TY]]) {{.*}}, [[ID_TY:%"class.sycl::_V1::id.1"]]* byval([[ID_TY]]) {{.*}}) #1

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

// CHECK-MLIR: gpu.func @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2(%arg0: memref<?xi32, 1>, %arg1: memref<?x!sycl_range_1_> {llvm.byval}, %arg2: memref<?x!sycl_range_1_> {llvm.byval}, %arg3: memref<?x!sycl_id_1_> {llvm.byval})
// CHECK-MLIR-SAME: kernel attributes {llvm.cconv = #llvm.cconv<spir_kernelcc>, llvm.linkage = #llvm.linkage<weak_odr>, [[PASSTHROUGH]]
// CHECK-MLIR-NOT: gpu.func kernel

// CHECK-LLVM: define weak_odr spir_kernel void @_ZTSZZ6host_2vENKUlRN4sycl3_V17handlerEE_clES2_E8kernel_2({{.*}}) #1

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

// CHECK-MLIR-NOT: SYCLKernel =
SYCL_EXTERNAL void function_1(sycl::item<2, true> item) {
  auto id = item.get_id(0);
}

// Keep at the end of the file.
// CHECK-LLVM: attributes #1 = { convergent mustprogress norecurse nounwind "sycl-module-id"="{{.*}}/polygeist/tools/cgeist/Test/Verification/sycl/kernels.cpp" }
