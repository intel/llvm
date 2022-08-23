// Copyright (C) Intel

//===--- types.mlir ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: cgeist %s -S -emit-llvm --args -fsycl-is-device | FileCheck %s

// CHECK-DAG: [[ARRAY_1:.*]] = type { [1 x i64] }
// CHECK-DAG: [[ARRAY_2:.*]] = type { [2 x i64] }
// CHECK-DAG: [[ID_1:%"class.cl::sycl::id.*]] = type { [[ARRAY_1]] }
// CHECK-DAG: [[ID_2:%"class.cl::sycl::id.*]] = type { [[ARRAY_2]] }
// CHECK-DAG: [[RANGE_1:%"class.cl::sycl::range.*]] = type { [[ARRAY_1]] }
// CHECK-DAG: [[RANGE_2:%"class.cl::sycl::range.*]] = type { [[ARRAY_2]] }
// CHECK-DAG: [[ACCESSORIMPLDEVICE_1:%"class.cl::sycl::detail::AccessorImplDevice.*]] = type { [[ID_1]], [[RANGE_1]], [[RANGE_1]] }
// CHECK-DAG: [[ACCESSORIMPLDEVICE_2:%"class.cl::sycl::detail::AccessorImplDevice.*]] = type { [[ID_2]], [[RANGE_2]], [[RANGE_2]] }
// CHECK-DAG: [[ACCESSOR_1:%"class.cl::sycl::accessor.*]] = type { [[ACCESSORIMPLDEVICE_1]], { i32 addrspace(1)* } }
// CHECK-DAG: [[ACCESSOR_2:%"class.cl::sycl::accessor.*]] = type { [[ACCESSORIMPLDEVICE_2]], { i64 addrspace(1)* } }
// CHECK: define void @test_id([[ID_1]] %0, [[ID_1]] %1)
// CHECK: define void @test_range.1([[RANGE_1]] %0)
// CHECK: define void @test_range.2([[RANGE_2]] %0)
// CHECK: define void @test_accessorImplDevice([[ACCESSORIMPLDEVICE_1]] %0)
// CEHCK: define void @test_accessor.1([[ACCESSOR_1]] %0)
// CEHCK: define void @test_accessor.2([[ACCESSOR_2]] %0)

module {
  func.func @test_id(%arg0: !sycl.id<1>, %arg1: !sycl.id<1>) {
    return
  }
  func.func @test_range.1(%arg0: !sycl.range<1>) {
    return
  }
  func.func @test_range.2(%arg0: !sycl.range<2>) {
    return
  }
  func.func @test_accessorImplDevice(%arg0: !sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>) {
    return
  }
  func.func @test_accessor.1(%arg0: !sycl.accessor<[1, i32, write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>)>) {
    return
  }
  func.func @test_accessor.2(%arg0: !sycl.accessor<[2, i64, write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl.id<2>, !sycl.range<2>, !sycl.range<2>)>)>) {
    return
  }
}
