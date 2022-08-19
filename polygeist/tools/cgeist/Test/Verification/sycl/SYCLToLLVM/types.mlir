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
// CHECK-DAG: [[ID:.*]] = type { [[ARRAY_1]] }
// CHECK-DAG: [[RANGE_1:.*]] = type { [[ARRAY_1]] }
// CHECK-DAG: [[RANGE_2:.*]] = type { [[ARRAY_2]] }
// CHECK: define void @test_id([[ID]]* %0, [[ID]]* %1)
// CHECK: define void @test_range.1([[RANGE_1]]* %0)
// CHECK: define void @test_range.2([[RANGE_2]]* %0)

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
}
