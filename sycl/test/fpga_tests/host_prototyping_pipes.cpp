// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out | FileCheck %s
//==------- host_prototyping_pipes.cpp - SYCL FPGA pipes test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

// For simple non-blocking pipes with explicit type
class some_pipe;
class another_pipe;

int main() {
  using SomePipe = cl::sycl::pipe<some_pipe, int>;
  using AnotherPipe = cl::sycl::pipe<another_pipe, int, 4>;

  // Non-blocking pipes
  bool Success = false;
  // CHECK: Non-blocking write data to pipe:
  // CHECK-SAME: some_pipe
  // CHECK-NEXT: Size: 4 Alignment: 4 Capacity: 0
  SomePipe::write(42, Success);
  int Data = 0;
  // CHECK-NEXT: Non-blocking read data from pipe:
  // CHECK-SAME: some_pipe
  // CHECK-NEXT: Size: 4 Alignment: 4 Capacity: 0
  Data = SomePipe::read(Success);
  // Blocking pipes
  // CHECK-NEXT: Blocking write data to pipe:
  // CHECK-SAME: another_pipe
  // CHECK-NEXT: Size: 4 Alignment: 4 Capacity: 4
  AnotherPipe::write(42);
  // CHECK-NEXT: Blocking read data from pipe:
  // CHECK-SAME: another_pipe
  // CHECK-NEXT: Size: 4 Alignment: 4 Capacity: 4
  Data = AnotherPipe::read();

  return 0;
}
