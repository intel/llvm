// RUN: %clangxx -S -emit-llvm -fsycl-device-only %s -o - -Xclang -disable-llvm-passes | FileCheck %s

//==------------------ no_intrinsics.cpp - SYCL stream test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test to check that intrinsics to get a global id are not generated for the
// stream.

// CHECK-NOT: call spir_func void @{{.*}}__spirvL22initGlobalInvocationId{{.*}}

#include <sycl/sycl.hpp>

using namespace sycl;

SYCL_EXTERNAL void integral(stream Out) { Out << "Hello, World!\n"; }
