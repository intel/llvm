// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate !spirv.Decorations is applied to each fpga_datapath.
// Make sure the mapping from property to SPIR-V decoration is correct

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental; // for fpga_mem

const intel::fpga_datapath<int[10]> empty{};

// CHECK: %[[datapath:.*]] = type { [10 x i32] }
// CHECK: {{.*}}empty = internal addrspace(1) constant %[[datapath]] zeroinitializer, align 4, !spirv.Decorations ![[empty_md:[0-9]*]]

SYCL_EXTERNAL void fpga_datapath_global() {
  int f = 5;
  volatile int ReadVal = empty[f];
}

// CHECK: ![[empty_md]] = !{![[register:[0-9]*]]}
// CHECK: ![[register]] = !{i32 5825}
