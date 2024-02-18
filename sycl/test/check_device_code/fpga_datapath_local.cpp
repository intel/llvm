// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Test appropriate llvm.ptr.annotation is applied to each fpga_datapath.
// Make sure the mapping from property to SPIR-V decoration is correct

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental; // for fpga_datapath

// CHECK: [[RegisterINTEL:@.*]] = private unnamed_addr addrspace(1) constant [7 x i8] c"{5825}\00"

int main() {
  queue Q;
  int f = 5;

  Q.single_task([=]() {
    intel::fpga_datapath<int[10]> empty;
    // CHECK: @llvm.ptr.annotation{{.*}}(ptr addrspace(4) {{.*}}, ptr addrspace(1) [[RegisterINTEL]]
    // CHECK-NOT: call void @llvm.memset
    volatile int ReadVal = empty[f];
  });
  return 0;
}
