// RUN: %clang++ -fsycl-device-only -fintelfpga -S %s -o - | FileCheck %s

#include "Inputs/sycl.hpp"

// This test checks that FPGA loop metadata is not lost due to optimizations.

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class ivdep>([=](){
      int m =  16;
      int i = 0;
      int b = 1;
      // CHECK: {!"llvm.loop.ivdep.enable"}
      [[intel::ivdep]] do {
        if (i >= m) {
          break;
        } else {
          int b = b *2;
        }
        ++i;
      } while (1);
    });
  });
  return 0;
}
