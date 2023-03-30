// UNSUPPORTED: hip

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// Missing GroupNonUniformArithmetic capability on CPU RT
// XFAIL: cpu

// This test verifies the correct work of SPIR-V 1.3 exclusive_scan() and
// inclusive_scan() algoriths used with the operation MUL, bitwise OR, XOR, AND.

#include "scan.hpp"
#include <iostream>

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check_mul<class MulA, int>(Queue);
  check_mul<class MulB, unsigned int>(Queue);
  check_mul<class MulC, long>(Queue);
  check_mul<class MulD, unsigned long>(Queue);
  check_mul<class MulE, float>(Queue);

  check_bit_ops<class A, int>(Queue);
  check_bit_ops<class B, unsigned int>(Queue);
  check_bit_ops<class C, unsigned>(Queue);
  check_bit_ops<class D, long>(Queue);
  check_bit_ops<class E, unsigned long>(Queue);
  check_bit_ops<class F, long long>(Queue);
  check_bit_ops<class G, unsigned long long>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
