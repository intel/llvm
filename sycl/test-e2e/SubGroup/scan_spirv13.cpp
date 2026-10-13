// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies the correct work of SPIR-V 1.3 exclusive_scan() and
// inclusive_scan() algoriths used with the operation MUL, bitwise OR, XOR, AND.

#include "scan.hpp"
#include <iostream>

int main() {
  queue Queue;
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
