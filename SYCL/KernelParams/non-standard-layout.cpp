// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

using namespace cl::sycl;

struct F1 {};
struct F2 {};
struct F : F1, F2 {
  cl::sycl::cl_char x;
};

bool test0() {
  F S;
  S.x = 0;
  F S0;
  S0.x = 1;
  {
    buffer<F, 1> Buf(&S0, range<1>(1));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = Buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class NonStandard>([=] { B[0] = S; });
    });
  }
  bool Passed = (S.x == S0.x);

  if (!Passed) {
    std::cout << "test0 failed" << std::endl;
  }
  return Passed;
}

int main() {

  bool Pass = test0();

  std::cout << "Test " << (Pass ? "passed" : "FAILED") << std::endl;
  return Pass ? 0 : 1;
}
