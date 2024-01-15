// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include "sycl/sycl.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iostream>

// clang-format on

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

struct MyIP {
  annotated_ptr<int, decltype(properties(buffer_location<0>, awidth<32>,
                                         dwidth<32>, alignment<8>))>
      a;

  int b;

  MyIP(int *a_, int b_) : a(a_), b(b_) {}

  void operator()() const {
    for (int i = 0; i < b - 2; i++) {
      // expected-error@+1 3+ {{overload resolution selected deleted operator '[]'}}
      a[i + 2] = a[i + 1] + a[i];
    }
  }
};

void TestVectorAddWithAnnotatedMMHosts() {
  sycl::queue q;

  auto raw = malloc_shared<int>(5, q);
  annotated_ptr<int> x1{raw};
  x1++; // OK, since annotated_ptr properties don't contain alignment

  annotated_ptr<int, decltype(properties(alignment<8>))> x2;
  // expected-error@+1 {{overload resolution selected deleted operator '++'}}
  x2++;
  // expected-error@+1 {{overload resolution selected deleted operator '--'}}
  x2--;
  // expected-error@+1 {{overload resolution selected deleted operator '+'}}
  x2 + 1;

  q.submit([&](handler &h) { h.single_task(MyIP{raw, 5}); }).wait();

  for (int i = 0; i < 5; i++) {
    std::cout << raw[i] << std::endl;
  }

  free(raw, q);
}

int main() {
  TestVectorAddWithAnnotatedMMHosts();
  return 0;
}
