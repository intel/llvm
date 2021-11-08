// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s -o %t.out
// expected-error@CL/sycl/stream.hpp:* {{Convert the byte to a numeric value using std::to_integer}}

#include <CL/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([](auto &h) {
    sycl::stream os(1024, 256, h);
    h.single_task([=] { os << std::byte(2) << "\n"; });
  });
  q.wait();
}
