// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

int main() {
  using acc_type =
      sycl::accessor<int, 0, sycl::access::mode::read, sycl::target::device>;
  using acc_const_type = sycl::accessor<const int, 0, sycl::access::mode::read,
                                        sycl::target::device>;

  acc_type acc_a;
  auto acc_b = acc_const_type(acc_a);
  assert(acc_a == acc_b);

  acc_const_type acc_c;
  acc_c = acc_a;
  acc_a = acc_b;
  assert(acc_a == acc_c);
  assert(acc_b == acc_c);
  return 0;
}
