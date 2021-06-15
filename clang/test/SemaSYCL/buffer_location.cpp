// RUN: %clang_cc1 -fsycl-is-device -Wno-sycl-2017-compat -ast-dump %s
// RUN: %clang_cc1 -fsycl-is-device -Wno-sycl-2017-compat -verify -pedantic %s

// expected-no-diagnostics

#include "Inputs/sycl.hpp"

class another_property {};

template <typename... properties>
class another_property_list {
};

template <int I>
using buffer_location = cl::sycl::INTEL::property::buffer_location::instance<I>;

struct Base {
  int A, B;
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t>
      AccField;
};

struct Captured
    : Base,
      cl::sycl::accessor<char, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer,
                         cl::sycl::access::placeholder::false_t> {
  int C;
};

int main() {
  Captured Obj;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t>
      accessorA;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t>
      accessorB;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t>
      accessorC;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t>
      accessorD;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t>
      accessorE;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t>
      accessorF;
  cl::sycl::kernel_single_task<class kernel_function>(
      [=]() {
        Obj.use();
        accessorA.use();
        accessorB.use();
        accessorC.use();
        accessorD.use();
        accessorE.use();
        accessorF.use();
      });
  return 0;
}
