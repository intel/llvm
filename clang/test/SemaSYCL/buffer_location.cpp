// RUN: %clang_cc1 -fsycl -fsycl-is-device -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -Wno-sycl-2017-compat -verify -pedantic -DTRIGGER_ERROR %s

#include "Inputs/sycl.hpp"

class another_property {};

template <typename... properties>
class another_property_list {
};

struct Base {
  int A, B;
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::property_list<
                         cl::sycl::property::buffer_location<1>>>
      AccField;
};

struct Captured : Base,
                  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read,
                                     cl::sycl::access::target::global_buffer,
                                     cl::sycl::access::placeholder::false_t,
                                     cl::sycl::property_list<
                                         cl::sycl::property::buffer_location<1>>> {
  int C;
};

int main() {
#ifndef TRIGGER_ERROR
  // CHECK: SYCLIntelBufferLocationAttr {{.*}} Implicit 1
  Captured Obj;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::property_list<
                         cl::sycl::property::buffer_location<2>>>
      // CHECK: SYCLIntelBufferLocationAttr {{.*}} Implicit 2
      accessorA;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::property_list<another_property,
                                             cl::sycl::property::buffer_location<3>>>
      // CHECK: SYCLIntelBufferLocationAttr {{.*}} Implicit 3
      accessorB;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::property_list<another_property>>
      accessorC;
#else
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::property_list<
                         cl::sycl::property::buffer_location<-2>>>
      accessorD;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     another_property_list<another_property>>
      accessorE;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::property_list<
                         cl::sycl::property::buffer_location<1>,
                         cl::sycl::property::buffer_location<2>>>
      accessorF;
#endif
  cl::sycl::kernel_single_task<class kernel_function>(
      [=]() {
#ifndef TRIGGER_ERROR
        // expected-no-diagnostics
        Obj.use();
        accessorA.use();
        accessorB.use();
        accessorC.use();
#else
        //expected-error@+1{{buffer_location template parameter must be a non-negative integer}}
        accessorD.use();
        //expected-error@+1{{Fifth template parameter of the accessor must be of a property_list type}}
        accessorE.use();
        //expected-error@+1{{Can't apply buffer_location property twice to the same accessor}}
        accessorF.use();
#endif
      });
  return 0;
}
