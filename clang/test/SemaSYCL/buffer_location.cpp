// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -verify -pedantic -DTRIGGER_ERROR %s

#include "sycl.hpp"

class another_property {};

template <typename... properties>
class another_property_list {
};

int main() {
#ifndef TRIGGER_ERROR
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
        accessorA.use();
        accessorB.use();
        accessorC.use();
#else
        //expected-error@+1{{buffer_location template parameter must be a compiletime known non-negative integer}}
        accessorD.use();
        //expected-error@+1{{accessor's 5th template parameter must be a property_list}}
        accessorE.use();
        //expected-error@+1{{Can't apply buffer_location property twice to the same accessor}}
        accessorF.use();
#endif
      });
  return 0;
}
