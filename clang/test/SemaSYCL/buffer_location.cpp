// RUN: %clang_cc1 -fsycl-is-device -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -verify -pedantic -DTRIGGER_ERROR %s

#include "Inputs/sycl.hpp"

class another_property {};

template <typename... properties>
class another_property_list {
};

template <int I>
using buffer_location = sycl::ext::intel::property::buffer_location::instance<I>;

struct Base {
  int A, B;
  sycl::accessor<char, 1, sycl::access::mode::read,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<buffer_location<1>>>
      AccField;
};

struct Captured
    : Base,
      sycl::accessor<char, 1, sycl::access::mode::read,
                         sycl::access::target::global_buffer,
                         sycl::access::placeholder::false_t,
                         sycl::ext::oneapi::accessor_property_list<buffer_location<1>>> {
  int C;
};

int main() {
#ifndef TRIGGER_ERROR
  // CHECK: SYCLIntelBufferLocationAttr {{.*}} Implicit 1
  Captured Obj;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<buffer_location<2>>>
      // CHECK: SYCLIntelBufferLocationAttr {{.*}} Implicit 2
      accessorA;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<
                         another_property,
                         buffer_location<3>>>
      // CHECK: SYCLIntelBufferLocationAttr {{.*}} Implicit 3
      accessorB;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<
                         another_property>>
      accessorC;

  sycl::kernel_single_task<class kernel_function>(
      [=]() {
        // expected-no-diagnostics
        Obj.use();
        accessorA.use();
        accessorB.use();
        accessorC.use();
      });
#else
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<buffer_location<-2>>>
      accessorD;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     another_property_list<another_property>>
      accessorE;
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<
                         buffer_location<1>,
                         buffer_location<2>>>
      accessorF;
  sycl::kernel_single_task<class kernel_function>(
      [=]() {
        //expected-error@+1{{buffer_location template parameter must be a non-negative integer}}
        accessorD.use();
        //expected-error@+1{{sixth template parameter of the accessor must be of accessor_property_list type}}
        accessorE.use();
      });
  sycl::kernel_single_task<class kernel_function>(
      [=]() {
        //expected-error@+1{{can't apply buffer_location property twice to the same accessor}}
        accessorF.use();
      });
#endif
  return 0;
}
