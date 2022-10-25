// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -verify %s

// Test diagnostic for nested annotated_arg and annotated_ptr type.

#include "sycl.hpp"

sycl::queue myQueue;

struct MockProperty {};

struct WrappedAnnotatedTypes {
  // expected-error@+1 3{{'sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty>' cannot be a data member of a struct kernel parameter}}
  sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty> AA;
  // expected-error@+1 3{{'sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty>' cannot be a data member of a struct kernel parameter}}
  sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty> AP;
  sycl::accessor<int, 1, sycl::access::mode::read_write> Acc;
};

struct KernelBase {
  sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty> BaseAA; // OK
  sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty> BaseAP; // OK
  WrappedAnnotatedTypes NestedInBase; // Error
};

struct KernelFunctor : KernelBase  {
  sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty> AA; // OK
  sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty> AP; // OK
  void operator()() const {}
};

struct KernelFunctor2  {
  WrappedAnnotatedTypes NestedInField; // Error
  void operator()() const {}
};

int main() {
  sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty> AA;
  sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty> AP;
  WrappedAnnotatedTypes Obj;
  myQueue.submit([&](sycl::handler &h) {
    // expected-note@+1 {{in instantiation of}}
    h.single_task<class kernel_half>(
        [=]() {
          (void)AA;  // OK
	  (void)AP;  // OK
	  (void)Obj; // Error
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    KernelFunctor f;
    // expected-note@+1 {{in instantiation of}}
    h.single_task(f);
  });

  myQueue.submit([&](sycl::handler &h) {
    KernelFunctor2 f2;
    // expected-note@+1 {{in instantiation of}}
    h.single_task(f2);
  });
}

