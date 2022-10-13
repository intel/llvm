// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -verify %s

#include "sycl.hpp"

sycl::queue myQueue;


struct MockProperty {};

struct WrappedAnnotatedTypes {
  // expected-error@+1 2{{Kernel argument of type 'sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty>' cannot be nested}}
  sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty> AA;
  // expected-error@+1 2{{Kernel argument of type 'sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty>' cannot be nested}}
  sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty> AP;
  sycl::accessor<int, 1, sycl::access::mode::read_write> Acc;
};

struct KernelBase {
  sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty> BaseAA;
  sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty> BaseAP;
  WrappedAnnotatedTypes NestedInBase;
};

struct KernelFunctor : KernelBase  {
  sycl::ext::oneapi::experimental::annotated_arg<int, MockProperty> AA;
  sycl::ext::oneapi::experimental::annotated_ptr<int, MockProperty> AP;
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

}

