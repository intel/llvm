// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -fsyntax-only -Wno-sycl-2017-compat -verify %s
//
// This test checks if compiler reports compilation error on an attempt to use
// a zero-length array inside device code

#include "Inputs/sycl.hpp"
using namespace cl::sycl;
queue q;

typedef float ZEROARR[0];

struct Wrapper {
  int A;
  int BadArray[0]; // expected-note 7{{field of illegal type 'int[0]' declared here}}
};

struct WrapperOfWrapper { // expected-error 2{{zero-length arrays are not permitted in C++}}
  Wrapper F; // expected-note 6{{within field of type 'Wrapper' declared here}}
  ZEROARR *Ptr; //expected-note 5{{field of illegal pointer type 'ZEROARR *' (aka 'float (*)[0]') declared here}}
};

template <unsigned Size> struct InnerTemplated {
  double Array[Size]; // expected-note 8{{field of illegal type 'double[0]' declared here}}
};

template <unsigned Size, typename Ty> struct Templated {
  unsigned A;
  // expected-note@+1 2{{field of illegal type 'double[0]' declared here}}
  Ty Arr[Size]; // expected-note 7{{field of illegal type 'float[0]' declared here}}
  InnerTemplated<Size> Array[Size + 1]; // expected-note 8{{within field of type 'InnerTemplated<0U>[1]' declared here}}
};

struct KernelSt {
  int A;
  int BadArray[0]; // expected-note {{field of illegal type 'int[0]' declared here}}
  void operator()() const {}
};

WrapperOfWrapper offendingFoo() {
  // expected-note@+1 {{called by 'offendingFoo'}}
  return WrapperOfWrapper{};
}

SYCL_EXTERNAL WrapperOfWrapper offendingFooExt();


template <unsigned Size>
void templatedContext() {
  Templated<Size, float> Var;
  // expected-error@#KernelSingleTaskKernelFuncCall 2{{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<TempContext, (lambda at}}
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class TempContext>([=] {
      // expected-note@+1 2{{within field of type 'Templated<0U, float>' declared here}}
      (void)Var; // expected-error 2{{zero-length arrays are not permitted in C++}}
    });
  });
  // expected-error@#KernelSingleTaskKernelFuncCall {{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    // expected-note@+2 {{in instantiation of function template specialization}}
    // expected-note@+1 2{{within field of type 'Templated<0U, float>' declared here}}
    h.single_task<class TempContext1>([Var] {
    });
  });
}

void foo(const unsigned X) {
  int Arr[0]; // expected-note 2{{declaration 'Arr' of illegal type 'int[0]' is here}}
  ZEROARR TypeDef; // expected-note {{declaration 'TypeDef' of illegal type 'ZEROARR' (aka 'float[0]') is here}}
  ZEROARR *Ptr; // expected-note {{declaration 'Ptr' of illegal type 'ZEROARR *' (aka 'float (*)[0]') is here}}
  // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<Simple, (lambda at}}
  // expected-error@#KernelSingleTaskKernelFuncCall 3{{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class Simple>([=]() {
      // expected-note@+1 {{field of illegal type 'int[0]' declared here}}
      (void)Arr; // expected-error {{zero-length arrays are not permitted in C++}}
      // expected-note@+1 {{field of illegal type 'ZEROARR' (aka 'float[0]') declared here}}
      (void)TypeDef; // expected-error {{zero-length arrays are not permitted in C++}}
      // expected-note@+1 {{field of illegal pointer type 'ZEROARR *' (aka 'float (*)[0]') declared here}}
      (void)Ptr; // expected-error {{zero-length arrays are not permitted in C++}}
    });
  });
  // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<Simple1, (lambda at}}
  // expected-error@#KernelSingleTaskKernelFuncCall {{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    // expected-note@+2 {{in instantiation of function template specialization}}
    // expected-note@+1 {{field of illegal type 'int[0]' declared here}}
    h.single_task<class Simple1>([Arr]{ // expected-error {{zero-length arrays are not permitted in C++}}
    });
  });
  WrapperOfWrapper St;
  // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<SimpleStruct, (lambda at}}
  // expected-error@#KernelSingleTaskKernelFuncCall 2{{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class SimpleStruct>([=]{
      // expected-note@+1 2{{within field of type 'WrapperOfWrapper' declared here}}
      (void)St.F.BadArray; // expected-error 4{{zero-length arrays are not permitted in C++}}
    });
  });
  // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<SimpleStruct1, (lambda at}}
  // expected-error@#KernelSingleTaskKernelFuncCall 2{{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    // expected-note@+2 {{in instantiation of function template specialization}}
    // expected-note@+1 2{{within field of type 'WrapperOfWrapper' declared here}}
    h.single_task<class SimpleStruct1>([St]{ // expected-error 2{{zero-length arrays are not permitted in C++}}
    });
  });

  Templated<1, int> OK;
  Templated<1 - 1, double> Weirdo;
  Templated<0, float> Zero;
  // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<UseTemplated, (lambda at}}
  // expected-error@#KernelSingleTaskKernelFuncCall 4{{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
  // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class UseTemplated>([=]{
      (void)OK; // No errors expected
      // expected-note@+1 {{within field of type 'Templated<0, float>' declared here}}
      (void)Zero; // expected-error 2{{zero-length arrays are not permitted in C++}}
      // expected-note@+1 2{{within field of type 'Templated<1 - 1, double>' declared here}}
      int A = Weirdo.A; // expected-error 2{{zero-length arrays are not permitted in C++}}
    });
  });

  // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<UseTemplated1, (lambda at}}
  // expected-error@#KernelSingleTaskKernelFuncCall 2{{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    // expected-note@+2 {{in instantiation of function template specialization}}
    // expected-note@+1 2{{within field of type 'Templated<0, float>' declared here}}
    h.single_task<class UseTemplated1>([Zero]{ // expected-error 2{{zero-length arrays are not permitted in C++}}
    });
  });

  templatedContext<10>();
  // expected-note@+1 2{{in instantiation of function template specialization}}
  templatedContext<0>();

  KernelSt K;
  // expected-error@#KernelSingleTaskKernelFuncCall {{zero-length arrays are not permitted in C++}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  // expected-note@#KernelSingleTask {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class UseFunctor>(K);
  });

  // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<ReturnFromFunc, (lambda at}}
  q.submit([&](handler &h) {
    h.single_task<class ReturnFromFunc>([=] {
      // expected-note@+1 {{called by 'operator()'}}
      offendingFoo();
      // TODO diagnose function types?
      offendingFooExt();
    });
  });
}
