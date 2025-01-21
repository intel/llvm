// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s
//
// This test checks that the compiler issues an error on attempt to capture
// "this" pointer by lambdas passed to the device code (directly and indirectly)

#include "Inputs/sycl.hpp"
using namespace sycl;
queue q;

class Class {
public:
  Class() : member(1) {}
  void function();
  void function2();
  void function3();
  int member;
};

void Class::function3() {
  auto Lambda = [=]() {
    int acc[1] = {5};
    acc[0] *= member;
  };
}

void Class::function2() {
  auto Lambda = [=]() {
    int acc[1] = {5};
    acc[0] *= member;
  };
  function3();
}

void Class::function() {
  auto Lambda = [=]() {
    int acc[1] = {5};
    acc[0] *= member; // expected-error 2{{implicit capture of 'this' is not allowed for kernel functions}}
  };
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class Simple>([=]() {
      int acc[1] = {5};
      acc[0] *= member; // expected-error{{implicit capture of 'this' is not allowed for kernel functions}}
    });
  });
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class CapturedOnDevice>([=]() {
      auto DeviceLambda = [=]() {
        int acc[1] = {5};
        acc[0] *= member; // expected-error{{implicit capture of 'this' is not allowed for kernel functions}}
      };
      DeviceLambda();
    });
  });
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class CapturedOnDevice1>([=]() {
      // FIXME: That is probably not correct source location for a diagnostic
      function2(); // expected-error{{implicit capture of 'this' is not allowed for kernel functions}}
    });
  });
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class CapturedOnDevice2>([=]() {
      // FIXME: That is probably not correct source location for a diagnostic
      function3(); // expected-error{{implicit capture of 'this' is not allowed for kernel functions}}
    });
  });
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class CapturedOnHost>([=]() {
      Lambda();
    });
  });
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class CapturedOnHost1>([Lambda]() {
    });
  });
  auto InnerLambda = [=]() {
    int A = 2 + member; // expected-error {{implicit capture of 'this' is not allowed for kernel functions}}
  };
  auto ExternalLambda = [=]() {
    InnerLambda();
  };
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class CapturedOnHost2>([=]() {
      ExternalLambda();
    });
  });
}

int main(int argc, char *argv[]) {
  Class c;
  c.function();
}
