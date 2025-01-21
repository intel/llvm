// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-device -fopenmp-simd -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s

// This test checks whether we diagnose cases of unmarked, undefined
// functions called on device from either kernels or sycl device functions.

#include "sycl.hpp"

sycl::queue deviceQueue;

void defined() {
}

void undefined();
// expected-note@-1 {{'undefined' declared here}}

SYCL_EXTERNAL void undefinedExternal();

template <typename T>
void definedTpl() {
}

template <typename T>
void undefinedTpl();
// expected-note@-1 {{'undefinedTpl<int>' declared here}}

template <typename T>
extern SYCL_EXTERNAL void undefinedExternalTpl();

template <typename T, bool X>
void definedPartialTpl() {
}

template <>
void definedPartialTpl<char, true>() {
}

template <typename T, bool X>
struct Tpl {
  void defined() {
  }
};

template <typename T>
struct Tpl<T, true> {
  void defined() {
  }
};

template <typename T, bool X>
struct TplWithTplMethod {
  template <typename T2, bool Y>
  void defined() {
  }
};

template <typename T>
struct TplWithTplMethod<T, true> {
  template <typename T2, bool Y>
  void defined() {
  }
};

template <typename T, bool X>
struct TplWithTplMethod2 {
  template <typename T2, bool Y>
  void defined() {
  }

  template <>
  void defined<char, true>() {
  }
};

template <typename T>
struct TplWithTplMethod2<T, true> {
  template <typename T2, bool Y>
  void defined() {
  }

  template <>
  void defined<char, true>() {
  }
};

void forwardDeclFn();
void forwardDeclFn2();

void useFwDeclFn() {
  forwardDeclFn();
  forwardDeclFn2();
}

void forwardDeclFn() {
}

int main() {
  // No problems in host code
  undefined();

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class CallToUndefinedFnTester>([]() {
      // expected-note@-1 {{called by 'operator()'}}
      // expected-note@-2 {{called by 'operator()'}}

      // simple functions
      defined();
      undefinedExternal();
      undefined();
      // expected-error@-1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}

      // templated functions
      definedTpl<int>();
      undefinedExternalTpl<int>();
      undefinedTpl<int>();
      // expected-error@-1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}

      // partially specialized template function
      definedPartialTpl<int, false>();
      definedPartialTpl<int, true>();
      definedPartialTpl<char, false>();
      definedPartialTpl<char, true>();

      // template class with specialization
      {
        Tpl<int, false> tpl;
        tpl.defined();
      }

      {
        Tpl<int, true> tpl;
        tpl.defined();
      }

      // template class with template method, both have specializations.
      {
        TplWithTplMethod<int, false> tpl;
        tpl.defined<char, false>();
        tpl.defined<char, true>();
        tpl.defined<int, false>();
        tpl.defined<int, true>();
      }

      {
        TplWithTplMethod<int, true> tpl;
        tpl.defined<char, false>();
        tpl.defined<char, true>();
        tpl.defined<int, false>();
        tpl.defined<int, true>();
      }

      {
        TplWithTplMethod2<int, false> tpl;
        tpl.defined<char, false>();
        tpl.defined<char, true>();
        tpl.defined<int, false>();
        tpl.defined<int, true>();
      }

      {
        TplWithTplMethod2<int, true> tpl;
        tpl.defined<char, false>();
        tpl.defined<char, true>();
        tpl.defined<int, false>();
        tpl.defined<int, true>();
      }

      // forward-declared function
      useFwDeclFn();
      forwardDeclFn();
      forwardDeclFn2();
    });
  });
}

void forwardDeclFn2() {
}
