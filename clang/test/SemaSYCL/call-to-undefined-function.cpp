// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s

#include "sycl.hpp"

sycl::queue deviceQueue;

void defined() {
}

void undefined(); // #UNDEFINED

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

template <typename T>
void defined_only_in_discarded_stmt(){}

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
    h.single_task<class CallToUndefinedFnTester>([]() { // #CALLOP

      // simple functions
      defined();
      undefinedExternal();
      undefined();
      // expected-error@-1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
      // expected-note@#UNDEFINED {{'undefined' declared here}}
      // expected-note@#CALLOP {{called by 'operator()'}}

      // templated functions
      definedTpl<int>();
      undefinedExternalTpl<int>();
      undefinedTpl<int>();
      // expected-error@-1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
      // expected-note@#CALLOP {{called by 'operator()'}}

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

      // expected-warning@+1 {{constexpr if is a C++17 extension}}
      if constexpr (true) {
        // expected-error@+3 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
        // expected-note@#CALLOP {{called by 'operator()'}}
        // expected-note@#UNDEFINED {{'undefined' declared here}}
        undefined();
      } else {
        // Should not diagnose.
        undefined();
      }

      // Similar to the one above, just make sure the active branch being empty changes nothing.
      // expected-warning@+1 {{constexpr if is a C++17 extension}}
      if constexpr (true) {
      } else {
        // Should not diagnose.
        undefined();
      }

      // Tests that an active 'else'  where 'else' doesn't exist won't crash.
      // expected-warning@+1 {{constexpr if is a C++17 extension}}
      if constexpr (false) {
        // Should not diagnose.
        undefined();
        defined_only_in_discarded_stmt<int>();
      }
    });
  });
}

void forwardDeclFn2() {
}
