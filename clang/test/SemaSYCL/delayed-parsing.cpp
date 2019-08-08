// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only %s
// RUN: %clang_cc1 -std=c++1y -verify -fsyntax-only %s -DDELAYED_TEMPLATE_PARSING -fdelayed-template-parsing
// RUN: %clang %s -DDELAYED_TEMPLATE_PARSING -fsycl -c
// REQUIRES: system-windows

template<typename T> struct U {
#ifndef DELAYED_TEMPLATE_PARSING
  auto f(); // expected-note {{here}}
  int g() { return f(); } // expected-error {{cannot be used before it is defined}}
#else
// expected-no-diagnostics
  auto f();
  int g() { return f(); }
#endif
};
#ifndef DELAYED_TEMPLATE_PARSING
  template int U<int>::g(); // expected-note {{in instantiation of}}
#else
  template int U<int>::g();
#endif
template<typename T> auto U<T>::f() { return T(); }
