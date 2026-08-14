// RUN: %clang_cc1 -std=c++2c -fsycl-is-host -fsyntax-only -verify %s

// declcall builds a qualified reference to the selected member function to form
// its pointer to member. That must not disable the normal diagnostics for
// improperly taking the address of a member function.

struct S {
  void m();
  ~S();
  void f() {
    auto p = &m; // expected-error {{must explicitly qualify name of member function when taking its address}}
    (void)p;
  }
};

auto pd = &S::~S; // expected-error {{taking the address of a destructor}}

// declcall of a member function is still accepted.
auto ok = declcall(((S *)0)->m());
