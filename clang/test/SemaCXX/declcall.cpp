// RUN: %clang_cc1 -std=c++2c -fsycl-is-host -fsyntax-only -verify %s

// Baseline semantics for declcall (P2825): overload resolution, operand
// requirements, and dependent-operand instantiation.

int f(int);
int f(char);
using FPInt = int (*)(int);
using FPChar = int (*)(char);

// declcall(f(0)) resolves the overload as if calling f(0) and yields a
// pointer to the selected function.
constexpr auto p = declcall(f(0));
static_assert(__is_same(decltype(p), int (*const)(int)));
static_assert(p == static_cast<FPInt>(f));

// A char argument selects the other overload.
constexpr auto pc = declcall(f('a'));
static_assert(__is_same(decltype(pc), int (*const)(char)));

// declcall preserves the exact function type, including noexcept.
int nef(int) noexcept;
static_assert(__is_same(decltype(declcall(nef(0))), int (*)(int) noexcept));
static_assert(!__is_same(decltype(declcall(nef(0))), int (*)(int)));
struct NE { int m(int) noexcept; };
static_assert(
    __is_same(decltype(declcall(((NE *)0)->m(0))), int (NE::*)(int) noexcept));

// declcall deduces and selects the right function template specialization.
template <class T> T ttmpl(T);
static_assert(__is_same(decltype(declcall(ttmpl(0))), int (*)(int)));
static_assert(__is_same(decltype(declcall(ttmpl('a'))), char (*)(char)));

// declcall selects the right cv/ref-qualified member overload and preserves
// the qualifier in the resulting member-pointer type.
struct CV {
  int m(int);
  int m(int) const;
  int r(int) &;
  int r(int) &&;
};
static_assert(__is_same(decltype(declcall(((CV *)0)->m(0))), int (CV::*)(int)));
static_assert(
    __is_same(decltype(declcall(((const CV *)0)->m(0))), int (CV::*)(int) const));
CV cv;
static_assert(__is_same(decltype(declcall(cv.r(0))), int (CV::*)(int) &));
static_assert(__is_same(decltype(declcall(CV{}.r(0))), int (CV::*)(int) &&));

// declcall of a static member function yields a plain function pointer, not a
// member pointer.
struct SM {
  static int sf(int);
  static int sf(char);
};
static_assert(__is_same(decltype(declcall(SM::sf(0))), int (*)(int)));
static_assert(__is_same(decltype(declcall(SM::sf('a'))), int (*)(char)));

// declcall on an operator call selects the operator function.
struct OpA {};
int operator+(OpA, OpA);
struct OpM { int operator-(OpM); };
static_assert(__is_same(decltype(declcall(OpA{} + OpA{})), int (*)(OpA, OpA)));
static_assert(
    __is_same(decltype(declcall(((OpM *)0)->operator-(OpM{}))), int (OpM::*)(OpM)));

// The operand must be a call expression.
int x;
auto bad = declcall(x); // expected-error {{declcall doesn't contain a call}}

// A call through a runtime function pointer is not constant-evaluable. This
// must produce exactly one diagnostic: the failed declcall must not leave a
// live expression behind that cascades into further errors.
int (*fp)(int) = &f;
constexpr auto rt = declcall(fp(0)); // expected-error {{declcall must not depend on runtime known value}}

// A call through a pointer to member has no compile-time-known callee. This
// must be diagnosed, not crash the compiler.
struct S { int m(int); };
S s;
int (S::*pmf)(int);
auto pm = declcall((s.*pmf)(0)); // expected-error {{declcall doesn't support a call through a pointer to member yet}}

// declcall is ill-formed for destructors and builtin functions.
struct D { ~D(); };
void ill_formed(D *d) {
  auto pd = declcall(d->~D()); // expected-error {{declcall doesn't support a destructor yet}}
  (void)pd;
}
auto pbuiltin = declcall(__builtin_abs(0)); // expected-error {{declcall doesn't support a builtin function yet}}

// Dependent operands are resolved at instantiation time.
template <class T> auto call(T v) { return declcall(f(v)); }
void use() {
  static_assert(__is_same(decltype(call(0)), FPInt));
  static_assert(__is_same(decltype(call('a')), FPChar));
}
