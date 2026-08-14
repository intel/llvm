// RUN: %clang_cc1 -std=c++2c -fsycl-is-host -Wundefined-internal -verify %s
// expected-no-diagnostics

// The operand of declcall is an unevaluated operand: its argument
// subexpressions are not odr-used. An internal-linkage function used only as an
// argument of the call must therefore not be diagnosed as used-but-not-defined
// (as it would be in a potentially-evaluated context).
namespace {
int undef_arg();
}
int f(int);
auto p = declcall(f(undef_arg()));
