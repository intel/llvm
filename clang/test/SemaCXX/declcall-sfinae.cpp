// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s
// expected-no-diagnostics

// A declcall that is ill-formed in a SFINAE context is a substitution failure,
// not a hard error.

int f(int);

template <class T> auto sel(int) -> decltype(declcall(f(T{})), char{});
template <class T> auto sel(...) -> int;

// declcall(f(int{})) is well-formed: the first overload is selected.
static_assert(sizeof(decltype(sel<int>(0))) == sizeof(char));

// __builtin_abs(T{}) is a valid call, but declcall rejects builtins. The
// failure must be a substitution failure, selecting the fallback overload,
// rather than a hard error.
template <class T> auto selb(int) -> decltype(declcall(__builtin_abs(T{})), char{});
template <class T> auto selb(...) -> int;
static_assert(sizeof(decltype(selb<int>(0))) == sizeof(int));
