// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -fsyntax-only %s
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify \
// RUN:  -aux-triple x86_64-pc-windows-msvc -fsyntax-only %s
//
// Ensure SYCL type restrictions are applied to accessors as well.

#include "Inputs/sycl.hpp"

using namespace sycl;

//alias template
template <typename...>
using int128alias_t = __uint128_t;

//templated return type
template <typename T>
T bar() { return T(); };

//typedef
typedef __float128 trickyFloatType;

//struct
struct Mesh {
  __int128 prohib; //#struct_member
};

int main() {
  accessor<int, 1, access::mode::read_write> ok_acc;
  // -- accessors using prohibited types
  accessor<__float128, 1, access::mode::read_write> f128_acc;
  accessor<__int128, 1, access::mode::read_write> i128_acc;
  accessor<long double, 1, access::mode::read_write> ld_acc;
  // -- pointers, aliases, auto, typedef, decltype of prohibited type
  accessor<__int128 *, 1, access::mode::read_write> i128Ptr_acc;
  accessor<int128alias_t<int>, 1, access::mode::read_write> aliased_acc;
  accessor<trickyFloatType, 1, access::mode::read_write> typedef_acc;
  auto V = bar<__int128>();
  accessor<decltype(V), 1, access::mode::read_write> declty_acc;
  // -- Accessor of struct that contains a prohibited type.
  accessor<Mesh, 1, access::mode::read_write> struct_acc;

  kernel_single_task<class use_local>(
      [=]() {
        ok_acc.use();

        // -- accessors using prohibited types
        // expected-error@+1 {{'__float128' is not supported on this target}}
        f128_acc.use();
        // expected-error@+1 {{'__int128' is not supported on this target}}
        i128_acc.use();
        // expected-error@+1 {{'long double' is not supported on this target}}
        ld_acc.use();

        // -- pointers, aliases, auto, typedef, decltype of prohibited type
        // expected-error@+1 {{'__int128' is not supported on this target}}
        i128Ptr_acc.use();
        // expected-error@+1 {{'unsigned __int128' is not supported on this target}}
        aliased_acc.use();
        // expected-error@+1 {{'__float128' is not supported on this target}}
        typedef_acc.use();
        // expected-error@+1 {{'__int128' is not supported on this target}}
        declty_acc.use();

        // -- Accessor of struct that contains a prohibited type.
        // expected-error@#struct_member {{'__int128' is not supported on this target}}
        // expected-note@+1 {{used here}}
        struct_acc.use();
      });
}
