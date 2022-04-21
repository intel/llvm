// RUN: %clang_cc1  -internal-isystem %S/Inputs -sycl-std=2020 -triple spir64 -fsycl-is-device \
// RUN:  -aux-triple x86_64-unknown-linux-gnu \
// RUN:  -verify -fsyntax-only  %s
// RUN: %clang_cc1  -internal-isystem %S/Inputs -sycl-std=2020 -triple spir64 -fsycl-is-device \
// RUN:  -aux-triple x86_64-pc-windows-msvc   \
// RUN:  -verify -fsyntax-only  %s
//
// Ensure that the SYCL diagnostics that are typically deferred are correctly emitted.

#include "sycl.hpp"

sycl::queue deviceQueue;

namespace std {
class type_info;
typedef __typeof__(sizeof(int)) size_t;
} // namespace std

//variadic functions from SYCL kernels emit a deferred diagnostic
void variadic(int, ...) {}

// there are more types like this checked in sycl-restrict.cpp
int calledFromKernel(int a) {
  // expected-error@+1 {{zero-length arrays are not permitted in SYCL device code}}
  int MalArray[0];

  // expected-error@+1 {{'__float128' is not supported on this target}}
  __float128 malFloat = 40;

  //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
  variadic(5);

  return a + 20;
}

// defines (early and late)
#define floatDef __float128
#define int128Def __int128
#define int128tDef __int128_t
#define intDef int

//typedefs (late )
typedef const __uint128_t megeType;
typedef const __float128 trickyFloatType;
typedef const __int128 tricky128Type;

// templated type (late)
//  expected-note@+6 2{{'bar<const unsigned __int128>' defined here}}
//  expected-note@+5 2{{'bar<const __int128>' defined here}}
//  expected-note@+4 4{{'bar<__int128>' defined here}}
//  expected-note@+3 2{{'bar<const __float128>' defined here}}
//  expected-note@+2 2{{'bar<__float128>' defined here}}
template <typename T>
T bar() { return T(); }; //#TemplatedType

//false positive. early incorrectly catches
template <typename t>
void foo(){};

//  template used to specialize a function that contains a lambda that should
//  result in a deferred diagnostic being emitted.

template <typename T>
void setup_sycl_operation(const T VA[]) {

  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<AName, (lambda}}
    h.single_task<class AName>([]() {
      // ======= Zero Length Arrays Not Allowed in Kernel ==========
      // expected-error@+1 {{zero-length arrays are not permitted in SYCL device code}}
      int MalArray[0];
      // expected-error@+1 {{zero-length arrays are not permitted in SYCL device code}}
      intDef MalArrayDef[0];
      // ---- false positive tests. These should not generate any errors.
      foo<int[0]>();
      std::size_t arrSz = sizeof(int[0]);

      // ======= Float128 Not Allowed in Kernel ==========
      // expected-note@+2 {{'malFloat' defined here}}
      // expected-error@+1 {{'__float128' is not supported on this target}}
      __float128 malFloat = 40;
      // expected-error@+1 {{'__float128' is not supported on this target}}
      trickyFloatType malFloatTrick = 41;
      // expected-error@+1 {{'__float128' is not supported on this target}}
      floatDef malFloatDef = 44;
      // expected-error@+2 {{'malFloat' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'__float128' is not supported on this target}}
      auto whatFloat = malFloat;
      // expected-error@#TemplatedType {{'bar<__float128>' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
      // expected-note@+3 {{called by 'operator()'}}
      // expected-error@+2 {{'bar<__float128>' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'__float128' is not supported on this target}}
      auto malAutoTemp5 = bar<__float128>();
      // expected-error@#TemplatedType {{'bar<const __float128>' requires 128 bit size 'const __float128' type support, but target 'spir64' does not support it}}
      // expected-note@+3 {{called by 'operator()'}}
      // expected-error@+2 {{'bar<const __float128>' requires 128 bit size 'const __float128' type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'__float128' is not supported on this target}}
      auto malAutoTemp6 = bar<trickyFloatType>();
      // expected-error@+1 {{'__float128' is not supported on this target}}
      decltype(malFloat) malDeclFloat = 42;
      // ---- false positive tests
      std::size_t someSz = sizeof(__float128);
      foo<__float128>();

      // ======= __int128 Not Allowed in Kernel ==========
      // expected-note@+2 {{'malIntent' defined here}}
      // expected-error@+1 {{'__int128' is not supported on this target}}
      __int128 malIntent = 2;
      // expected-error@+1 {{'__int128' is not supported on this target}}
      tricky128Type mal128Trick = 2;
      // expected-error@+1 {{'__int128' is not supported on this target}}
      int128Def malIntDef = 9;
      // expected-error@+2 {{'malIntent' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'__int128' is not supported on this target}}
      auto whatInt128 = malIntent;
      // expected-error@#TemplatedType {{'bar<__int128>' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
      // expected-note@+3 {{called by 'operator()'}}
      // expected-error@+2 {{'bar<__int128>' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'__int128' is not supported on this target}}
      auto malAutoTemp = bar<__int128>();
      // expected-error@#TemplatedType {{'bar<const __int128>' requires 128 bit size 'const __int128' type support, but target 'spir64' does not support it}}
      // expected-note@+3 {{called by 'operator()'}}
      // expected-error@+2 {{'bar<const __int128>' requires 128 bit size 'const __int128' type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'__int128' is not supported on this target}}
      auto malAutoTemp2 = bar<tricky128Type>();
      // expected-error@+1 {{'__int128' is not supported on this target}}
      decltype(malIntent) malDeclInt = 2;

      // expected-error@+1 {{'__int128' is not supported on this target}}
      __int128_t malInt128 = 2;
      // expected-note@+2 {{'malUInt128' defined here}}
      // expected-error@+1 {{'unsigned __int128' is not supported on this target}}
      __uint128_t malUInt128 = 3;
      // expected-error@+1 {{'unsigned __int128' is not supported on this target}}
      megeType malTypeDefTrick = 4;
      // expected-error@+1 {{'__int128' is not supported on this target}}
      int128tDef malInt2Def = 6;
      // expected-error@+2 {{'malUInt128' requires 128 bit size '__uint128_t' (aka 'unsigned __int128') type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'unsigned __int128' is not supported on this target}}
      auto whatUInt = malUInt128;
      // expected-error@#TemplatedType {{'bar<__int128>' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
      // expected-note@+3 {{called by 'operator()'}}
      // expected-error@+2 {{'bar<__int128>' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'__int128' is not supported on this target}}
      auto malAutoTemp3 = bar<__int128_t>();
      // expected-error@#TemplatedType {{'bar<const unsigned __int128>' requires 128 bit size 'const unsigned __int128' type support, but target 'spir64' does not support it}}
      // expected-note@+3 {{called by 'operator()'}}
      // expected-error@+2 {{'bar<const unsigned __int128>' requires 128 bit size 'const unsigned __int128' type support, but target 'spir64' does not support it}}
      // expected-error@+1 {{'unsigned __int128' is not supported on this target}}
      auto malAutoTemp4 = bar<megeType>();
      // expected-error@+1 {{'__int128' is not supported on this target}}
      decltype(malInt128) malDeclInt128 = 5;

      // ---- false positive tests These should not generate any errors.
      std::size_t i128Sz = sizeof(__int128);
      foo<__int128>();
      std::size_t u128Sz = sizeof(__uint128_t);
      foo<__int128_t>();

      // ========= variadic
      //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
      variadic(5);
    });
  });
}

int main(int argc, char **argv) {

  // --- direct lambda testing ---
  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall 8 {{called by 'kernel_single_task<AName, (lambda}}
    h.single_task<class AName>([]() {
      // expected-error@+1 {{zero-length arrays are not permitted in SYCL device code}}
      int BadArray[0];

      // expected-error@+1 {{'__float128' is not supported on this target}}
      __float128 badFloat = 40; // this SHOULD  trigger a diagnostic

      //expected-error@+1 {{SYCL kernel cannot call a variadic function}}
      variadic(5);

      // expected-note@+1 {{called by 'operator()'}}
      calledFromKernel(10);
    });
  });

  // --- lambda in specialized function testing ---

  //array A is only used to feed the template
  const int array_size = 4;
  int A[array_size] = {1, 2, 3, 4};
  setup_sycl_operation(A);

  return 0;
}
