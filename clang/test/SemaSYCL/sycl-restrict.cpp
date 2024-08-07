// RUN: %clang_cc1 -fsycl-is-device -fcxx-exceptions -triple spir64 \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -Wno-return-type -verify     \
// RUN:  -fsyntax-only -std=c++17 %s
// RUN: %clang_cc1 -fsycl-is-device -fcxx-exceptions -triple spir64 \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -fno-sycl-allow-func-ptr     \
// RUN:  -Wno-return-type -verify -fsyntax-only      \
// RUN:  -std=c++17 %s
// RUN: %clang_cc1 -fsycl-is-device -fcxx-exceptions -triple spir64 \
// RUN:  -aux-triple x86_64-unknown-linux-gnu -DALLOW_FP=1                 \
// RUN:  -fsycl-allow-func-ptr -Wno-return-type -verify                    \
// RUN:  -fsyntax-only -std=c++17 %s

namespace std {
class type_info;
typedef __typeof__(sizeof(int)) size_t;
} // namespace std

// we're testing a restricted mode, thus just provide a stub implementation for
// function with address-space-unspecified pointers.
void *operator new(std::size_t) {
  return reinterpret_cast<void *>(1);
}

namespace Check_User_Operators {
class Fraction {
  // expected-error@+2 {{SYCL kernel cannot call a recursive function}}
  // expected-note@+1 {{function implemented using recursion declared here}}
  int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
  int n, d;

public:
  Fraction(int n, int d = 1) : n(n / gcd(n, d)), d(d / gcd(n, d)) {}
  int num() const { return n; }
  int den() const { return d; }
};
bool operator==(const Fraction &lhs, const Fraction &rhs) {
  new int; // expected-error {{SYCL kernel cannot allocate storage}}
  return lhs.num() == rhs.num() && lhs.den() == rhs.den();
}
} // namespace Check_User_Operators

namespace Check_VLA_Restriction {
void no_restriction(int p) { // expected-note {{declared here}}
  // expected-note@+2 {{function parameter 'p' with unknown value cannot be used in a constant expression}}
  // expected-warning@+1 {{variable length arrays in C++ are a Clang extension}}
  int index[p + 2];
}
void restriction(int p) { // expected-note {{declared here}}
  // This particular violation is nested under two kernels with intermediate function calls.
  // e.g. main -> 1stkernel -> usage -> 2ndkernel -> isa_B -> restriction -> !!
  // Because the error is in two different kernels, we are given helpful notes for the origination of the error, twice.
  // expected-note@#call_usage {{called by 'operator()'}}
  // expected-note@#call_kernelFunc {{called by 'kernel_single_task<fake_kernel, (lambda at}}
  // expected-note@#call_isa_B 2{{called by 'operator()'}}
  // expected-note@#rtti_kernel 2{{called by 'kernel1<kernel_name, (lambda at }}
  // expected-note@#call_vla {{called by 'isa_B'}}
  // expected-note@+2 {{function parameter 'p' with unknown value cannot be used in a constant expression}}
  // expected-warning@+1 {{variable length arrays in C++ are a Clang extension}}
  int index[p + 2]; // expected-error {{variable length arrays are not supported for the current target}}
}
} // namespace Check_VLA_Restriction

void *operator new(std::size_t size, void *ptr) throw() { return ptr; };
namespace Check_RTTI_Restriction {
struct A {
  virtual ~A(){};
};

struct B : public A {
  B() : A() {}
};

struct OverloadedNewDelete {
  // This overload allocates storage, give diagnostic.
  void *operator new(std::size_t size) throw() {
    float *pt = new float; // expected-error {{SYCL kernel cannot allocate storage}}
    return 0;
  }
  // This overload does not allocate: no diagnostic.
  void *operator new[](std::size_t size) throw() { return 0; }
  void operator delete(void *){};
  void operator delete[](void *){};
};

bool isa_B(A *a) {
  Check_User_Operators::Fraction f1(3, 8), f2(1, 2), f3(10, 2);
  if (f1 == f2) // expected-note {{called by 'isa_B'}}
    return false;

  Check_VLA_Restriction::restriction(7); //#call_vla
  int *ip = new int;                     // expected-error {{SYCL kernel cannot allocate storage}}
  int i;
  int *p3 = new (&i) int;                                    // no error on placement new
  OverloadedNewDelete *x = new (struct OverloadedNewDelete); // expected-note {{called by 'isa_B'}}
  auto y = new struct OverloadedNewDelete[5];
  (void)typeid(int);                // expected-error {{SYCL kernel cannot use rtti}}
  return dynamic_cast<B *>(a) != 0; // expected-error {{SYCL kernel cannot use rtti}}
}

template <typename N, typename L>
__attribute__((sycl_kernel)) void kernel1(const L &l) {
  l(); //#rtti_kernel  // expected-note 2{{called by 'kernel1<kernel_name, (lambda at }}
}
} // namespace Check_RTTI_Restriction

typedef struct A {
  static int stat_member;
  const static int const_stat_member;
  constexpr static int constexpr_stat_member = 0;

  int fm(void) {
    return stat_member; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
  }
} a_type;

using myFuncDef = int(int, int);

// defines (early and late)
#define floatDef __float128
#define longdoubleDef long double
#define int128Def __int128
#define int128tDef __int128_t
#define intDef int

//typedefs (late )
typedef __uint128_t megeType;
typedef __float128 trickyFloatType;
typedef __int128 tricky128Type;
typedef long double trickyLDType;

// templated return type
//  expected-note@+5 4{{'bar<long double>' defined here}}
//  expected-note@+4 2{{'bar<unsigned __int128>' defined here}}
//  expected-note@+3 6{{'bar<__int128>' defined here}}
//  expected-note@+2 4{{'bar<__float128>' defined here}}
template <typename T>
T bar() { return T(); }; //#TemplatedType

//variable template
// expected-note@+5 2{{'solutionToEverything<long double>' defined here}}
// expected-note@+4 {{solutionToEverything<unsigned __int128>' defined here}}
// expected-note@+3 3{{solutionToEverything<__int128>' defined here}}
// expected-note@+2 2{{solutionToEverything<__float128>' defined here}}
template <class T>
constexpr T solutionToEverything = T(42);

//alias template
template <typename...>
using floatalias_t = __float128;

//alias template
template <typename...>
using int128alias_t = __int128;

//alias template
template <typename...>
using ldalias_t = long double;

//false positive. early incorrectly catches
template <typename t>
void foo(){};
//false positive template alias
template <typename...>
using safealias_t = int;

//struct
struct frankenStruct {
  // expected-error@+1 {{zero-length arrays are not permitted in SYCL device code}}
  int mosterArr[0];
  // expected-error@+1 {{'__float128' is not supported on this target}}
  __float128 scaryQuad;
  // expected-error@+1 {{'__int128' is not supported on this target}}
  __int128 frightenInt;
  // expected-error@+1 {{'long double' is not supported on this target}}
  long double terrorLD;
};

//struct
struct trickyStruct {
  // expected-error@+1 {{'__float128' is not supported on this target}}
  trickyFloatType trickySructQuad;
  // expected-error@+1 {{'__int128' is not supported on this target}}
  tricky128Type trickyStructInt;
  // expected-error@+1 {{'long double' is not supported on this target}}
  trickyLDType trickyStructLD;
};

// function return type and argument both unsupported
// expected-note@+1 2{{'commitInfraction' defined here}}
[[intel::device_indirectly_callable]] __int128 commitInfraction(__int128 a) {
  return 0;
}

void eh_ok(void) {
  __float128 A;
  try {
    ;
  } catch (...) {
    ;
  }
  throw 20;
}

void eh_not_ok(void) {
  try { // expected-error {{SYCL kernel cannot use exceptions}}
    ;
  } catch (...) {
    ;
  }
  throw 20; // expected-error {{SYCL kernel cannot use exceptions}}
}

void usage(myFuncDef functionPtr) {
  eh_not_ok(); // expected-note {{called by 'usage'}}

#if ALLOW_FP
  // No error message for function pointer.
#else
  // expected-error@+2 {{SYCL kernel cannot call through a function pointer}}
#endif
  if ((*functionPtr)(1, 2))
    /* no-op */;

  Check_RTTI_Restriction::kernel1<class kernel_name>([]() { //#call_rtti_kernel
    Check_RTTI_Restriction::A *a;
    Check_RTTI_Restriction::isa_B(a); //#call_isa_B  // expected-note 2{{called by 'operator()'}}
  });

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
  // expected-note@+3 {{called by 'usage'}}
  // expected-error@+2 {{'bar<__float128>' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__float128' is not supported on this target}}
  auto malAutoTemp5 = bar<__float128>();
  // expected-error@#TemplatedType {{'bar<__float128>' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
  // expected-note@+3 {{called by 'usage'}}
  // expected-error@+2 {{'bar<__float128>' requires 128 bit size '__float128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__float128' is not supported on this target}}
  auto malAutoTemp6 = bar<trickyFloatType>();
  // expected-error@+1 {{'__float128' is not supported on this target}}
  decltype(malFloat) malDeclFloat = 42;
  // expected-error@+2 {{'solutionToEverything<__float128>' requires 128 bit size 'const __float128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__float128' is not supported on this target}}
  auto malFloatTemplateVar = solutionToEverything<__float128>;
  // expected-error@+2 {{'solutionToEverything<__float128>' requires 128 bit size 'const __float128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__float128' is not supported on this target}}
  auto malTrifectaFloat = solutionToEverything<trickyFloatType>;
  // expected-error@+1 {{'__float128' is not supported on this target}}
  floatalias_t<void> aliasedFloat = 42;
  // ---- false positive tests
  std::size_t someSz = sizeof(__float128);
  foo<__float128>();
  safealias_t<__float128> notAFloat = 3;

  // ======= long double Not Allowed in Kernel ==========
  // expected-note@+2 {{'malLD' defined here}}
  // expected-error@+1 {{'long double' is not supported on this target}}
  long double malLD = 50;
  // expected-error@+1 {{'long double' is not supported on this target}}
  trickyLDType malLDTrick = 51;
  // expected-error@+1 {{'long double' is not supported on this target}}
  longdoubleDef malLDDef = 52;
  // expected-error@+2 {{'malLD' requires 128 bit size 'long double' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'long double' is not supported on this target}}
  auto whatLD = malLD;
  // expected-error@#TemplatedType {{'bar<long double>' requires 128 bit size 'long double' type support, but target 'spir64' does not support it}}
  // expected-note@+3 {{called by 'usage'}}
  // expected-error@+2 {{'bar<long double>' requires 128 bit size 'long double' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'long double' is not supported on this target}}
  auto malAutoLD = bar<long double>();
  // expected-error@#TemplatedType {{'bar<long double>' requires 128 bit size 'long double' type support, but target 'spir64' does not support it}}
  // expected-note@+3 {{called by 'usage'}}
  // expected-error@+2{{'bar<long double>' requires 128 bit size 'long double' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'long double' is not supported on this target}}
  auto malAutoLD2 = bar<trickyLDType>();
  // expected-error@+1 {{'long double' is not supported on this target}}
  decltype(malLD) malDeclLD = 53;
  // expected-error@+2 {{'solutionToEverything<long double>' requires 128 bit size 'const long double' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'long double' is not supported on this target}}
  auto malLDTemplateVar = solutionToEverything<long double>;
  // expected-error@+2 {{'solutionToEverything<long double>' requires 128 bit size 'const long double' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'long double' is not supported on this target}}
  auto malTrifectaLD = solutionToEverything<trickyLDType>;
  // expected-error@+1 {{'long double' is not supported on this target}}
  ldalias_t<void> aliasedLongDouble = 54;
  // ---- false positive tests
  std::size_t someLDSz = sizeof(long double);
  foo<long double>();
  safealias_t<long double> notALD = 55;

  // ======= Zero Length Arrays Not Allowed in Kernel ==========
  // expected-error@+1 {{zero-length arrays are not permitted in SYCL device code}}
  int MalArray[0];
  // expected-error@+1 {{zero-length arrays are not permitted in SYCL device code}}
  intDef MalArrayDef[0];
  // ---- false positive tests. These should not generate any errors.
  foo<int[0]>();
  std::size_t arrSz = sizeof(int[0]);

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
  // expected-note@+3 {{called by 'usage'}}
  // expected-error@+2 {{'bar<__int128>' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__int128' is not supported on this target}}
  auto malAutoTemp = bar<__int128>();
  // expected-error@#TemplatedType {{'bar<__int128>' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  // expected-note@+3 {{called by 'usage'}}
  // expected-error@+2 {{'bar<__int128>' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__int128' is not supported on this target}}
  auto malAutoTemp2 = bar<tricky128Type>();
  // expected-error@+1 {{'__int128' is not supported on this target}}
  decltype(malIntent) malDeclInt = 2;
  // expected-error@+2 {{'solutionToEverything<__int128>' requires 128 bit size 'const __int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__int128' is not supported on this target}}
  auto mal128TemplateVar = solutionToEverything<__int128>;
  // expected-error@+2 {{'solutionToEverything<__int128>' requires 128 bit size 'const __int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__int128' is not supported on this target}}
  auto malTrifecta128 = solutionToEverything<tricky128Type>;
  // expected-error@+1 {{'__int128' is not supported on this target}}
  int128alias_t<void> aliasedInt128 = 79;

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
  // expected-note@+3 {{called by 'usage'}}
  // expected-error@+2 {{'bar<__int128>' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__int128' is not supported on this target}}
  auto malAutoTemp3 = bar<__int128_t>();
  // expected-error@#TemplatedType {{'bar<unsigned __int128>' requires 128 bit size 'unsigned __int128' type support, but target 'spir64' does not support it}}
  // expected-note@+3 {{called by 'usage'}}
  // expected-error@+2 {{'bar<unsigned __int128>' requires 128 bit size 'unsigned __int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'unsigned __int128' is not supported on this target}}
  auto malAutoTemp4 = bar<megeType>();
  // expected-error@+1 {{'__int128' is not supported on this target}}
  decltype(malInt128) malDeclInt128 = 5;
  // expected-error@+2 {{'solutionToEverything<__int128>' requires 128 bit size 'const __int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'__int128' is not supported on this target}}
  auto mal128TIntTemplateVar = solutionToEverything<__int128_t>;
  // expected-error@+2 {{'solutionToEverything<unsigned __int128>' requires 128 bit size 'const unsigned __int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 {{'unsigned __int128' is not supported on this target}}
  auto malTrifectaInt128T = solutionToEverything<megeType>;

  // ======= Struct Members Checked  =======
  frankenStruct strikesFear; // expected-note 4{{used here}}
  trickyStruct incitesPanic; // expected-note 3{{used here}}

  // ======= Function Prototype Checked  =======
  // expected-error@+2 2{{'commitInfraction' requires 128 bit size '__int128' type support, but target 'spir64' does not support it}}
  // expected-error@+1 2{{'__int128' is not supported on this target}}
  auto notAllowed = &commitInfraction;

  // ---- false positive tests These should not generate any errors.
  std::size_t i128Sz = sizeof(__int128);
  foo<__int128>();
  std::size_t u128Sz = sizeof(__uint128_t);
  foo<__int128_t>();
  safealias_t<__int128> notAnInt128 = 3;
}

namespace ns {
int glob;
}
extern "C++" {
int another_global = 5;
namespace AnotherNS {
int moar_globals = 5;
}
}

template<const auto &T>
int uses_global(){}

[[intel::device_indirectly_callable]] int addInt(int n, int m) {
  return n + m;
}

int use2(a_type ab, a_type *abp) {

  if (ab.constexpr_stat_member)
    return 2;
  if (ab.const_stat_member)
    return 1;
  if (ab.stat_member) // expected-error {{SYCL kernel cannot use a non-const static data variable}}
    return 0;
  if (abp->stat_member) // expected-error {{SYCL kernel cannot use a non-const static data variable}}
    return 0;
  if (ab.fm()) // expected-note {{called by 'use2'}}
    return 0;

  // No error, as this is not in an evaluated context.
  (void)(uses_global<another_global>() + uses_global<ns::glob>());

  return another_global; // expected-error {{SYCL kernel cannot use a non-const global variable}}

  return ns::glob +               // expected-error {{SYCL kernel cannot use a non-const global variable}}
         AnotherNS::moar_globals; // expected-error {{SYCL kernel cannot use a non-const global variable}}
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc(); //#call_kernelFunc // expected-note 11{{called by 'kernel_single_task<fake_kernel, (lambda at}}
}

int main() {
  // Outside Kernel, these should not generate errors.
  a_type ab;

  int PassOver[0];
  __float128 okFloat = 40;
  __int128 fineInt = 20;
  __int128_t acceptable = 30;
  __uint128_t whatever = 50;
  frankenStruct noProblem;
  trickyStruct noTrouble;
  auto notACrime = &commitInfraction;

  kernel_single_task<class fake_kernel>([=]() {
    usage(&addInt); //#call_usage // expected-note 9{{called by 'operator()'}}
    a_type *p;
    use2(ab, p); // expected-note 2{{called by 'operator()'}}
  });
  return 0;
}
