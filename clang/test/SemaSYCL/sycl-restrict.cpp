// RUN: %clang_cc1 -fsycl -fsycl-is-device -fcxx-exceptions -triple spir64 -Wno-return-type -verify -fsyntax-only -std=c++17 %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -fcxx-exceptions -triple spir64 -fno-sycl-allow-func-ptr -Wno-return-type -verify -fsyntax-only -std=c++17 %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -fcxx-exceptions -triple spir64 -DALLOW_FP=1 -fsycl-allow-func-ptr -Wno-return-type -verify -fsyntax-only -std=c++17 %s

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
  new int; // expected-error 2{{SYCL kernel cannot allocate storage}}
  return lhs.num() == rhs.num() && lhs.den() == rhs.den();
}
} // namespace Check_User_Operators

namespace Check_VLA_Restriction {
void no_restriction(int p) {
  int index[p + 2];
}
void restriction(int p) {
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
    float *pt = new float; // expected-error 2{{SYCL kernel cannot allocate storage}}
    return 0;
  }
  // This overload does not allocate: no diagnostic.
  void *operator new[](std::size_t size) throw() { return 0; }
  void operator delete(void *){};
  void operator delete[](void *){};
};

bool isa_B(A *a) {
  Check_User_Operators::Fraction f1(3, 8), f2(1, 2), f3(10, 2);
  if (f1 == f2) // expected-note 2{{called by 'isa_B'}}
    return false;

  Check_VLA_Restriction::restriction(7);
  int *ip = new int; // expected-error 2{{SYCL kernel cannot allocate storage}}
  int i;
  int *p3 = new (&i) int; // no error on placement new
  OverloadedNewDelete *x = new (struct OverloadedNewDelete); // expected-note 2{{called by 'isa_B'}}
  auto y = new struct OverloadedNewDelete[5];
  (void)typeid(int); // expected-error {{SYCL kernel cannot use rtti}}
  return dynamic_cast<B *>(a) != 0; // expected-error {{SYCL kernel cannot use rtti}}
}

template <typename N, typename L>
__attribute__((sycl_kernel)) void kernel1(L l) {
  l(); // expected-note 6{{called by 'kernel1<kernel_name, (lambda at }}
}
} // namespace Check_RTTI_Restriction

typedef struct Base {
  virtual void f() const {}
} b_type;

typedef struct A {
  static int stat_member;
  const static int const_stat_member;
  constexpr static int constexpr_stat_member = 0;

  int fm(void) {
    return stat_member; // expected-error {{SYCL kernel cannot use a non-const static data variable}}
  }
} a_type;

b_type b;

using myFuncDef = int(int, int);

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
    // expected-error@+1 {{SYCL kernel cannot use a non-const global variable}}
    b.f(); // expected-error {{SYCL kernel cannot call a virtual function}}

  Check_RTTI_Restriction::kernel1<class kernel_name>([]() { // expected-note 3{{called by 'usage'}}
  Check_RTTI_Restriction::A *a;
  Check_RTTI_Restriction::isa_B(a); }); // expected-note 6{{called by 'operator()'}}

  __float128 A; // expected-error {{__float128 is not supported on this target}}

  int BadArray[0]; // expected-error {{zero-length arrays are not permitted in C++}}
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

int addInt(int n, int m) {
  return n + m;
}

int use2(a_type ab, a_type *abp) {

  if (ab.constexpr_stat_member)
    return 2;
  if (ab.const_stat_member)
    return 1;
  if (ab.stat_member)  // expected-error {{SYCL kernel cannot use a non-const static data variable}}
    return 0;
  if (abp->stat_member) // expected-error {{SYCL kernel cannot use a non-const static data variable}}
    return 0;
  if (ab.fm()) // expected-note {{called by 'use2'}}
    return 0;

  return another_global; // expected-error {{SYCL kernel cannot use a non-const global variable}}

  return ns::glob + // expected-error {{SYCL kernel cannot use a non-const global variable}}
         AnotherNS::moar_globals; // expected-error {{SYCL kernel cannot use a non-const global variable}}
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc(); // expected-note 7{{called by 'kernel_single_task<fake_kernel, (lambda at}}
}

int main() {
  a_type ab;
  kernel_single_task<class fake_kernel>([=]() {
    usage(&addInt); // expected-note 5{{called by 'operator()'}}
    a_type *p;
    use2(ab, p); // expected-note 2{{called by 'operator()'}}
  });
  return 0;
}
