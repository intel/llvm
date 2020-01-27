// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify -fsyntax-only %s

template <class T>
class Z {
public:
  T field;
// TODO
// Restriction usages inside class declarations is not diagnosed because in
// this case getCurLexicalContext() returns record decl and we can diagnose only
// inside function decls
  __float128 field1;
};

void eh_ok(void) {
  __float128 A;
  int B = sizeof(__float128);
  Z<__float128> C;
}

template <typename T>
T bar(T t) {
  return t;
}
__float128 foo1() { return 0; }

template<typename t> void foo2(){};

// TODO
// Function parameters and return values is not diagnosed because in this case
// getCurLexicalContext() returns translation unit decl and we can diagnose only
// inside function decls
__float128 foo(__float128 P) {
  // expected-error@+1 {{__float128 is not supported on this target}}
  Z<__float128> A;
  bar(P);
  bar(A);
  // expected-error@+1 {{__float128 is not supported on this target}}
  __float128 B;
  bar(B);
  // expected-error@+1 {{__float128 is not supported on this target}}
  bar((__float128)0);

  // TODO
  // NOT ANALYZED PROBLEM
  // expected-error@+1 {{__float128 is not supported on this target}}
  foo2<__float128>();
  return P;
}

void usage() {
  // expected-error@+1 {{__float128 is not supported on this target}}
  __float128 A;
  // expected-note@+1{{called by 'usage'}}
  auto B = foo(A);
  int E = sizeof(__float128);

  // NOT ANALYZED PROBLEM
  decltype(A) C;
  decltype(foo1()) D;
}

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  // expected-note@+1 2{{called by 'kernel_single_task}}
  kernelFunc();
}

int main() {
  // expected-note@+1 2{{called by 'operator()'}}
  kernel_single_task<class fake_kernel>([]() { usage(); });
  return 0;
}

