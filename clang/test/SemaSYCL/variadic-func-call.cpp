// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsyntax-only -Wno-sycl-2017-compat -verify %s

void variadic(int, ...) {}
namespace NS {
void variadic(int, ...) {}
}

struct S {
  S(int, ...) {}
  void operator()(int, ...) {}
};

void foo() {
  auto x = [](int, ...) {};
  x(5, 10); //expected-error{{SYCL kernel cannot call a variadic function}}
}

void overloaded(int, int) {}
void overloaded(int, ...) {}
template <typename, typename Func>
__attribute__((sycl_kernel)) void task(const Func &KF) {
  KF(); // expected-note 2 {{called by 'task}}
}

int main() {
  task<class FK>([]() {
    variadic(5);        //expected-error{{SYCL kernel cannot call a variadic function}}
    variadic(5, 2);     //expected-error{{SYCL kernel cannot call a variadic function}}
    NS::variadic(5, 3); //expected-error{{SYCL kernel cannot call a variadic function}}
    S s(5, 4);          //expected-error{{SYCL kernel cannot call a variadic function}}
    S s2(5);            //expected-error{{SYCL kernel cannot call a variadic function}}
    s(5, 5);            //expected-error{{SYCL kernel cannot call a variadic function}}
    s2(5);              //expected-error{{SYCL kernel cannot call a variadic function}}
    foo();              //expected-note{{called by 'operator()'}}
    overloaded(5, 6);   //expected-no-error
    overloaded(5, s);   //expected-error{{SYCL kernel cannot call a variadic function}}
    overloaded(5);      //expected-error{{SYCL kernel cannot call a variadic function}}
  });
}
