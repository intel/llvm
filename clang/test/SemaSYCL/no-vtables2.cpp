// RUN: %clang_cc1 -fsycl-is-device -Wno-return-type -verify -fsyntax-only -x c++ -emit-llvm-only %s

struct Base {
  virtual void f() const {}
};

// expected-error@+1 9{{SYCL kernel cannot have a class with a virtual function table}}
struct Inherit : Base {
  virtual void f() const override {}
};

Inherit always_uses() {
  Inherit u;
}

static constexpr Inherit IH;

// expected-note@+1{{used here}}
Inherit *usage_child(){}

  // expected-note@+1{{used here}}
Inherit usage() {
  // expected-note@+1{{used here}}
  Inherit u;
  // expected-note@+1{{used here}}
  Inherit *u_ptr;

  // expected-note@+1{{used here}}
  using foo = Inherit;
  // expected-note@+1{{used here}}
  typedef Inherit bar;
  // expected-error@+2 {{SYCL kernel cannot call a virtual function}}
  // expected-note@+1{{used here}}
  IH.f();

  // expected-note@+1{{used here}}
  usage_child();
}


template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}
int main() {
  // expected-note@+1{{used here}}
  kernel_single_task<class fake_kernel>([]() { usage(); });
  return 0;
}

