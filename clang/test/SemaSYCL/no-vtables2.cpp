// RUN: %clang_cc1 -fsycl-is-device -Wno-return-type -verify -Wno-sycl-2017-compat -fsyntax-only %s

struct Base {
  virtual void f() const {}
};

struct Inherit : Base {
  virtual void f() const override {}
};

Inherit always_uses() {
  Inherit u;
}

static constexpr Inherit IH;

Inherit *usage_child(){}

Inherit usage() {
  Inherit u;
  Inherit *u_ptr;

  using foo = Inherit;
  typedef Inherit bar;
  // expected-error@+1 {{SYCL kernel cannot call a virtual function}}
  IH.f();

  usage_child();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}
int main() {
  kernel_single_task<class fake_kernel>([]() { usage(); });
  return 0;
}

