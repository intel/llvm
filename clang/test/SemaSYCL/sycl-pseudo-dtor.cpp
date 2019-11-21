// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -x c++ %s

template <typename functor_t>
struct functor_wrapper{
  functor_t f;

  auto operator()() -> void {
    return;
  };
};

// expected-error@+1 2{{SYCL kernel cannot have a class with a virtual function table}}
struct S { virtual void foo(); };
// expected-error@+1 2{{SYCL kernel cannot have a class with a virtual function table}}
struct T { virtual ~T(); };

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  // expected-no-note@+1
  using DATA_I = int;
  // expected-note@+1{{used here}}
  using DATA_S = S;
  // expected-note@+1{{used here}}
  using DATA_T = T;
  // this expression should be okay
  auto functor = [](DATA_I & v1, DATA_S &v2, DATA_T& v3) {
    // expected-no-error@+1
    v1.~DATA_I();
    // expected-note@+1{{used here}}
    v2.~DATA_S();
    // expected-error@+2{{SYCL kernel cannot call a virtual function}}
    // expected-note@+1{{used here}}
    v3.~DATA_T();
  };
  auto wrapped_functor = functor_wrapper<decltype(functor)>{functor};
  wrapped_functor();
}

int main() {
  kernel_single_task<class fake_kernel>([]() { });
  return 0;
}
