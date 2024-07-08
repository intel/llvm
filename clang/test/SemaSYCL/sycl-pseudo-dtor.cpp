// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

template <typename functor_t>
struct functor_wrapper{
  functor_t f;

  auto operator()() -> void {
    return;
  };
};

struct S { virtual void foo(); };
struct T { virtual ~T(); };

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  // expected-no-note@+1
  using DATA_I = int;
  using DATA_S = S;
  using DATA_T = T;
  // this expression should be okay
  auto functor = [](DATA_I & v1, DATA_S &v2, DATA_T& v3) {
    // expected-no-error@+1
    v1.~DATA_I();
    v2.~DATA_S();
    // expected-error@+1{{SYCL kernel cannot call a virtual function}}
    v3.~DATA_T();
  };
  auto wrapped_functor = functor_wrapper<decltype(functor)>{functor};
  wrapped_functor();
}

int main() {
  kernel_single_task<class fake_kernel>([]() { });
  return 0;
}
