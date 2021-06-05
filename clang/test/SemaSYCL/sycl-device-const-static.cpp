// RUN: %clang_cc1 -fsycl-is-device -verify -Wno-sycl-2017-compat -fsyntax-only %s

struct Base {};
struct S {
  void foo() {}
  S() {}
};

struct T {
  const static S s1;
};
const S T::s1;

template <typename T>
struct U {
  static const S s2;
};
template <class T>
const S U<T>::s2;

template struct U<Base>;

const S s5;

void usage() {
  // expected-error@+1{{SYCL kernel cannot use a non-const static data variable}}
  static int s1;
  const static int cs = 0;
  constexpr static int ces = 0;
  static const S s6;
  // expected-error@+1{{SYCL kernel cannot use a const static or global variable that is neither zero-initialized nor constant-initialized}}
  (void)T::s1;
  // expected-error@+1{{SYCL kernel cannot use a const static or global variable that is neither zero-initialized nor constant-initialized}}
  (void)s5;
  // expected-error@+1{{SYCL kernel cannot use a const static or global variable that is neither zero-initialized nor constant-initialized}}
  (void)s6;
}

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  // expected-error@+1{{SYCL kernel cannot use a non-const static data variable}}
  static int z;
  // expected-note-re@+3{{called by 'kernel_single_task<fake_kernel, (lambda at {{.*}})>}}
  // expected-note-re@+2{{called by 'kernel_single_task<fake_kernel, (lambda at {{.*}})>}}
  // expected-error@+1{{SYCL kernel cannot use a const static or global variable that is neither zero-initialized nor constant-initialized}}
  kernelFunc(U<Base>::s2);
}

int main() {
  static int s2;
  kernel_single_task<class fake_kernel>([](S s4) {
    //  expected-note@+1{{called by 'operator()'}}
    usage();
    s4.foo();
    // expected-error@+1{{SYCL kernel cannot use a non-const static data variable}}
    static int s3;
  });

  return 0;
}
