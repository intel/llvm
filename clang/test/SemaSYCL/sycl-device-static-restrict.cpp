// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s
const int glob1 = 1;
int glob2 = 2;
template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  // expected-note-re@+1{{called by 'kernel_single_task<fake_kernel, (lambda at {{.*}})>}}
  kernelFunc();
}

int main() {
  static int n = 0;
  const static int l = 0;
  kernel_single_task<class fake_kernel>([]() {
    int m = l;
    m = glob1;
    // expected-error@+1{{SYCL kernel cannot use a non-const static data variable}}
    m = n;
    // expected-error@+1{{SYCL kernel cannot use a non-const global variable}}
    m = glob2;
  });
  return 0;
}
