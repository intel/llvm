// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -verify -fsyntax-only %s

void eh_ok(void)
{
  __float128 A;
}

void usage() {
  // expected-error@+1 {{__float128 is not supported on this target}}
  __float128 A;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
// expected-note@+1{{called by 'kernel_single_task}}
  kernelFunc();
}

int main() {
// expected-note@+1{{called by 'operator()'}}
  kernel_single_task<class fake_kernel>([]() { usage(); });
  return 0;
}

