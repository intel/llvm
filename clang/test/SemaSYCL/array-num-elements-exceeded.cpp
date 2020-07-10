// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir-unknown -verify -fsyntax-only %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown -verify -fsyntax-only %s

const double big[67000] = {1.0, 2.0, 3.0};
const int zero_init_array[67000] = {};

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc(); // expected-note {{called by 'kernel_single_task<fake_kernel, (lambda}}
}

int main() {
  double hostResult = big[0]; // no error thrown, since accessing global const arrays with size > 65532 is valid from host code.
  kernel_single_task<class fake_kernel>([]() {
    int a = zero_init_array[0]; // no error thrown, since accessing zero-initialized arrays with size > 65532 is valid.
    double kernelGlobal = big[0]; // expected-error {{Due to SPIR-V intermediate format limitations, constant arrays with number of elements more than 65532 cannot be used in SYCL.You can workaround this limitation by splitting the array into several ones}}
  });
  return 0;
}
