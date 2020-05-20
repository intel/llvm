// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl -fsycl-is-device -aux-triple x86_64-unknown-linux-gnu -verify -fsyntax-only  %s

inline namespace cl {
namespace sycl {

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  // expected-note@+1 {{called by 'kernel_single_task<AName, (lambda}}
  kernelFunc();
}

} // namespace sycl
} // namespace cl

int main(int argc, char **argv) {
  //_mm_prefetch is an x86-64 specific builtin where the second integer parameter is required to be a constant
  //between 0 and 7.
  _mm_prefetch("test", 4); // no error thrown, since this is a valid invocation

  _mm_prefetch("test", 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  cl::sycl::kernel_single_task<class AName>([]() {
    _mm_prefetch("test", 4); // expected-error {{builtin is not supported on this target}}
    _mm_prefetch("test", 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}} expected-error {{builtin is not supported on this target}}
  });
  return 0;
}
