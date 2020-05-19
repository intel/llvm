// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl -fsycl-is-device -aux-triple x86_64-unknown-linux-gnu -verify -fsyntax-only  %s
// 
//
/*
 This test is to verify that deferred diagnostics are emitted whenever there is an AUX target builtin function inside device code.
 x86_64 is the AUX target. Spir64 is the device target.
 _mm_prefetch is the AUX target builtin.
*/

// Testing that the deferred diagnostics work in conjunction with the SYCL namespaces.
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

  //This is host code. This will not be compiled for the device.
  /* 
  _mm_prefetch is an x86-64 target architecture built-in function.
  Its parameters are char const* p, int i
  The valid values for "i" are 0 to 7 :
  #define _MM_HINT_T0 1
  #define _MM_HINT_T1 2
  #define _MM_HINT_T2 3
  #define _MM_HINT_NTA 0
  #define _MM_HINT_ENTA 4
  #define _MM_HINT_ET0 5
  #define _MM_HINT_ET1 6
  #define _MM_HINT_ET2 7
  */
  _mm_prefetch("test", 4); // no error thrown, since this is a valid invocation

  _mm_prefetch("test", 8);// expected-error {{argument value 8 is outside the valid range [0, 7]}}
  
  cl::sycl::kernel_single_task<class AName>([]() {
    //SYCL device compiler will compile this kernel for a device as well as any functions that the kernel calls
    _mm_prefetch("test", 4); // expected-error {{AUX target specific builtins should not be present in device code}}
  });
  return 0;
}
