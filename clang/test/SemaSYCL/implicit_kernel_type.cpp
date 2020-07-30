// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsyntax-only -verify %s -Werror=sycl-strict -DERROR
// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsyntax-only -verify %s  -Wsycl-strict -DWARN
// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsycl-unnamed-lambda -fsyntax-only -verify %s  -Werror=sycl-strict

// SYCL 1.2 Definitions
template <typename name, typename Func>
__attribute__((sycl_kernel)) void sycl_121_single_task(Func kernelFunc) {
  kernelFunc();
}

class event {};
class queue {
public:
  template <typename T>
  event submit(T cgf) { return event{}; }
};
class auto_name {};
template <typename Name, typename Type>
struct get_kernel_name_t {
  using name = Name;
};
class handler {
public:
  template <typename KernelName = auto_name, typename KernelType>
  void single_task(KernelType kernelFunc) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    sycl_121_single_task<NameT>(kernelFunc);
#else
    kernelFunc();
#endif
  }
};
// -- /Definitions

#ifdef __SYCL_UNNAMED_LAMBDA__
// expected-no-diagnostics
#endif

//using namespace cl::sycl;

void function() {
}

// user-defined class
struct myWrapper {
};

// user-declared class
class myWrapper2;

int main() {
  queue q;
#ifndef __SYCL_UNNAMED_LAMBDA__
  // expected-note@+1 {{InvalidKernelName1 declared here}}
  class InvalidKernelName1 {};
  q.submit([&](handler &h) {
    // expected-error@+1 {{kernel needs to have a globally-visible name}}
    h.single_task<InvalidKernelName1>([]() {});
  });
#endif
#if defined(WARN)
  // expected-warning@+6 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-note@+5 {{fake_kernel declared here}}
#elif defined(ERROR)
  // expected-error@+3 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-note@+2 {{fake_kernel declared here}}
#endif
  sycl_121_single_task<class fake_kernel>([]() { function(); });
#if defined(WARN)
  // expected-warning@+6 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-note@+5 {{fake_kernel2 declared here}}
#elif defined(ERROR)
  // expected-error@+3 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-note@+2 {{fake_kernel2 declared here}}
#endif
  sycl_121_single_task<class fake_kernel2>([]() {
    auto l = [](auto f) { f(); };
  });
  sycl_121_single_task<class myWrapper>([]() { function(); });
  sycl_121_single_task<class myWrapper2>([]() { function(); });
  return 0;
}
