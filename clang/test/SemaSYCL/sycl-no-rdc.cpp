// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -fsycl-allow-func-ptr -fno-gpu-rdc %s
SYCL_EXTERNAL void syclExternal();

SYCL_EXTERNAL void notSyclExternal() {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel]] void kernel_single_task(const KernelType& kernelFunc) { // #kernelSingleTask
  kernelFunc();
}

void kernel() {
  // expected-error@+1{{seperate compilation unit without relocatable device code}}
  syclExternal();
  notSyclExternal();
  // expected-error@+1{{seperate compilation unit without relocatable device code}}
  auto fcnPtr = 1 == 0 ? syclExternal : notSyclExternal;
  fcnPtr();
  // expected-error@+1{{seperate compilation unit without relocatable device code}}
  constexpr auto constExprFcnPtr = 1 == 0 ? syclExternal : notSyclExternal;
  constExprFcnPtr();
}

void callKernel() {
  kernel_single_task<class Kernel>([]() {kernel();});
}
