// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -o - %s | FileCheck %s --check-prefix=DEVICE

// SYCL_EXT_ONEAPI_KERNEL_FUNCTION names the launch arguments twice: in the call
// that selects the kernel, and as the launch arguments. The selecting call is
// not emitted, so each argument is evaluated once, and the launch targets the
// kernel the device side emits.

#define ND_KERNEL                                                              \
  [[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]   \
  __attribute__((sycl_device))

template <auto *Func> struct kernel_function_s {};
template <auto *Func> constexpr kernel_function_s<Func> kernel_function{};
template <auto *Func, typename... Args>
void nd_launch(kernel_function_s<Func>, Args &&...) {}

#define SYCL_EXT_ONEAPI_KERNEL_FUNCTION(NAME, ...)                             \
  kernel_function<__builtin_sycl_launch_kernel(NAME(__VA_ARGS__))>,            \
      ##__VA_ARGS__

template <typename T> ND_KERNEL void axpy(T *y, T a) {}
ND_KERNEL void ovl(int *p) {}
ND_KERNEL void ovl(float *p) {}

float next();

// HOST-LABEL: define {{.*}} @_Z9eval_oncePf(
// HOST-NOT: @_Z4nextv
// HOST: getelementptr inbounds {{.*}} float, ptr %{{.*}}, i32 1
// HOST-NOT: getelementptr
// HOST: call {{.*}} @_Z4nextv()
// HOST-NOT: @_Z4nextv
// HOST-NOT: @_Z4axpyIfEvPT_S0_(
// HOST: call void @_Z9nd_launch{{.*}}axpyIf{{.*}}(
// HOST-NEXT: ret void
void eval_once(float *p) {
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(axpy, p++, next()));
}

// HOST-LABEL: define {{.*}} @_Z8overloadPi(
// HOST-NOT: @_Z3ovlPi(
// HOST: call void @_Z9nd_launch{{.*}}ovlPi{{.*}}(
void overload(int *ip) {
  nd_launch(SYCL_EXT_ONEAPI_KERNEL_FUNCTION(ovl, ip));
}

// DEVICE-DAG: define {{.*}} spir_kernel void @_Z18__sycl_kernel_axpyIfEvPT_S0_(
// DEVICE-DAG: define {{.*}} spir_kernel void @_Z17__sycl_kernel_ovlPi(
