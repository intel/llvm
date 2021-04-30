// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// CHECK: @[[INT:[^\w]+]] = private unnamed_addr addrspace(1) constant [[INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSi\00"

// Note: the following 2 are the same, and need to remain the same, since they are mangling the same thing.
// CHECK: @[[LAMBDA_X:[^\w]+]] = private unnamed_addr addrspace(1) constant [[LAMBDA_X_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE_\00"
// CHECK: @[[LAMBDA_X2:[^\w]+]] = private unnamed_addr addrspace(1) constant [[LAMBDA_X_SIZE]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE_\00"

// Note: None of these are used to name the kernel, so they just get the default names. You can see the kernel name in the E10000.
// CHECK: @[[MACRO_X:[^\w]+]] = private unnamed_addr addrspace(1) constant [[MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE0_\00"
// CHECK: @[[MACRO_Y:[^\w]+]] =  private unnamed_addr addrspace(1) constant [[MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE1_\00"
// CHECK: @[[MACRO_MACRO_X:[^\w]+]] = private unnamed_addr addrspace(1) constant [[MACRO_MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE2_\00"
// CHECK: @[[MACRO_MACRO_Y:[^\w]+]] = private unnamed_addr addrspace(1) constant [[MACRO_MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE3_\00"
// CHECK: @[[MACRO_MACRO_X2:[^\w]+]] = private unnamed_addr addrspace(1) constant [[MACRO_MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE4_\00"
// CHECK: @[[MACRO_MACRO_Y2:[^\w]+]] = private unnamed_addr addrspace(1) constant [[MACRO_MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE5_\00"
// CHECK: @[[INT2:[^\w]+]] = private unnamed_addr addrspace(1) constant [[INT_SIZE]] c"_ZTSi\00"

// Another repetition of LAMBDA_X for the template_param function call.  Its the same type there.
// CHECK: @[[LAMBDA_X3:[^\w]+]] = private unnamed_addr addrspace(1) constant [[LAMBDA_X_SIZE]] c"_ZTSZZ4mainENKUlvE10000_clEvEUlvE_\00"

// CHECK: @[[LAMBDA_IN_DEP_INT:[^\w]+]] = private unnamed_addr addrspace(1) constant [[DEP_INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIiEvvEUlvE_\00",
// CHECK: @[[LAMBDA_IN_DEP_X:[^\w]+]] = private unnamed_addr addrspace(1) constant [[DEP_LAMBDA_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIZZ4mainENKUlvE10000_clEvEUlvE_EvvEUlvE_\00",

extern "C" void printf(const char *) {}

template <typename T>
void template_param() {
  printf(__builtin_unique_stable_name(T));
}

template <typename T>
T getT() { return T{}; }

template <typename T>
void lambda_in_dependent_function() {
  auto y = [] {};
  printf(__builtin_unique_stable_name(y));
}

#define DEF_IN_MACRO()                                  \
  auto MACRO_X = []() {};auto MACRO_Y = []() {};        \
  printf(__builtin_unique_stable_name(MACRO_X));        \
  printf(__builtin_unique_stable_name(MACRO_Y));

#define MACRO_CALLS_MACRO()                             \
  {DEF_IN_MACRO();}{DEF_IN_MACRO();}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel]] void kernel_single_task(const KernelType &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel>(
    []() {
      printf(__builtin_unique_stable_name(int));
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]] addrspace(1)* @[[INT]], i32 0, i32 0) to i8 addrspace(4)*))

      auto x = [](){};
      printf(__builtin_unique_stable_name(x));
      printf(__builtin_unique_stable_name(decltype(x)));
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]] addrspace(1)* @[[LAMBDA_X]], i32 0, i32 0) to i8 addrspace(4)*))
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]] addrspace(1)* @[[LAMBDA_X2]], i32 0, i32 0) to i8 addrspace(4)*))

      DEF_IN_MACRO();
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]] addrspace(1)* @[[MACRO_X]], i32 0, i32 0) to i8 addrspace(4)*))
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]] addrspace(1)* @[[MACRO_Y]], i32 0, i32 0) to i8 addrspace(4)*))

      MACRO_CALLS_MACRO();
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]] addrspace(1)* @[[MACRO_MACRO_X]], i32 0, i32 0) to i8 addrspace(4)*))
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]] addrspace(1)* @[[MACRO_MACRO_Y]], i32 0, i32 0) to i8 addrspace(4)*))
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]] addrspace(1)* @[[MACRO_MACRO_X2]], i32 0, i32 0) to i8 addrspace(4)*))
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]] addrspace(1)* @[[MACRO_MACRO_Y2]], i32 0, i32 0) to i8 addrspace(4)*))

      template_param<int>();
      // CHECK: define linkonce_odr spir_func void @_Z14template_paramIiEvv
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]] addrspace(1)* @[[INT2]], i32 0, i32 0) to i8 addrspace(4)*))

      template_param<decltype(x)>();
      // CHECK: define internal spir_func void @"_Z14template_paramIZZ4mainENK3
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]] addrspace(1)* @[[LAMBDA_X3]], i32 0, i32 0) to i8 addrspace(4)*))

      lambda_in_dependent_function<int>();
      // CHECK: define linkonce_odr spir_func void @_Z28lambda_in_dependent_functionIiEvv
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[DEP_INT_SIZE]], [[DEP_INT_SIZE]] addrspace(1)* @[[LAMBDA_IN_DEP_INT]], i32 0, i32 0) to i8 addrspace(4)*))

      lambda_in_dependent_function<decltype(x)>();
      // CHECK: define internal spir_func void @"_Z28lambda_in_dependent_functionIZZ4mainENK3$_0clEvEUlvE_Evv
      // CHECK: call spir_func void @printf(i8 addrspace(4)* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([[DEP_LAMBDA_SIZE]], [[DEP_LAMBDA_SIZE]] addrspace(1)* @[[LAMBDA_IN_DEP_X]], i32 0, i32 0) to i8 addrspace(4)*))

    });
}

