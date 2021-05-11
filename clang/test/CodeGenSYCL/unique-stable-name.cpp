// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// CHECK: @[[INT:[^\w]+]] = private unnamed_addr addrspace(1) constant [[INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSi\00"
// CHECK: @[[LAMBDA_X:[^\w]+]] = private unnamed_addr addrspace(1) constant [[LAMBDA_X_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE42_5clEvEUlvE46_16\00"
// CHECK: @[[MACRO_X:[^\w]+]] = private unnamed_addr addrspace(1) constant [[MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE42_5clEvEUlvE52_7m28_18\00"
// CHECK: @[[MACRO_Y:[^\w]+]] =  private unnamed_addr addrspace(1) constant [[MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE42_5clEvEUlvE52_7m28_41\00"
// CHECK: @[[MACRO_MACRO_X:[^\w]+]] = private unnamed_addr addrspace(1) constant [[MACRO_MACRO_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZZ4mainENKUlvE42_5clEvEUlvE55_7m28_18m33_4\00"
// CHECK: @[[MACRO_MACRO_Y:[^\w]+]] = private unnamed_addr addrspace(1) constant [[MACRO_MACRO_SIZE]] c"_ZTSZZ4mainENKUlvE42_5clEvEUlvE55_7m28_41m33_4\00"
// CHECK: @[[LAMBDA_IN_DEP_INT:[^\w]+]] = private unnamed_addr addrspace(1) constant [[DEP_INT_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIiEvvEUlvE23_12\00",
// CHECK: @[[LAMBDA_IN_DEP_X:[^\w]+]] = private unnamed_addr addrspace(1) constant [[DEP_LAMBDA_SIZE:\[[0-9]+ x i8\]]] c"_ZTSZ28lambda_in_dependent_functionIZZ4mainENKUlvE42_5clEvEUlvE46_16EvvEUlvE23_12\00",

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
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]] addrspace(4)* addrspacecast ([[INT_SIZE]] addrspace(1)* @[[INT]] to [[INT_SIZE]] addrspace(4)*

      auto x = [](){};
      printf(__builtin_unique_stable_name(x));
      printf(__builtin_unique_stable_name(decltype(x)));
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]] addrspace(4)* addrspacecast ([[LAMBDA_X_SIZE]] addrspace(1)* @[[LAMBDA_X]] to [[LAMBDA_X_SIZE]] addrspace(4)*
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]] addrspace(4)* addrspacecast ([[LAMBDA_X_SIZE]] addrspace(1)* @[[LAMBDA_X]] to [[LAMBDA_X_SIZE]] addrspace(4)*

      DEF_IN_MACRO();
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]] addrspace(4)* addrspacecast ([[MACRO_SIZE]] addrspace(1)* @[[MACRO_X]] to [[MACRO_SIZE]] addrspace(4)*
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[MACRO_SIZE]], [[MACRO_SIZE]] addrspace(4)* addrspacecast ([[MACRO_SIZE]] addrspace(1)* @[[MACRO_Y]] to [[MACRO_SIZE]] addrspace(4)*
      MACRO_CALLS_MACRO();
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]] addrspace(4)* addrspacecast ([[MACRO_MACRO_SIZE]] addrspace(1)* @[[MACRO_MACRO_X]] to [[MACRO_MACRO_SIZE]] addrspace(4)*
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[MACRO_MACRO_SIZE]], [[MACRO_MACRO_SIZE]] addrspace(4)* addrspacecast ([[MACRO_MACRO_SIZE]] addrspace(1)* @[[MACRO_MACRO_Y]] to [[MACRO_MACRO_SIZE]] addrspace(4)*

      template_param<int>();
      // CHECK: define linkonce_odr spir_func void @_Z14template_paramIiEvv
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[INT_SIZE]], [[INT_SIZE]] addrspace(4)* addrspacecast ([[INT_SIZE]] addrspace(1)* @[[INT]] to [[INT_SIZE]] addrspace(4)*

      template_param<decltype(x)>();
      // CHECK: define internal spir_func void @"_Z14template_paramIZZ4mainENK3
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[LAMBDA_X_SIZE]], [[LAMBDA_X_SIZE]] addrspace(4)* addrspacecast ([[LAMBDA_X_SIZE]] addrspace(1)* @[[LAMBDA_X]] to [[LAMBDA_X_SIZE]] addrspace(4)*

      lambda_in_dependent_function<int>();
      // CHECK: define linkonce_odr spir_func void @_Z28lambda_in_dependent_functionIiEvv
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[DEP_INT_SIZE]], [[DEP_INT_SIZE]] addrspace(4)* addrspacecast ([[DEP_INT_SIZE]] addrspace(1)* @[[LAMBDA_IN_DEP_INT]] to [[DEP_INT_SIZE]] addrspace(4)*

      lambda_in_dependent_function<decltype(x)>();
      // CHECK: define internal spir_func void @"_Z28lambda_in_dependent_functionIZZ4mainENK3$_0clEvEUlvE_Evv
      // CHECK: call spir_func void @printf(i8 addrspace(4)* getelementptr inbounds ([[DEP_LAMBDA_SIZE]], [[DEP_LAMBDA_SIZE]] addrspace(4)* addrspacecast ([[DEP_LAMBDA_SIZE]] addrspace(1)* @[[LAMBDA_IN_DEP_X]] to [[DEP_LAMBDA_SIZE]] addrspace(4)*

    });
}

