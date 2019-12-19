// RUN: %clang %s -fsyntax-only -fsycl-device-only -DTRIGGER_ERROR -Xclang -verify
// RUN: %clang %s -fsyntax-only -Xclang -ast-dump -fsycl-device-only | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

#ifndef __SYCL_DEVICE_ONLY__
struct FuncObj {
  [[intelfpga::max_global_work_dim(1)]] // expected-no-diagnostics
  void operator()() {}
};

template <typename name, typename Func>
void kernel(Func kernelFunc) {
  kernelFunc();
}

void foo() {
  kernel<class test_kernel1>(
      FuncObj());
}

#else // __SYCL_DEVICE_ONLY__

[[intelfpga::max_global_work_dim(2)]] // expected-warning{{'max_global_work_dim' attribute ignored}}
void func_ignore() {}

struct FuncObj {
  [[intelfpga::max_global_work_dim(1)]]
  void operator()() {}
};

struct TRIFuncObjGood1 {
  [[intelfpga::max_global_work_dim(0)]]
  [[intelfpga::max_work_group_size(1, 1, 1)]]
  [[cl::reqd_work_group_size(1, 1, 1)]]
  void operator()() {}
};

struct TRIFuncObjGood2 {
  [[intelfpga::max_global_work_dim(3)]]
  [[intelfpga::max_work_group_size(8, 1, 1)]]
  [[cl::reqd_work_group_size(4, 1, 1)]]
  void operator()() {}
};

#ifdef TRIGGER_ERROR
struct TRIFuncObjBad {
  [[intelfpga::max_global_work_dim(0)]]
  [[intelfpga::max_work_group_size(8, 8, 8)]] // expected-error{{'max_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  [[cl::reqd_work_group_size(4, 4, 4)]] // expected-error{{'reqd_work_group_size' X-, Y- and Z- sizes must be 1 when 'max_global_work_dim' attribute is used with value 0}}
  void operator()() {}
};
#endif // TRIGGER_ERROR

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12test_kernel1
  // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}} 1
  kernel<class test_kernel1>(
      FuncObj());

  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12test_kernel2
  // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}} 2
  kernel<class test_kernel2>(
      []() [[intelfpga::max_global_work_dim(2)]] {});

  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12test_kernel3
  // CHECK-NOT:   SYCLIntelMaxGlobalWorkDimAttr {{.*}}
  kernel<class test_kernel3>(
      []() {func_ignore();});

  kernel<class test_kernel4>(
      TRIFuncObjGood1());
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12test_kernel4
  // CHECK:       ReqdWorkGroupSizeAttr {{.*}} 1 1 1
  // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}} 1 1 1
  // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}} 0

  kernel<class test_kernel5>(
      TRIFuncObjGood2());
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12test_kernel5
  // CHECK:       ReqdWorkGroupSizeAttr {{.*}} 4 1 1
  // CHECK:       SYCLIntelMaxWorkGroupSizeAttr {{.*}} 8 1 1
  // CHECK:       SYCLIntelMaxGlobalWorkDimAttr {{.*}} 3

#ifdef TRIGGER_ERROR
  [[intelfpga::max_global_work_dim(1)]] int Var = 0; // expected-error{{'max_global_work_dim' attribute only applies to functions}}

  kernel<class test_kernel6>(
      []() [[intelfpga::max_global_work_dim(-8)]] {}); // expected-error{{'max_global_work_dim' attribute requires a non-negative integral compile time constant expression}}

  kernel<class test_kernel7>(
      []() [[intelfpga::max_global_work_dim(3),
             intelfpga::max_global_work_dim(2)]] {}); // expected-warning{{attribute 'max_global_work_dim' is already applied with different parameters}}

  kernel<class test_kernel8>(
      TRIFuncObjBad());

  kernel<class test_kernel9>(
      []() [[intelfpga::max_global_work_dim(4)]] {}); // expected-error{{The value of 'max_global_work_dim' attribute must be in range from 0 to 3}}

#endif // TRIGGER_ERROR
}
#endif // __SYCL_DEVICE_ONLY__
