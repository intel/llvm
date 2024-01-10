// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s
// RUN: clang++ -fsycl -fsycl-device-only -O1 -w -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ROUNDED
// RUN: clang++ -fsycl -fsycl-device-only -O2 -w -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ROUNDED
// RUN: clang++ -fsycl -fsycl-device-only -O3 -w -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ROUNDED

#include <sycl/sycl.hpp>

// FIXME: Add spec constants test when fixed in frontend

// COM: Test to check 'sycl.kernel_func_obj' attribute is attached to every KernelFuncObj function.
//
// COM: Range rounded kernels do not call that function directly, so detection will depend on
// COM: the not rounded kernel.
//
// COM: 'CHECK-NOT attributes...' check only the desired functions have the attribute attached.

template <typename KernelFuncObjArg>
class KernelImpl {
public:
  KernelImpl(float *in, float *out) : in(in), out(out) {}
  void operator()(KernelFuncObjArg i) const;
private:
  float *in;
  float *out;
};

template<>
void KernelImpl<sycl::nd_item<1>>::operator()(sycl::nd_item<1> i) const {
  out[i.get_global_id()] = in[i.get_global_id()];
}

template <typename KernelFuncObjArg>
void KernelImpl<KernelFuncObjArg>::operator()(KernelFuncObjArg i) const {
  out[i] = in[i];
}

template <typename Range, typename KernelFuncObjArg>
void test(sycl::queue q, Range r, float *in, float *out) {
  q.parallel_for(r, KernelImpl<KernelFuncObjArg>(in, out));
}

// COM: Do not check for -O0; range rounding is disabled.

// CHECK-ROUNDED-NOT:     sycl.kernel_func_obj

// CHECK-ROUNDED-LABEL:     gpu.func @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1E10KernelImplINS0_2idILi1EEEEEE
// CHECK-ROUNDED-NOT:         func.call
// CHECK-ROUNDED:             gpu.return

// CHECK-NOT:         sycl.kernel_func_obj

// CHECK-LABEL:     gpu.func @_ZTS10KernelImplIN4sycl3_V12idILi1EEEE

// CHECK-NOT:         sycl.kernel_func_obj

// COM: As `range` is used, a rounded range kernel should be generated too.

// CHECK-LABEL:     func.func @_ZNK10KernelImplIN4sycl3_V12idILi1EEEEclES3_
// CHECK-SAME:        sycl.kernel_func_obj = [@_ZTS10KernelImplIN4sycl3_V12idILi1EEEE
// CHECK-ROUNDED:     , @_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1E10KernelImplINS0_2idILi1EEEEEE]

template
void test<sycl::range<1>, sycl::id<1>>(sycl::queue q, sycl::range<1> r, float *in, float *out);

// CHECK-NOT:         sycl.kernel_func_obj

// CHECK-LABEL:     gpu.func @_ZTS10KernelImplIN4sycl3_V14itemILi1ELb1EEEE

// CHECK-NOT:         sycl.kernel_func_obj

// CHECK-LABEL:     func.func @_ZNK10KernelImplIN4sycl3_V14itemILi1ELb1EEEEclES3_
// CHECK-SAME:        sycl.kernel_func_obj = [@_ZTS10KernelImplIN4sycl3_V14itemILi1ELb1EEEE]

template
void test<sycl::nd_range<1>, sycl::item<1>>(sycl::queue q, sycl::nd_range<1> r, float *in, float *out);

// CHECK-NOT:         sycl.kernel_func_obj

// CHECK-LABEL:     gpu.func @_ZTS10KernelImplIN4sycl3_V17nd_itemILi1EEEE

// CHECK-NOT:         sycl.kernel_func_obj

// CHECK:           func.func @_ZNK10KernelImplIN4sycl3_V17nd_itemILi1EEEEclES3_
// CHECK-SAME:        sycl.kernel_func_obj = [@_ZTS10KernelImplIN4sycl3_V17nd_itemILi1EEEE]

template
void test<sycl::nd_range<1>, sycl::nd_item<1>>(sycl::queue q, sycl::nd_range<1> r, float *in, float *out);
