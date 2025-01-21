// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fno-sycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem=false -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATEFUL

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATELESS

// Checks ESIMD copy_to and copy_from functions accepting compile time
// properties. NOTE: must be run in -O0, as optimizer optimizes away some of the
// code.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

using AccType = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write>;
using LocalAccType = sycl::local_accessor<double, 1>;
using LocalAccTypeInt = sycl::local_accessor<int, 1>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_copy_to(AccType &, LocalAccType &,
                                                    float *, int byte_offset32,
                                                    size_t byte_offset64);
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_copy_from(AccType &, LocalAccType &,
                                                      float *,
                                                      int byte_offset32,
                                                      size_t byte_offset64);
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_ctor(AccType &, LocalAccType &,
                                                 float *, int byte_offset32,
                                                 size_t byte_offset64);
class EsimdFunctor {
public:
  AccType acc;
  LocalAccType local_acc;
  LocalAccTypeInt local_acc_int;
  float *ptr;
  int byte_offset32;
  size_t byte_offset64;
  void operator()() __attribute__((sycl_explicit_simd)) {
    test_copy_to(acc, local_acc, ptr, byte_offset32, byte_offset64);
    test_copy_from(acc, local_acc, ptr, byte_offset32, byte_offset64);
    test_ctor(acc, local_acc, ptr, byte_offset32, byte_offset64);
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar(AccType &acc, LocalAccType &local_acc, LocalAccTypeInt &local_acc_int,
         float *ptr, int byte_offset32, size_t byte_offset64) {
  EsimdFunctor esimdf{acc, local_acc,     local_acc_int,
                      ptr, byte_offset32, byte_offset64};
  kernel<class kernel_esimd>(esimdf);
}

// CHECK-LABEL: define {{.*}} @_Z12test_copy_to{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_copy_to(AccType &acc, LocalAccType &local_acc, float *ptrf,
             int byte_offset32, size_t byte_offset64) {
  properties props_a{cache_hint_L1<cache_hint::write_back>,
                     cache_hint_L2<cache_hint::write_back>, alignment<32>};
  properties props_b{alignment<32>};
  simd<float, 16> vals;
  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v16f32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 6, i8 2, i8 0, <1 x i64> {{[^)]+}}, <16 x float> {{[^)]+}}, i32 0)
  vals.copy_to(ptrf, props_a);
  // CHECK: store <16 x float> {{[^)]+}}, ptr addrspace(4) {{[^)]+}}, align 32
  vals.copy_to(ptrf, props_b);

  // CHECK-STATEFUL: call void @llvm.genx.lsc.store.bti.v1i1.v1i32.v16f32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 6, i8 2, i8 0, <1 x i32> {{[^)]+}}, <16 x float> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v16f32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 6, i8 2, i8 0, <1 x i64> {{[^)]+}}, <16 x float> {{[^)]+}}, i32 0)
  vals.copy_to(acc, byte_offset64, props_a);

  // CHECK-STATEFUL: call void @llvm.genx.oword.st.v16f32(i32 {{[^)]+}}, i32 {{[^)]+}}, <16 x float> {{[^)]+}})
  // CHECK-STATELESS: store <16 x float> {{[^)]+}}, ptr addrspace(4) {{[^)]+}}, align 32
  vals.copy_to(acc, byte_offset64, props_b);

  // CHECK-COUNT-2: store <16 x float> {{[^)]+}}, ptr addrspace(3) {{[^)]+}}, align 32
  vals.copy_to(local_acc, byte_offset64, props_a);
  vals.copy_to(local_acc, byte_offset64, props_b);
}

// CHECK-LABEL: define {{.*}} @_Z14test_copy_from{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_copy_from(AccType &acc, LocalAccType &local_acc, float *ptrf,
               int byte_offset32, size_t byte_offset64) {
  properties props_a{cache_hint_L1<cache_hint::cached>,
                     cache_hint_L2<cache_hint::cached>, alignment<32>};
  properties props_b{alignment<32>};
  simd<float, 16> vals;
  // CHECK: call <16 x float> @llvm.genx.lsc.load.merge.stateless.v16f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 2, i16 1, i32 0, i8 3, i8 6, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <16 x float> {{[^)]+}})
  vals.copy_from(ptrf, props_a);

  // CHECK: load <16 x float>, ptr addrspace(4) {{[^)]+}}, align 32
  vals.copy_from(ptrf, props_b);

  // CHECK-STATEFUL: call <16 x float> @llvm.genx.lsc.load.bti.v16f32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 2, i16 1, i32 0, i8 3, i8 6, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: call <16 x float> @llvm.genx.lsc.load.merge.stateless.v16f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 2, i16 1, i32 0, i8 3, i8 6, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <16 x float> {{[^)]+}})
  vals.copy_from(acc, byte_offset64, props_a);

  // CHECK-STATEFUL: call <16 x float> @llvm.genx.oword.ld.v16f32(i32 0, i32 {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: load <16 x float>, ptr addrspace(4) {{[^)]+}}, align 32
  vals.copy_from(acc, byte_offset64, props_b);

  // CHECK-COUNT-2: load <16 x float>, ptr addrspace(3) {{[^)]+}}, align 32
  vals.copy_from(local_acc, byte_offset64, props_a);
  vals.copy_from(local_acc, byte_offset64, props_b);
}

// CHECK-LABEL: define {{.*}} @_Z9test_ctor{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_ctor(AccType &acc,
                                                 LocalAccType &local_acc,
                                                 float *ptrf, int byte_offset32,
                                                 size_t byte_offset64) {
  properties props_a{cache_hint_L1<cache_hint::cached>,
                     cache_hint_L2<cache_hint::cached>, alignment<32>};
  properties props_b{alignment<32>};
  // CHECK: call <8 x float> @llvm.genx.lsc.load.merge.stateless.v8f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 2, i16 1, i32 0, i8 3, i8 5, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <8 x float> {{[^)]+}})
  simd<float, 8> foo(ptrf, props_a);
  // CHECK: load <8 x float>, ptr addrspace(4) {{[^)]+}}, align 32
  simd<float, 8> bar(ptrf, props_b);

  // CHECK-STATEFUL: call <8 x float> @llvm.genx.lsc.load.bti.v8f32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 2, i16 1, i32 0, i8 3, i8 5, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: call <8 x float> @llvm.genx.lsc.load.merge.stateless.v8f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 2, i16 1, i32 0, i8 3, i8 5, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <8 x float> {{[^)]+}})
  simd<float, 8> acc_foo(acc, byte_offset64, props_a);

  // CHECK-STATEFUL: call <8 x float> @llvm.genx.oword.ld.v8f32(i32 0, i32 {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: load <8 x float>, ptr addrspace(4) {{[^)]+}}, align 32
  simd<float, 8> acc_bar(acc, byte_offset64, props_b);

  // CHECK-COUNT-2: load <8 x float>, ptr addrspace(3) {{[^)]+}}, align 32
  simd<float, 8> lacc_foo(local_acc, byte_offset64, props_a);
  simd<float, 8> lacc_bar(local_acc, byte_offset64, props_b);
}
