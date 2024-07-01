// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fno-sycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem=false -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATEFUL

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATELESS

// Checks ESIMD memory functions accepting compile time properties for scatter
// APIs. NOTE: must be run in -O0, as optimizer optimizes away some of the code.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

using AccType = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write>;
using LocalAccType = sycl::local_accessor<double, 1>;
using LocalAccTypeInt = sycl::local_accessor<int, 1>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_scatter(AccType &, LocalAccType &,
                                                    float *, int byte_offset32,
                                                    size_t byte_offset64);
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_slm_scatter(int byte_offset32);

class EsimdFunctor {
public:
  AccType acc;
  LocalAccType local_acc;
  LocalAccTypeInt local_acc_int;
  float *ptr;
  int byte_offset32;
  size_t byte_offset64;
  void operator()() __attribute__((sycl_explicit_simd)) {
    test_scatter(acc, local_acc, ptr, byte_offset32, byte_offset64);
    test_slm_scatter(byte_offset32);
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

// CHECK-LABEL: define {{.*}} @_Z12test_scatter{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_scatter(AccType &acc, LocalAccType &local_acc, float *ptrf,
             int byte_offset32, size_t byte_offset64) {
  properties props_cache_load{cache_hint_L1<cache_hint::uncached>,
                              cache_hint_L2<cache_hint::uncached>,
                              alignment<8>};

  properties props_align4{alignment<4>};
  properties props_align8{alignment<8>};
  properties props_align16{alignment<16>};

  int *ptri = reinterpret_cast<int *>(ptrf);

  simd<uint32_t, 32> ioffset_n32(byte_offset32, 8);
  simd<uint64_t, 32> loffset_n32(byte_offset64, 16);
  auto ioffset_n32_view = ioffset_n32.select<32, 1>();
  auto loffset_n32_view = loffset_n32.select<32, 1>();

  simd<uint32_t, 16> ioffset_n16(byte_offset32, 8);
  simd<uint64_t, 16> loffset_n16(byte_offset64, 16);
  auto ioffset_n16_view = ioffset_n16.select<16, 1>();
  auto loffset_n16_view = loffset_n16.select<16, 1>();

  simd_mask<32> mask_n32 = 1;
  simd_mask<16> mask_n16 = 1;

  simd<float, 32> usm;
  simd<float, 32> acc_res;
  simd<float, 32> pass_thru;
  auto pass_thru_view = pass_thru.select<32, 1>();

  auto usm_view = usm.select<32, 1>();

  // Validate that a new API doesn't conflict with the old API.
  // CHECK-COUNT-2: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  acc_res = gather<float, 32>(local_acc, ioffset_n32, 0);
  acc_res = gather<float, 32>(local_acc, ioffset_n32, 0, mask_n32);

  // CHECK-COUNT-4: call void @llvm.masked.scatter.v32f32.v32p4(<32 x float> {{[^)]+}}, <32 x ptr addrspace(4)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}})
  scatter(ptrf, ioffset_n32, usm, mask_n32);

  scatter(ptrf, ioffset_n32, usm);

  scatter(ptrf, ioffset_n32, usm, mask_n32, props_align4);

  scatter(ptrf, ioffset_n32, usm, props_align4);

  // CHECK-COUNT-22: call void @llvm.genx.lsc.store.stateless.v32i1.v32i64.v32i32(<32 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i64> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 0)
  scatter(ptrf, ioffset_n32, usm, mask_n32, props_cache_load);
  scatter(ptrf, ioffset_n32, usm, props_cache_load);

  scatter(ptrf, ioffset_n32_view, usm, mask_n32, props_cache_load);
  scatter(ptrf, ioffset_n32_view, usm, props_cache_load);

  scatter<float, 32>(ptrf, ioffset_n32, usm_view, mask_n32, props_cache_load);
  scatter<float, 32>(ptrf, ioffset_n32, usm_view, props_cache_load);

  scatter<float, 32>(ptrf, ioffset_n32_view, usm_view, mask_n32,
                     props_cache_load);
  scatter<float, 32>(ptrf, ioffset_n32_view, usm_view, props_cache_load);

  scatter(ptrf, ioffset_n32, usm_view, mask_n32, props_cache_load);
  scatter(ptrf, ioffset_n32, usm_view, props_cache_load);

  scatter(ptrf, ioffset_n32_view, usm_view, mask_n32, props_cache_load);
  scatter(ptrf, ioffset_n32_view, usm_view, props_cache_load);

  scatter(ptrf, ioffset_n32_view.select<32, 1>(), usm, mask_n32,
          props_cache_load);
  scatter(ptrf, ioffset_n32_view.select<32, 1>(), usm, props_cache_load);

  scatter<float, 32>(ptrf, ioffset_n32, usm_view.select<32, 1>(), mask_n32,
                     props_cache_load);
  scatter<float, 32>(ptrf, ioffset_n32, usm_view.select<32, 1>(),
                     props_cache_load);

  scatter<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(),
                     usm_view.select<32, 1>(), mask_n32, props_cache_load);
  scatter<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(),
                     usm_view.select<32, 1>(), props_cache_load);
  scatter(ptrf, ioffset_n32, usm_view.select<32, 1>(), mask_n32,
          props_cache_load);
  scatter(ptrf, ioffset_n32, usm_view.select<32, 1>(), props_cache_load);

  scatter(ptrf, ioffset_n32_view.select<32, 1>(), usm_view.select<32, 1>(),
          mask_n32, props_cache_load);
  scatter(ptrf, ioffset_n32_view.select<32, 1>(), usm_view.select<32, 1>(),
          props_cache_load);

  // VS > 1
  // CHECK-COUNT-24: call void @llvm.genx.lsc.store.stateless.v16i1.v16i64.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 0)
  scatter<float, 32, 2>(ptrf, ioffset_n16, usm, mask_n16, props_cache_load);

  scatter<float, 32, 2>(ptrf, ioffset_n16, usm, props_cache_load);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view, usm, mask_n16,
                        props_cache_load);
  scatter<float, 32, 2>(ptrf, ioffset_n16_view, usm, props_cache_load);

  scatter<float, 32, 2>(ptrf, ioffset_n16, usm_view, mask_n16,
                        props_cache_load);
  scatter<float, 32, 2>(ptrf, ioffset_n16, usm_view, props_cache_load);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view, usm_view, mask_n16,
                        props_cache_load);
  scatter<float, 32, 2>(ptrf, ioffset_n16_view, usm_view, props_cache_load);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), usm, mask_n16,
                        props_cache_load);
  scatter<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), usm,
                        props_cache_load);

  scatter<2>(ptrf, ioffset_n16, usm_view, mask_n16, props_cache_load);
  scatter<2>(ptrf, ioffset_n16, usm_view, props_cache_load);

  scatter<2>(ptrf, ioffset_n16_view, usm_view, mask_n16, props_cache_load);
  scatter<2>(ptrf, ioffset_n16_view, usm_view, props_cache_load);

  scatter<2>(ptrf, ioffset_n16_view.select<16, 1>(), usm, mask_n16,
             props_cache_load);
  scatter<2>(ptrf, ioffset_n16_view.select<16, 1>(), usm, props_cache_load);

  scatter<float, 32, 2>(ptrf, ioffset_n16, usm_view.select<32, 1>(), mask_n16,
                        props_cache_load);
  scatter<float, 32, 2>(ptrf, ioffset_n16, usm_view.select<32, 1>(),
                        props_cache_load);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), mask_n16, props_cache_load);
  scatter<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), props_cache_load);

  scatter<2>(ptrf, ioffset_n16, usm_view.select<32, 1>(), mask_n16,
             props_cache_load);
  scatter<2>(ptrf, ioffset_n16, usm_view.select<32, 1>(), props_cache_load);

  scatter<2>(ptrf, ioffset_n16_view.select<16, 1>(), usm_view.select<32, 1>(),
             mask_n16, props_cache_load);
  scatter<2>(ptrf, ioffset_n16_view.select<16, 1>(), usm_view.select<32, 1>(),
             props_cache_load);

  // CHECK-COUNT-14: call void @llvm.genx.lsc.store.stateless.v16i1.v16i64.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 0)
  scatter<float, 32, 2>(ptrf, ioffset_n16, usm, mask_n16);

  scatter<float, 32, 2>(ptrf, ioffset_n16, usm);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view, usm, mask_n16);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view, usm);

  scatter<float, 32, 2>(ptrf, ioffset_n16, usm_view, mask_n16);

  scatter<float, 32, 2>(ptrf, ioffset_n16, usm_view);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view, usm_view, mask_n16);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view, usm_view);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), usm, mask_n16);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), usm);

  scatter<float, 32, 2>(ptrf, ioffset_n16, usm_view.select<32, 1>(), mask_n16);

  scatter<float, 32, 2>(ptrf, ioffset_n16, usm_view.select<32, 1>());

  scatter<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), mask_n16);

  scatter<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>());

  // CHECK-COUNT-4: call void @llvm.masked.scatter.v32f32.v32p3(<32 x float> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}})
  scatter(local_acc, ioffset_n32, usm, mask_n32);

  scatter(local_acc, ioffset_n32, usm);

  scatter(local_acc, ioffset_n32, usm, mask_n32, props_align4);

  scatter(local_acc, ioffset_n32, usm, props_align4);

  // CHECK-COUNT-22: call void @llvm.masked.scatter.v32f32.v32p3(<32 x float> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}})
  scatter(local_acc, ioffset_n32, usm, mask_n32, props_align4);
  scatter(local_acc, ioffset_n32, usm, props_align4);

  scatter(local_acc, ioffset_n32_view, usm, mask_n32, props_align4);
  scatter(local_acc, ioffset_n32_view, usm, props_align4);

  scatter<float, 32>(local_acc, ioffset_n32, usm_view, mask_n32, props_align4);
  scatter<float, 32>(local_acc, ioffset_n32, usm_view, props_align4);

  scatter<float, 32>(local_acc, ioffset_n32_view, usm_view, mask_n32,
                     props_align4);
  scatter<float, 32>(local_acc, ioffset_n32_view, usm_view, props_align4);

  scatter(local_acc, ioffset_n32, usm_view, mask_n32, props_align4);
  scatter(local_acc, ioffset_n32, usm_view, props_align4);

  scatter(local_acc, ioffset_n32_view, usm_view, mask_n32, props_align4);
  scatter(local_acc, ioffset_n32_view, usm_view, props_align4);

  scatter(local_acc, ioffset_n32_view.select<32, 1>(), usm, mask_n32,
          props_align4);
  scatter(local_acc, ioffset_n32_view.select<32, 1>(), usm, props_align4);

  scatter<float, 32>(local_acc, ioffset_n32, usm_view.select<32, 1>(), mask_n32,
                     props_align4);
  scatter<float, 32>(local_acc, ioffset_n32, usm_view.select<32, 1>(),
                     props_align4);

  scatter<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(),
                     usm_view.select<32, 1>(), mask_n32, props_align4);
  scatter<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(),
                     usm_view.select<32, 1>(), props_align4);

  scatter(local_acc, ioffset_n32, usm_view.select<32, 1>(), mask_n32,
          props_align4);
  scatter(local_acc, ioffset_n32, usm_view.select<32, 1>(), props_align4);

  scatter(local_acc, ioffset_n32_view.select<32, 1>(), usm_view.select<32, 1>(),
          mask_n32, props_align4);
  scatter(local_acc, ioffset_n32_view.select<32, 1>(), usm_view.select<32, 1>(),
          props_align4);

  // VS > 1
  // CHECK-COUNT-26: call void @llvm.genx.lsc.store.slm.v16i1.v16i32.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, <32 x i32>{{[^)]+}}, i32 0)
  scatter<float, 32, 2>(local_acc, ioffset_n16, usm, mask_n16, props_align4);

  scatter<float, 32, 2>(local_acc, ioffset_n16, usm, props_align4);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view, usm, mask_n16,
                        props_align4);
  scatter<float, 32, 2>(local_acc, ioffset_n16_view, usm, props_align4);

  scatter<float, 32, 2>(local_acc, ioffset_n16, usm_view, mask_n16,
                        props_align4);
  scatter<float, 32, 2>(local_acc, ioffset_n16, usm_view, props_align4);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view, usm_view, mask_n16,
                        props_align4);
  scatter<float, 32, 2>(local_acc, ioffset_n16_view, usm_view, props_align4);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(), usm,
                        mask_n16, props_align4);
  scatter<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(), usm,
                        props_align4);

  scatter<float, 32, 2>(local_acc, ioffset_n16, usm_view.select<32, 1>(),
                        mask_n16, props_align4);
  scatter<float, 32, 2>(local_acc, ioffset_n16, usm_view.select<32, 1>(),
                        props_align4);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), mask_n16, props_align4);
  scatter<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), props_align4);

  scatter<2>(local_acc, ioffset_n16_view, usm, mask_n16, props_align4);
  scatter<2>(local_acc, ioffset_n16_view, usm, props_align4);

  scatter<2>(local_acc, ioffset_n16, usm_view, mask_n16, props_align4);
  scatter<2>(local_acc, ioffset_n16, usm_view, props_align4);

  scatter<2>(local_acc, ioffset_n16_view, usm_view, mask_n16, props_align4);
  scatter<2>(local_acc, ioffset_n16_view, usm_view, props_align4);

  scatter<2>(local_acc, ioffset_n16_view.select<16, 1>(), usm, mask_n16,
             props_align4);
  scatter<2>(local_acc, ioffset_n16_view.select<16, 1>(), usm, props_align4);

  scatter<2>(local_acc, ioffset_n16, usm_view.select<32, 1>(), mask_n16,
             props_align4);
  scatter<2>(local_acc, ioffset_n16, usm_view.select<32, 1>(), props_align4);

  scatter<2>(local_acc, ioffset_n16_view.select<16, 1>(),
             usm_view.select<32, 1>(), mask_n16, props_align4);
  scatter<2>(local_acc, ioffset_n16_view.select<16, 1>(),
             usm_view.select<32, 1>(), props_align4);

  // CHECK-COUNT-26: call void @llvm.genx.lsc.store.slm.v16i1.v16i32.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, <32 x i32>{{[^)]+}}, i32 0)
  scatter<float, 32, 2>(local_acc, ioffset_n16, usm, mask_n16);

  scatter<float, 32, 2>(local_acc, ioffset_n16, usm);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view, usm, mask_n16);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view, usm);

  scatter<float, 32, 2>(local_acc, ioffset_n16, usm_view, mask_n16);

  scatter<float, 32, 2>(local_acc, ioffset_n16, usm_view);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view, usm_view, mask_n16);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view, usm_view);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(), usm,
                        mask_n16);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(), usm);

  scatter<float, 32, 2>(local_acc, ioffset_n16, usm_view.select<32, 1>(),
                        mask_n16);

  scatter<float, 32, 2>(local_acc, ioffset_n16, usm_view.select<32, 1>());

  scatter<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), mask_n16);

  scatter<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>());

  scatter<2>(local_acc, ioffset_n16_view, usm, mask_n16);

  scatter<2>(local_acc, ioffset_n16_view, usm);

  scatter<2>(local_acc, ioffset_n16, usm_view, mask_n16);

  scatter<2>(local_acc, ioffset_n16, usm_view);

  scatter<2>(local_acc, ioffset_n16_view, usm_view, mask_n16);

  scatter<2>(local_acc, ioffset_n16_view, usm_view);

  scatter<2>(local_acc, ioffset_n16_view.select<16, 1>(), usm, mask_n16);

  scatter<2>(local_acc, ioffset_n16_view.select<16, 1>(), usm);

  scatter<2>(local_acc, ioffset_n16, usm_view.select<32, 1>(), mask_n16);

  scatter<2>(local_acc, ioffset_n16, usm_view.select<32, 1>());

  scatter<2>(local_acc, ioffset_n16_view.select<16, 1>(),
             usm_view.select<32, 1>(), mask_n16);

  scatter<2>(local_acc, ioffset_n16_view.select<16, 1>(),
             usm_view.select<32, 1>());

  simd<uint32_t, 10> ioffset_n10(byte_offset32, 8);
  simd<float, 10> usm_n10;

  // Check special case involving global offset and mask
  // CHECK-COUNT-2: call void @llvm.masked.scatter.v32f32.v32p3(<32 x float> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}})
  scatter<float, 32>(local_acc, ioffset_n32, usm, 0, 1);
  scatter<float, 32>(local_acc, ioffset_n32, usm, 0);

  // Check special case to verify that for cases when N is not power of 2 llvm
  // intrinsic is used
  // CHECK-COUNT-1: call void @llvm.masked.scatter.v10f32.v10p4(<10 x float> {{[^)]+}}, <10 x ptr addrspace(4)> {{[^)]+}}, i32 4, <10 x i1> {{[^)]+}})
  scatter(ptrf, ioffset_n10, usm_n10);

  // Test accessor
  // CHECK-STATEFUL-COUNT-4: call void @llvm.genx.scatter.scaled.v32i1.v32i32.v32f32(<32 x i1> {{[^)]+}}, i16 0, i32 {{[^)]+}}, i32 {{[^)]+}}, <32 x i32> {{[^)]+}}, <32 x float> {{[^)]+}})
  // CHECK-STATELESS-COUNT-4: call void @llvm.masked.scatter.v32f32.v32p4(<32 x float> {{[^)]+}}, <32 x ptr addrspace(4)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}})
  scatter(acc, ioffset_n32, usm, mask_n32);

  scatter(acc, ioffset_n32, usm);

  scatter(acc, ioffset_n32, usm, mask_n32, props_align4);

  scatter(acc, ioffset_n32, usm, props_align4);

  // CHECK-STATEFUL-COUNT-20: call void @llvm.genx.lsc.store.bti.v32i1.v32i32.v32i32(<32 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i32> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS-COUNT-20: call void @llvm.genx.lsc.store.stateless.v32i1.v32i64.v32i32(<32 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i64> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 0)
  scatter(acc, ioffset_n32, usm, mask_n32, props_cache_load);
  scatter(acc, ioffset_n32, usm, props_cache_load);

  scatter(acc, ioffset_n32_view, usm, mask_n32, props_cache_load);
  scatter(acc, ioffset_n32_view, usm, props_cache_load);

  scatter<float, 32>(acc, ioffset_n32, usm_view, mask_n32, props_cache_load);
  scatter<float, 32>(acc, ioffset_n32, usm_view, props_cache_load);

  scatter<float, 32>(acc, ioffset_n32_view, usm_view, mask_n32,
                     props_cache_load);
  scatter<float, 32>(acc, ioffset_n32_view, usm_view, props_cache_load);

  scatter<float, 32>(acc, ioffset_n32, usm_view.select<32, 1>(), mask_n32,
                     props_cache_load);
  scatter<float, 32>(acc, ioffset_n32, usm_view.select<32, 1>(),
                     props_cache_load);

  scatter<float, 32>(acc, ioffset_n32_view.select<32, 1>(),
                     usm_view.select<32, 1>(), mask_n32, props_cache_load);
  scatter<float, 32>(acc, ioffset_n32_view.select<32, 1>(),
                     usm_view.select<32, 1>(), props_cache_load);

  scatter(acc, ioffset_n32, usm_view, mask_n32, props_cache_load);
  scatter(acc, ioffset_n32, usm_view, props_cache_load);

  scatter(acc, ioffset_n32_view, usm_view, mask_n32, props_cache_load);
  scatter(acc, ioffset_n32_view, usm_view, props_cache_load);

  scatter(acc, ioffset_n32, usm_view.select<32, 1>(), mask_n32,
          props_cache_load);
  scatter(acc, ioffset_n32, usm_view.select<32, 1>(), props_cache_load);

  scatter(acc, ioffset_n32_view.select<32, 1>(), usm_view.select<32, 1>(),
          mask_n32, props_cache_load);
  scatter(acc, ioffset_n32_view.select<32, 1>(), usm_view.select<32, 1>(),
          props_cache_load);

  // VS > 1
  // CHECK-STATELESS-COUNT-26: call void @llvm.genx.lsc.store.stateless.v16i1.v16i64.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 0)
  // CHECK-STATEFUL-COUNT-26: call void @llvm.genx.lsc.store.bti.v16i1.v16i32.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 {{[^)]+}})
  scatter<float, 32, 2>(acc, ioffset_n16, usm, mask_n16, props_cache_load);

  scatter<float, 32, 2>(acc, ioffset_n16, usm, props_cache_load);

  scatter<float, 32, 2>(acc, ioffset_n16_view, usm, mask_n16, props_cache_load);
  scatter<float, 32, 2>(acc, ioffset_n16_view, usm, props_cache_load);

  scatter<float, 32, 2>(acc, ioffset_n16, usm_view, mask_n16, props_cache_load);
  scatter<float, 32, 2>(acc, ioffset_n16, usm_view, props_cache_load);

  scatter<float, 32, 2>(acc, ioffset_n16_view, usm_view, mask_n16,
                        props_cache_load);
  scatter<float, 32, 2>(acc, ioffset_n16_view, usm_view, props_cache_load);

  scatter<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(), usm, mask_n16,
                        props_cache_load);
  scatter<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(), usm,
                        props_cache_load);

  scatter<float, 32, 2>(acc, ioffset_n16, usm_view.select<32, 1>(), mask_n16,
                        props_cache_load);
  scatter<float, 32, 2>(acc, ioffset_n16, usm_view.select<32, 1>(),
                        props_cache_load);

  scatter<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), mask_n16, props_cache_load);
  scatter<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), props_cache_load);

  scatter<2>(acc, ioffset_n16_view, usm, mask_n16, props_cache_load);
  scatter<2>(acc, ioffset_n16_view, usm, props_cache_load);

  scatter<2>(acc, ioffset_n16, usm_view, mask_n16, props_cache_load);
  scatter<2>(acc, ioffset_n16, usm_view, props_cache_load);

  scatter<2>(acc, ioffset_n16_view, usm_view, mask_n16, props_cache_load);
  scatter<2>(acc, ioffset_n16_view, usm_view, props_cache_load);

  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm, mask_n16,
             props_cache_load);
  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm, props_cache_load);

  scatter<2>(acc, ioffset_n16, usm_view.select<32, 1>(), mask_n16,
             props_cache_load);
  scatter<2>(acc, ioffset_n16, usm_view.select<32, 1>(), props_cache_load);

  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm_view.select<32, 1>(),
             mask_n16, props_cache_load);
  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm_view.select<32, 1>(),
             props_cache_load);

  // CHECK-STATELESS-COUNT-26: call void @llvm.genx.lsc.store.stateless.v16i1.v16i64.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 0)
  // CHECK-STATEFUL-COUNT-26:  call void @llvm.genx.lsc.store.bti.v16i1.v16i32.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, <32 x i32> {{[^)]+}}, i32 {{[^)]+}})
  scatter<float, 32, 2>(acc, ioffset_n16, usm, mask_n16);

  scatter<float, 32, 2>(acc, ioffset_n16, usm);

  scatter<float, 32, 2>(acc, ioffset_n16_view, usm, mask_n16);

  scatter<float, 32, 2>(acc, ioffset_n16_view, usm);

  scatter<float, 32, 2>(acc, ioffset_n16, usm_view, mask_n16);

  scatter<float, 32, 2>(acc, ioffset_n16, usm_view);

  scatter<float, 32, 2>(acc, ioffset_n16_view, usm_view, mask_n16);

  scatter<float, 32, 2>(acc, ioffset_n16_view, usm_view);

  scatter<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(), usm, mask_n16);

  scatter<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(), usm);

  scatter<float, 32, 2>(acc, ioffset_n16, usm_view.select<32, 1>(), mask_n16);

  scatter<float, 32, 2>(acc, ioffset_n16, usm_view.select<32, 1>());

  scatter<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(),
                        usm_view.select<32, 1>(), mask_n16);

  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm_view.select<32, 1>());

  scatter<2>(acc, ioffset_n16_view, usm, mask_n16);

  scatter<2>(acc, ioffset_n16_view, usm);

  scatter<2>(acc, ioffset_n16, usm_view, mask_n16);

  scatter<2>(acc, ioffset_n16, usm_view);

  scatter<2>(acc, ioffset_n16_view, usm_view, mask_n16);

  scatter<2>(acc, ioffset_n16_view, usm_view);

  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm, mask_n16);

  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm);

  scatter<2>(acc, ioffset_n16, usm_view.select<32, 1>(), mask_n16);

  scatter<2>(acc, ioffset_n16, usm_view.select<32, 1>());

  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm_view.select<32, 1>(),
             mask_n16);

  scatter<2>(acc, ioffset_n16_view.select<16, 1>(), usm_view.select<32, 1>());
}

// CHECK-LABEL: define {{.*}} @_Z16test_slm_scatter{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_slm_scatter(int byte_offset32) {

  properties props_align4{alignment<4>};
  properties props_align8{alignment<8>};

  simd<uint32_t, 32> ioffset_n32(byte_offset32, 8);
  auto ioffset_n32_view = ioffset_n32.select<32, 1>();

  simd<uint32_t, 16> ioffset_n16(byte_offset32, 8);
  auto ioffset_n16_view = ioffset_n16.select<16, 1>();

  simd_mask<32> mask_n32 = 1;
  simd_mask<16> mask_n16 = 1;

  simd<float, 32> slm;
  simd<float, 32> pass_thru;
  auto pass_thru_view = pass_thru.select<32, 1>();
  auto slm_view = slm.select<32, 1>();

  // Test SLM scatter using this plan:
  // 1) slm_scatter(offsets, vals): offsets/vals is simd or simd_view
  // 2) slm_scatter(offsets, vals, mask): offsets/vals is simd or simd_view
  // 3) slm_scatter(...): same as (1), (2) above, but with VS > 1.

  // 1) slm_scatter(offsets): offsets is simd or simd_view
  // CHECK-COUNT-13: call void @llvm.masked.scatter.v32f32.v32p3(<32 x float> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}})
  slm_scatter<float>(ioffset_n32, slm);
  slm_scatter<float, 32>(ioffset_n32_view, slm);
  slm_scatter<float, 32>(ioffset_n32, slm_view);
  slm_scatter<float, 32>(ioffset_n32_view, slm_view);
  slm_scatter<float, 32>(ioffset_n32_view.select<32, 1>(), slm);
  slm_scatter<float, 32>(ioffset_n32, slm_view.select<32, 1>());
  slm_scatter<float, 32>(ioffset_n32_view.select<32, 1>(),
                         slm_view.select<32, 1>());
  slm_scatter(ioffset_n32_view, slm);
  slm_scatter(ioffset_n32, slm_view);
  slm_scatter(ioffset_n32_view, slm_view);
  slm_scatter(ioffset_n32_view.select<32, 1>(), slm);
  slm_scatter(ioffset_n32, slm_view.select<32, 1>());
  slm_scatter(ioffset_n32_view.select<32, 1>(), slm_view.select<32, 1>());

  // CHECK-COUNT-13: call void @llvm.masked.scatter.v32f32.v32p3(<32 x float> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}})
  slm_scatter<float>(ioffset_n32, slm, props_align8);
  slm_scatter<float, 32>(ioffset_n32_view, slm, props_align8);
  slm_scatter<float, 32>(ioffset_n32, slm_view, props_align8);
  slm_scatter<float, 32>(ioffset_n32_view, slm_view, props_align8);
  slm_scatter<float, 32>(ioffset_n32_view.select<32, 1>(), slm, props_align8);
  slm_scatter<float, 32>(ioffset_n32, slm_view.select<32, 1>(), props_align8);
  slm_scatter<float, 32>(ioffset_n32_view.select<32, 1>(),
                         slm_view.select<32, 1>(), props_align8);
  slm_scatter(ioffset_n32_view, slm, props_align8);
  slm_scatter(ioffset_n32, slm_view, props_align8);
  slm_scatter(ioffset_n32_view, slm_view, props_align8);
  slm_scatter(ioffset_n32_view.select<32, 1>(), slm, props_align8);
  slm_scatter(ioffset_n32, slm_view.select<32, 1>(), props_align8);
  slm_scatter(ioffset_n32_view.select<32, 1>(), slm_view.select<32, 1>(),
              props_align8);

  // 2) slm_gather(offsets, mask): offsets is simd or simd_view
  // CHECK-COUNT-13: call void @llvm.masked.scatter.v32f32.v32p3(<32 x float> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}})
  slm_scatter<float>(ioffset_n32, slm, mask_n32);
  slm_scatter<float, 32>(ioffset_n32_view, slm, mask_n32);
  slm_scatter<float, 32>(ioffset_n32, slm_view, mask_n32);
  slm_scatter<float, 32>(ioffset_n32_view, slm_view, mask_n32);
  slm_scatter<float, 32>(ioffset_n32_view.select<32, 1>(), slm, mask_n32);
  slm_scatter<float, 32>(ioffset_n32, slm_view.select<32, 1>(), mask_n32);
  slm_scatter<float, 32>(ioffset_n32_view.select<32, 1>(),
                         slm_view.select<32, 1>(), mask_n32);
  slm_scatter(ioffset_n32_view, slm, mask_n32);
  slm_scatter(ioffset_n32, slm_view, mask_n32);
  slm_scatter(ioffset_n32_view, slm_view, mask_n32);
  slm_scatter(ioffset_n32_view.select<32, 1>(), slm, mask_n32);
  slm_scatter(ioffset_n32, slm_view.select<32, 1>(), mask_n32);
  slm_scatter(ioffset_n32_view.select<32, 1>(), slm_view.select<32, 1>(),
              mask_n32);

  // CHECK-COUNT-13: call void @llvm.masked.scatter.v32f32.v32p3(<32 x float> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}})
  slm_scatter<float>(ioffset_n32, slm, mask_n32, props_align8);
  slm_scatter<float, 32>(ioffset_n32_view, slm, mask_n32, props_align8);
  slm_scatter<float, 32>(ioffset_n32, slm_view, mask_n32, props_align8);
  slm_scatter<float, 32>(ioffset_n32_view, slm_view, mask_n32, props_align8);
  slm_scatter<float, 32>(ioffset_n32_view.select<32, 1>(), slm, mask_n32,
                         props_align8);
  slm_scatter<float, 32>(ioffset_n32, slm_view.select<32, 1>(), mask_n32,
                         props_align8);
  slm_scatter<float, 32>(ioffset_n32_view.select<32, 1>(),
                         slm_view.select<32, 1>(), mask_n32, props_align8);
  slm_scatter(ioffset_n32_view, slm, mask_n32, props_align8);
  slm_scatter(ioffset_n32, slm_view, mask_n32, props_align8);
  slm_scatter(ioffset_n32_view, slm_view, mask_n32, props_align8);
  slm_scatter(ioffset_n32_view.select<32, 1>(), slm, mask_n32, props_align8);
  slm_scatter(ioffset_n32, slm_view.select<32, 1>(), mask_n32, props_align8);
  slm_scatter(ioffset_n32_view.select<32, 1>(), slm_view.select<32, 1>(),
              mask_n32, props_align8);

  // 4) slm_gather(...): same as (1), (2), above, but with VS > 1.
  // CHECK-COUNT-52: call void @llvm.genx.lsc.store.slm.v16i1.v16i32.v32i32(<16 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, <32 x i32>{{[^)]+}}, i32 0)
  // 4a) check VS > 1. no 'mask' operand first.
  slm_scatter<float, 32, 2>(ioffset_n16, slm);
  slm_scatter<float, 32, 2>(ioffset_n16_view, slm);
  slm_scatter<float, 32, 2>(ioffset_n16, slm_view);
  slm_scatter<float, 32, 2>(ioffset_n16_view, slm_view);
  slm_scatter<float, 32, 2>(ioffset_n16_view.select<16, 1>(), slm);
  slm_scatter<float, 32, 2>(ioffset_n16, slm_view.select<32, 1>());
  slm_scatter<float, 32, 2>(ioffset_n16_view.select<16, 1>(),
                            slm_view.select<32, 1>());
  slm_scatter<2>(ioffset_n16_view, slm);
  slm_scatter<2>(ioffset_n16, slm_view);
  slm_scatter<2>(ioffset_n16_view, slm_view);
  slm_scatter<2>(ioffset_n16_view.select<16, 1>(), slm);
  slm_scatter<2>(ioffset_n16, slm_view.select<32, 1>());
  slm_scatter<2>(ioffset_n16_view.select<16, 1>(), slm_view.select<32, 1>());

  slm_scatter<float, 32, 2>(ioffset_n16, slm, props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16_view, slm, props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16, slm_view, props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16_view, slm_view, props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16_view.select<16, 1>(), slm,
                            props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16, slm_view.select<32, 1>(),
                            props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16_view.select<16, 1>(),
                            slm_view.select<32, 1>(), props_align4);

  slm_scatter<2>(ioffset_n16_view, slm, props_align4);
  slm_scatter<2>(ioffset_n16, slm_view, props_align4);
  slm_scatter<2>(ioffset_n16_view, slm_view, props_align4);
  slm_scatter<2>(ioffset_n16_view.select<16, 1>(), slm, props_align4);
  slm_scatter<2>(ioffset_n16, slm_view.select<32, 1>(), props_align4);
  slm_scatter<2>(ioffset_n16_view.select<16, 1>(), slm_view.select<32, 1>(),
                 props_align4);
  // 4b) check VS > 1. Pass the 'mask' operand this time.
  slm_scatter<float, 32, 2>(ioffset_n16, slm, mask_n16);
  slm_scatter<float, 32, 2>(ioffset_n16_view, slm, mask_n16);
  slm_scatter<float, 32, 2>(ioffset_n16, slm_view, mask_n16);
  slm_scatter<float, 32, 2>(ioffset_n16_view, slm_view, mask_n16);
  slm_scatter<float, 32, 2>(ioffset_n16_view.select<16, 1>(), slm, mask_n16);
  slm_scatter<float, 32, 2>(ioffset_n16, slm_view.select<32, 1>(), mask_n16);
  slm_scatter<float, 32, 2>(ioffset_n16_view.select<16, 1>(),
                            slm_view.select<32, 1>(), mask_n16);

  slm_scatter<2>(ioffset_n16_view, slm, mask_n16);
  slm_scatter<2>(ioffset_n16, slm_view, mask_n16);
  slm_scatter<2>(ioffset_n16_view, slm_view, mask_n16);
  slm_scatter<2>(ioffset_n16_view.select<16, 1>(), slm, mask_n16);
  slm_scatter<2>(ioffset_n16, slm_view.select<32, 1>(), mask_n16);
  slm_scatter<2>(ioffset_n16_view.select<16, 1>(), slm_view.select<32, 1>(),
                 mask_n16);

  slm_scatter<float, 32, 2>(ioffset_n16, slm, mask_n16, props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16_view, slm, mask_n16, props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16, slm_view, mask_n16, props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16_view, slm_view, mask_n16, props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16_view.select<16, 1>(), slm, mask_n16,
                            props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16, slm_view.select<32, 1>(), mask_n16,
                            props_align4);
  slm_scatter<float, 32, 2>(ioffset_n16_view.select<16, 1>(),
                            slm_view.select<32, 1>(), mask_n16, props_align4);

  slm_scatter<2>(ioffset_n16_view, slm, mask_n16, props_align4);
  slm_scatter<2>(ioffset_n16, slm_view, mask_n16, props_align4);
  slm_scatter<2>(ioffset_n16_view, slm_view, mask_n16, props_align4);
  slm_scatter<2>(ioffset_n16_view.select<16, 1>(), slm, mask_n16, props_align4);
  slm_scatter<2>(ioffset_n16, slm_view.select<32, 1>(), mask_n16, props_align4);
  slm_scatter<2>(ioffset_n16_view.select<16, 1>(), slm_view.select<32, 1>(),
                 mask_n16, props_align4);

  simd<uint32_t, 10> ioffset_n10(byte_offset32, 8);
  simd<float, 10> slm_n10;
  // Check special case to verify that for cases when N is not power of 2 llvm
  // intrinsic is used
  // CHECK-COUNT-1: call void @llvm.masked.scatter.v10f32.v10p3(<10 x float> {{[^)]+}}, <10 x ptr addrspace(3)> {{[^)]+}}, i32 4, <10 x i1> {{[^)]+}})
  slm_scatter(ioffset_n10, slm_n10);

  // Check a case to verify emulation for 64 bit data types
  // CHECK-COUNT-1: call <32 x i32> @llvm.masked.gather.v32i32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}}, <32 x i32> {{[^)]+}})
  // CHECK-COUNT-1: call <32 x i32> @llvm.masked.gather.v32i32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x i32> {{[^)]+}})
  auto slm_64 = slm_gather<int64_t>(ioffset_n32);
  // CHECK-COUNT-1: call void @llvm.masked.scatter.v32i32.v32p3(<32 x i32> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}})
  // CHECK-COUNT-1: call void @llvm.masked.scatter.v32i32.v32p3(<32 x i32> {{[^)]+}}, <32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}})
  slm_scatter<int64_t>(ioffset_n32, slm_64);
}
