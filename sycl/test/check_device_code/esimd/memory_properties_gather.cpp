// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fno-sycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem=false -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATEFUL

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATELESS

// Checks ESIMD memory functions accepting compile time properties for gather
// APIs. NOTE: must be run in -O0, as optimizer optimizes away some of the code.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

using AccType = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write>;
using LocalAccType = sycl::local_accessor<double, 1>;
using LocalAccTypeInt = sycl::local_accessor<int, 1>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_gather(AccType &, LocalAccType &,
                                                   float *, int byte_offset32,
                                                   size_t byte_offset64);
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_slm_gather(int byte_offset32);

class EsimdFunctor {
public:
  AccType acc;
  LocalAccType local_acc;
  LocalAccTypeInt local_acc_int;
  float *ptr;
  int byte_offset32;
  size_t byte_offset64;
  void operator()() __attribute__((sycl_explicit_simd)) {
    test_gather(acc, local_acc, ptr, byte_offset32, byte_offset64);
    test_slm_gather(byte_offset32);
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

// CHECK-LABEL: define {{.*}} @_Z11test_gather{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_gather(AccType &acc, LocalAccType &local_acc, float *ptrf,
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

  // Test USM and ACC gather using this plan:
  // 1) gather(usm, offsets): offsets is simd or simd_view
  // 2) gather(usm, offsets, mask): offsets is simd or simd_view
  // 3) gather(usm, offsets, mask, pass_thru)
  // 4) gather(usm, ...): same as (1), (2), (3) above, but with VS > 1.
  // 5) gather(acc, offsets): offsets is simd or simd_view
  // 6) gather(acc, offsets, mask): offsets is simd or simd_view
  // 7) gather(acc, offsets, mask, pass_thru)
  // 8) gather(acc, ...): same as (5), (6), (7) above, but with VS > 1.
  // 9) gather(lacc, offsets): offsets is simd or simd_view
  // 10) gather(lacc, offsets, mask): offsets is simd or simd_view
  // 11) gather(lacc, offsets, mask, pass_thru)
  // 12) gather(lacc, ...): same as (9), (10), (11) above, but with VS > 1.

  // 1) gather(usm, offsets): offsets is simd or simd_view
  // CHECK-COUNT-10: call <32 x float> @llvm.masked.gather.v32f32.v32p4(<32 x ptr addrspace(4)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  usm = gather(ptrf, ioffset_n32);
  usm = gather<float, 32>(ptrf, ioffset_n32_view);
  usm = gather<float, 32>(ptrf, ioffset_n32_view.select<32, 1>());

  usm = gather(ptrf, loffset_n32);
  usm = gather<float, 32>(ptrf, loffset_n32_view);
  usm = gather<float, 32>(ptrf, loffset_n32_view.select<32, 1>());

  usm = gather(ptrf, ioffset_n32_view);
  usm = gather(ptrf, ioffset_n32_view.select<32, 1>());
  usm = gather(ptrf, loffset_n32_view);
  usm = gather(ptrf, loffset_n32_view.select<32, 1>());

  // CHECK-COUNT-10: call <32 x float> @llvm.masked.gather.v32f32.v32p4(<32 x ptr addrspace(4)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  usm = gather(ptrf, ioffset_n32, props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32_view, props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), props_align8);

  usm = gather(ptrf, loffset_n32, props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32_view, props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), props_align8);

  usm = gather(ptrf, ioffset_n32_view, props_align8);
  usm = gather(ptrf, ioffset_n32_view.select<32, 1>(), props_align8);
  usm = gather(ptrf, loffset_n32_view, props_align8);
  usm = gather(ptrf, loffset_n32_view.select<32, 1>(), props_align8);

  // 2) gather(usm, offsets, mask): offsets is simd or simd_view
  // CHECK-COUNT-10: call <32 x float> @llvm.masked.gather.v32f32.v32p4(<32 x ptr addrspace(4)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  usm = gather(ptrf, ioffset_n32, mask_n32);
  usm = gather<float, 32>(ptrf, ioffset_n32_view, mask_n32);
  usm = gather<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32);

  usm = gather(ptrf, loffset_n32, mask_n32);
  usm = gather<float, 32>(ptrf, loffset_n32_view, mask_n32);
  usm = gather<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), mask_n32);

  usm = gather(ptrf, ioffset_n32_view, mask_n32);
  usm = gather(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32);
  usm = gather(ptrf, loffset_n32_view, mask_n32);
  usm = gather(ptrf, loffset_n32_view.select<32, 1>(), mask_n32);

  // CHECK-COUNT-10: call <32 x float> @llvm.masked.gather.v32f32.v32p4(<32 x ptr addrspace(4)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  usm = gather(ptrf, ioffset_n32, mask_n32, props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32_view, mask_n32, props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32,
                          props_align8);

  usm = gather(ptrf, loffset_n32, mask_n32, props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32_view, mask_n32, props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), mask_n32,
                          props_align8);

  usm = gather(ptrf, ioffset_n32_view, mask_n32, props_align8);
  usm = gather(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32, props_align8);
  usm = gather(ptrf, loffset_n32_view, mask_n32, props_align8);
  usm = gather(ptrf, loffset_n32_view.select<32, 1>(), mask_n32, props_align8);

  // 3) gather(usm, offsets, mask, pass_thru)
  // CHECK-COUNT-26: call <32 x float> @llvm.masked.gather.v32f32.v32p4(<32 x ptr addrspace(4)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  usm = gather(ptrf, ioffset_n32, mask_n32, pass_thru);
  usm = gather<float, 32>(ptrf, ioffset_n32_view, mask_n32, pass_thru);
  usm = gather<float, 32>(ptrf, ioffset_n32, mask_n32, pass_thru_view);
  usm = gather<float, 32>(ptrf, ioffset_n32_view, mask_n32, pass_thru_view);
  usm = gather<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32,
                          pass_thru);
  usm = gather<float, 32>(ptrf, ioffset_n32, mask_n32,
                          pass_thru_view.select<32, 1>());
  usm = gather<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32,
                          pass_thru_view.select<32, 1>());

  usm = gather(ptrf, loffset_n32, mask_n32, pass_thru);
  usm = gather<float, 32>(ptrf, loffset_n32_view, mask_n32, pass_thru);
  usm = gather<float, 32>(ptrf, loffset_n32, mask_n32, pass_thru_view);
  usm = gather<float, 32>(ptrf, loffset_n32_view, mask_n32, pass_thru_view);
  usm = gather<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), mask_n32,
                          pass_thru);
  usm = gather<float, 32>(ptrf, loffset_n32, mask_n32,
                          pass_thru_view.select<32, 1>());
  usm = gather<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), mask_n32,
                          pass_thru_view.select<32, 1>());

  usm = gather(ptrf, ioffset_n32_view, mask_n32, pass_thru);
  usm = gather(ptrf, ioffset_n32, mask_n32, pass_thru_view);
  usm = gather(ptrf, ioffset_n32_view, mask_n32, pass_thru_view);
  usm = gather(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32, pass_thru);
  usm = gather(ptrf, ioffset_n32, mask_n32, pass_thru_view.select<32, 1>());
  usm = gather(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32,
               pass_thru_view.select<32, 1>());

  usm = gather(ptrf, loffset_n32_view, mask_n32, pass_thru);
  usm = gather(ptrf, loffset_n32, mask_n32, pass_thru_view);
  usm = gather(ptrf, loffset_n32_view, mask_n32, pass_thru_view);
  usm = gather(ptrf, loffset_n32_view.select<32, 1>(), mask_n32, pass_thru);
  usm = gather(ptrf, loffset_n32, mask_n32, pass_thru_view.select<32, 1>());
  usm = gather(ptrf, loffset_n32_view.select<32, 1>(), mask_n32,
               pass_thru_view.select<32, 1>());

  // CHECK-COUNT-26: call <32 x float> @llvm.masked.gather.v32f32.v32p4(<32 x ptr addrspace(4)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  usm = gather(ptrf, ioffset_n32, mask_n32, pass_thru, props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32_view, mask_n32, pass_thru,
                          props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32, mask_n32, pass_thru_view,
                          props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32_view, mask_n32, pass_thru_view,
                          props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32,
                          pass_thru, props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32, mask_n32,
                          pass_thru_view.select<32, 1>(), props_align8);
  usm = gather<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32,
                          pass_thru_view.select<32, 1>(), props_align8);

  usm = gather(ptrf, loffset_n32, mask_n32, pass_thru, props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32_view, mask_n32, pass_thru,
                          props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32, mask_n32, pass_thru_view,
                          props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32_view, mask_n32, pass_thru_view,
                          props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), mask_n32,
                          pass_thru, props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32, mask_n32,
                          pass_thru_view.select<32, 1>(), props_align8);
  usm = gather<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), mask_n32,
                          pass_thru_view.select<32, 1>(), props_align8);

  usm = gather(ptrf, ioffset_n32_view, mask_n32, pass_thru, props_align8);
  usm = gather(ptrf, ioffset_n32, mask_n32, pass_thru_view, props_align8);
  usm = gather(ptrf, ioffset_n32_view, mask_n32, pass_thru_view, props_align8);
  usm = gather(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32, pass_thru,
               props_align8);
  usm = gather(ptrf, ioffset_n32, mask_n32, pass_thru_view.select<32, 1>(),
               props_align8);
  usm = gather(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32,
               pass_thru_view.select<32, 1>(), props_align8);

  usm = gather(ptrf, loffset_n32_view, mask_n32, pass_thru, props_align8);
  usm = gather(ptrf, loffset_n32, mask_n32, pass_thru_view, props_align8);
  usm = gather(ptrf, loffset_n32_view, mask_n32, pass_thru_view, props_align8);
  usm = gather(ptrf, loffset_n32_view.select<32, 1>(), mask_n32, pass_thru,
               props_align8);
  usm = gather(ptrf, loffset_n32, mask_n32, pass_thru_view.select<32, 1>(),
               props_align8);
  usm = gather(ptrf, loffset_n32_view.select<32, 1>(), mask_n32,
               pass_thru_view.select<32, 1>(), props_align8);

  // 4) gather(usm, ...): same as (1), (2), (3) above, but with VS > 1.
  // CHECK-COUNT-92: call <32 x i32> @llvm.genx.lsc.load.merge.stateless.v32i32.v16i1.v16i64(<16 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, i32 0, <32 x i32> {{[^)]+}})
  // 4a) check VS > 1. no 'mask' operand first.
  usm = gather<float, 32, 2>(ptrf, ioffset_n16);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>());

  usm = gather<float, 32, 2>(ptrf, loffset_n16);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>());

  usm = gather<float, 32, 2>(ptrf, ioffset_n16, props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view, props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(),
                             props_align4);

  usm = gather<float, 32, 2>(ptrf, loffset_n16, props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view, props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(),
                             props_align4);

  usm = gather<2>(ptrf, ioffset_n16_view);
  usm = gather<2>(ptrf, ioffset_n16_view.select<16, 1>());

  usm = gather<2>(ptrf, loffset_n16_view);
  usm = gather<2>(ptrf, loffset_n16_view.select<16, 1>());

  usm = gather<2>(ptrf, ioffset_n16_view, props_align4);
  usm = gather<2>(ptrf, ioffset_n16_view.select<16, 1>(), props_align4);

  usm = gather<2>(ptrf, loffset_n16_view, props_align4);
  usm = gather<2>(ptrf, loffset_n16_view.select<16, 1>(), props_align4);

  // 4b) check VS > 1. Pass the 'mask' operand this time.
  usm = gather<float, 32, 2>(ptrf, ioffset_n16, mask_n16);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view, mask_n16);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16);

  usm = gather<float, 32, 2>(ptrf, loffset_n16, mask_n16);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view, mask_n16);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16);

  usm = gather<float, 32, 2>(ptrf, ioffset_n16, mask_n16, props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view, mask_n16, props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
                             props_align4);

  usm = gather<float, 32, 2>(ptrf, loffset_n16, mask_n16, props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view, mask_n16, props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
                             props_align4);

  usm = gather<2>(ptrf, ioffset_n16_view, mask_n16);
  usm = gather<2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16);

  usm = gather<2>(ptrf, loffset_n16_view, mask_n16);
  usm = gather<2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16);

  usm = gather<2>(ptrf, ioffset_n16_view, mask_n16, props_align4);
  usm =
      gather<2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16, props_align4);

  usm = gather<2>(ptrf, loffset_n16_view, mask_n16, props_align4);
  usm =
      gather<2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16, props_align4);

  // 4c) check VS > 1. Pass the 'mask' and 'pass_thru' operands.
  usm = gather<float, 32, 2>(ptrf, ioffset_n16, mask_n16, pass_thru);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view, mask_n16, pass_thru);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16, mask_n16, pass_thru_view);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view, mask_n16, pass_thru_view);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
                             pass_thru);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16, mask_n16,
                             pass_thru_view.select<32, 1>());
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
                             pass_thru_view.select<32, 1>());

  usm = gather<float, 32, 2>(ptrf, loffset_n16, mask_n16, pass_thru);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view, mask_n16, pass_thru);
  usm = gather<float, 32, 2>(ptrf, loffset_n16, mask_n16, pass_thru_view);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view, mask_n16, pass_thru_view);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
                             pass_thru);
  usm = gather<float, 32, 2>(ptrf, loffset_n16, mask_n16,
                             pass_thru_view.select<32, 1>());
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
                             pass_thru_view.select<32, 1>());

  usm = gather<float, 32, 2>(ptrf, ioffset_n16, mask_n16, pass_thru,
                             props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view, mask_n16, pass_thru,
                             props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16, mask_n16, pass_thru_view,
                             props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view, mask_n16, pass_thru_view,
                             props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
                             pass_thru, props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16, mask_n16,
                             pass_thru_view.select<32, 1>(), props_align4);
  usm = gather<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
                             pass_thru_view.select<32, 1>(), props_align4);

  usm = gather<float, 32, 2>(ptrf, loffset_n16, mask_n16, pass_thru,
                             props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view, mask_n16, pass_thru,
                             props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16, mask_n16, pass_thru_view,
                             props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view, mask_n16, pass_thru_view,
                             props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
                             pass_thru, props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16, mask_n16,
                             pass_thru_view.select<32, 1>(), props_align4);
  usm = gather<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
                             pass_thru_view.select<32, 1>(), props_align4);

  usm = gather<2>(ptrf, ioffset_n16_view, mask_n16, pass_thru);
  usm = gather<2>(ptrf, ioffset_n16, mask_n16, pass_thru_view);
  usm = gather<2>(ptrf, ioffset_n16_view, mask_n16, pass_thru_view);
  usm = gather<2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16, pass_thru);
  usm = gather<2>(ptrf, ioffset_n16, mask_n16, pass_thru_view.select<32, 1>());
  usm = gather<2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
                  pass_thru_view.select<32, 1>());

  usm = gather<2>(ptrf, loffset_n16_view, mask_n16, pass_thru);
  usm = gather<2>(ptrf, loffset_n16, mask_n16, pass_thru_view);
  usm = gather<2>(ptrf, loffset_n16_view, mask_n16, pass_thru_view);
  usm = gather<2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16, pass_thru);
  usm = gather<2>(ptrf, loffset_n16, mask_n16, pass_thru_view.select<32, 1>());
  usm = gather<2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
                  pass_thru_view.select<32, 1>());

  usm = gather<2>(ptrf, ioffset_n16_view, mask_n16, pass_thru, props_align4);
  usm = gather<2>(ptrf, ioffset_n16, mask_n16, pass_thru_view, props_align4);
  usm =
      gather<2>(ptrf, ioffset_n16_view, mask_n16, pass_thru_view, props_align4);
  usm = gather<2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16, pass_thru,
                  props_align4);
  usm = gather<2>(ptrf, ioffset_n16, mask_n16, pass_thru_view.select<32, 1>(),
                  props_align4);
  usm = gather<2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
                  pass_thru_view.select<32, 1>(), props_align4);

  usm = gather<2>(ptrf, loffset_n16_view, mask_n16, pass_thru, props_align4);
  usm = gather<2>(ptrf, loffset_n16, mask_n16, pass_thru_view, props_align4);
  usm =
      gather<2>(ptrf, loffset_n16_view, mask_n16, pass_thru_view, props_align4);
  usm = gather<2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16, pass_thru,
                  props_align4);
  usm = gather<2>(ptrf, loffset_n16, mask_n16, pass_thru_view.select<32, 1>(),
                  props_align4);
  usm = gather<2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
                  pass_thru_view.select<32, 1>(), props_align4);

  // 5) gather(acc, offsets): offsets is simd or simd_view
  // CHECK-STATEFUL-COUNT-12: call <32 x float> @llvm.genx.gather.masked.scaled2.v32f32.v32i32.v32i1(i32 2, i16 0, i32 {{[^)]+}}, i32 {{[^)]+}}, <32 x i32> {{[^)]+}}, <32 x i1> {{[^)]+}})
  // CHECK-STATEFUL-COUNT-28: call <32 x i32> @llvm.genx.lsc.load.merge.bti.v32i32.v32i1.v32i32(<32 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i32> {{[^)]+}}, i32 {{[^)]+}}, <32 x i32> {{[^)]+}})
  // CHECK-STATELESS-COUNT-40: call <32 x float> @llvm.masked.gather.v32f32.v32p4(<32 x ptr addrspace(4)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  acc_res = gather<float>(acc, ioffset_n32);
  acc_res = gather<float, 32>(acc, ioffset_n32_view);
  acc_res = gather<float, 32>(acc, ioffset_n32_view.select<32, 1>());
  acc_res = gather<float>(acc, ioffset_n32, props_align4);
  acc_res = gather<float, 32>(acc, ioffset_n32_view, props_align4);
  acc_res =
      gather<float, 32>(acc, ioffset_n32_view.select<32, 1>(), props_align4);

  // 6) gather(acc, offsets, mask): offsets is simd or simd_view
  acc_res = gather<float>(acc, ioffset_n32, mask_n32);
  acc_res = gather<float, 32>(acc, ioffset_n32_view, mask_n32);
  acc_res = gather<float, 32>(acc, ioffset_n32_view.select<32, 1>(), mask_n32);

  acc_res = gather<float>(acc, ioffset_n32, mask_n32, props_align4);
  acc_res = gather<float, 32>(acc, ioffset_n32_view, mask_n32, props_align4);
  acc_res = gather<float, 32>(acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                              props_align4);

  // 7) gather(acc, offsets, mask, pass_thru)
  acc_res = gather<float>(acc, ioffset_n32, mask_n32, pass_thru);
  acc_res = gather<float, 32>(acc, ioffset_n32_view, mask_n32, pass_thru);
  acc_res = gather<float, 32>(acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                              pass_thru);

  acc_res = gather<float>(acc, ioffset_n32, mask_n32, pass_thru, props_align4);
  acc_res = gather<float, 32>(acc, ioffset_n32_view, mask_n32, pass_thru,
                              props_align4);
  acc_res = gather<float, 32>(acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                              pass_thru, props_align4);

  acc_res = gather<float, 32>(acc, ioffset_n32, mask_n32, pass_thru_view);
  acc_res = gather<float, 32>(acc, ioffset_n32_view, mask_n32, pass_thru_view);
  acc_res = gather<float, 32>(acc, ioffset_n32, mask_n32, pass_thru_view,
                              props_align4);
  acc_res = gather<float, 32>(acc, ioffset_n32_view, mask_n32, pass_thru_view,
                              props_align4);
  acc_res = gather<float, 32>(acc, ioffset_n32, mask_n32,
                              pass_thru_view.select<32, 1>());
  acc_res = gather<float, 32>(acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                              pass_thru_view.select<32, 1>());
  acc_res = gather<float, 32>(acc, ioffset_n32, mask_n32,
                              pass_thru_view.select<32, 1>(), props_align4);
  acc_res = gather<float, 32>(acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                              pass_thru_view.select<32, 1>(), props_align4);

  acc_res = gather(acc, ioffset_n32, mask_n32, pass_thru);
  acc_res = gather(acc, ioffset_n32_view, mask_n32, pass_thru);
  acc_res = gather(acc, ioffset_n32_view.select<32, 1>(), mask_n32, pass_thru);

  acc_res = gather(acc, ioffset_n32, mask_n32, pass_thru, props_align4);
  acc_res = gather(acc, ioffset_n32_view, mask_n32, pass_thru, props_align4);
  acc_res = gather(acc, ioffset_n32_view.select<32, 1>(), mask_n32, pass_thru,
                   props_align4);

  acc_res = gather(acc, ioffset_n32, mask_n32, pass_thru_view);
  acc_res = gather(acc, ioffset_n32_view, mask_n32, pass_thru_view);
  acc_res = gather(acc, ioffset_n32, mask_n32, pass_thru_view, props_align4);
  acc_res =
      gather(acc, ioffset_n32_view, mask_n32, pass_thru_view, props_align4);
  acc_res = gather(acc, ioffset_n32, mask_n32, pass_thru_view.select<32, 1>());
  acc_res = gather(acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                   pass_thru_view.select<32, 1>());
  acc_res = gather(acc, ioffset_n32, mask_n32, pass_thru_view.select<32, 1>(),
                   props_align4);
  acc_res = gather(acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                   pass_thru_view.select<32, 1>(), props_align4);

  // 8) gather(ac, ...): same as (5), (6), (7) above, but with VS > 1.
  // CHECK-STATEFUL-COUNT-38: call <32 x i32> @llvm.genx.lsc.load.merge.bti.v32i32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, i32 {{[^)]+}}, <32 x i32> {{[^)]+}})
  // CHECK-STATELESS-COUNT-38: call <32 x i32> @llvm.genx.lsc.load.merge.stateless.v32i32.v16i1.v16i64(<16 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, i32 0, <32 x i32> {{[^)]+}})
  acc_res = gather<float, 32, 2>(acc, ioffset_n16);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16, props_align4);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view, props_align4);

  acc_res = gather<float, 32, 2>(acc, ioffset_n16, mask_n16);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view, mask_n16);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16, mask_n16, props_align4);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view, mask_n16, props_align4);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16, mask_n16, pass_thru);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view, mask_n16, pass_thru);
  acc_res =
      gather<float, 32, 2>(acc, ioffset_n16, mask_n16, pass_thru, props_align4);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view, mask_n16, pass_thru,
                                 props_align4);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16, mask_n16, pass_thru_view);
  acc_res =
      gather<float, 32, 2>(acc, ioffset_n16_view, mask_n16, pass_thru_view);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16, mask_n16, pass_thru_view,
                                 props_align4);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view, mask_n16,
                                 pass_thru_view, props_align4);

  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>());
  acc_res =
      gather<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(), props_align4);
  acc_res =
      gather<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(), mask_n16);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, props_align4);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, pass_thru);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, pass_thru, props_align4);
  acc_res = gather<float, 32, 2>(acc, ioffset_n16, mask_n16,
                                 pass_thru_view.select<32, 1>());
  acc_res = gather<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, pass_thru_view.select<32, 1>());
  acc_res = gather<float, 32, 2>(acc, ioffset_n16, mask_n16,
                                 pass_thru_view.select<32, 1>(), props_align4);
  acc_res =
      gather<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                           pass_thru_view.select<32, 1>(), props_align4);
  acc_res = gather<2>(acc, ioffset_n16_view, mask_n16, pass_thru);

  acc_res = gather<2>(acc, ioffset_n16_view, mask_n16, pass_thru, props_align4);
  acc_res = gather<2>(acc, ioffset_n16, mask_n16, pass_thru_view);
  acc_res = gather<2>(acc, ioffset_n16_view, mask_n16, pass_thru_view);
  acc_res = gather<2>(acc, ioffset_n16, mask_n16, pass_thru_view, props_align4);
  acc_res =
      gather<2>(acc, ioffset_n16_view, mask_n16, pass_thru_view, props_align4);

  acc_res =
      gather<2>(acc, ioffset_n16_view.select<16, 1>(), mask_n16, pass_thru);
  acc_res = gather<2>(acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru, props_align4);
  acc_res =
      gather<2>(acc, ioffset_n16, mask_n16, pass_thru_view.select<32, 1>());
  acc_res = gather<2>(acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru_view.select<32, 1>());
  acc_res = gather<2>(acc, ioffset_n16, mask_n16,
                      pass_thru_view.select<32, 1>(), props_align4);
  acc_res = gather<2>(acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru_view.select<32, 1>(), props_align4);

  // 9) gather(lacc, offsets): offsets is simd or simd_view
  // CHECK-COUNT-38: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  acc_res = gather<float>(local_acc, ioffset_n32);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view.select<32, 1>());

  acc_res = gather<float>(local_acc, ioffset_n32, props_align4);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view, props_align4);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(),
                              props_align4);

  // 10) gather(lacc, offsets, mask): offsets is simd or simd_view
  acc_res = gather<float>(local_acc, ioffset_n32, mask_n32);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view, mask_n32);
  acc_res =
      gather<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(), mask_n32);

  acc_res = gather<float>(local_acc, ioffset_n32, mask_n32, props_align4);
  acc_res =
      gather<float, 32>(local_acc, ioffset_n32_view, mask_n32, props_align4);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(),
                              mask_n32, props_align4);

  // 11) gather(lacc, offsets, mask, pass_thru)
  acc_res = gather<float>(local_acc, ioffset_n32, mask_n32, pass_thru);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view, mask_n32, pass_thru);
  acc_res =
      gather<float>(local_acc, ioffset_n32, mask_n32, pass_thru, props_align4);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view, mask_n32, pass_thru,
                              props_align4);

  acc_res = gather<float, 32>(local_acc, ioffset_n32, mask_n32, pass_thru_view);
  acc_res =
      gather<float, 32>(local_acc, ioffset_n32_view, mask_n32, pass_thru_view);
  acc_res = gather<float, 32>(local_acc, ioffset_n32, mask_n32, pass_thru_view,
                              props_align4);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view, mask_n32,
                              pass_thru_view, props_align4);

  acc_res = gather<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(),
                              mask_n32, pass_thru);
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(),
                              mask_n32, pass_thru, props_align4);

  acc_res = gather<float, 32>(local_acc, ioffset_n32, mask_n32,
                              pass_thru_view.select<32, 1>());
  acc_res = gather<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(),
                              mask_n32, pass_thru_view.select<32, 1>());
  acc_res = gather<float, 32>(local_acc, ioffset_n32, mask_n32,
                              pass_thru_view.select<32, 1>(), props_align4);
  acc_res =
      gather<float, 32>(local_acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                        pass_thru_view.select<32, 1>(), props_align4);

  acc_res = gather(local_acc, ioffset_n32_view, mask_n32, pass_thru);

  acc_res =
      gather(local_acc, ioffset_n32_view, mask_n32, pass_thru, props_align4);

  acc_res = gather(local_acc, ioffset_n32, mask_n32, pass_thru_view);
  acc_res = gather(local_acc, ioffset_n32_view, mask_n32, pass_thru_view);
  acc_res =
      gather(local_acc, ioffset_n32, mask_n32, pass_thru_view, props_align4);
  acc_res = gather(local_acc, ioffset_n32_view, mask_n32, pass_thru_view,
                   props_align4);

  acc_res =
      gather(local_acc, ioffset_n32_view.select<32, 1>(), mask_n32, pass_thru);
  acc_res = gather(local_acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                   pass_thru, props_align4);

  acc_res =
      gather(local_acc, ioffset_n32, mask_n32, pass_thru_view.select<32, 1>());
  acc_res = gather(local_acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                   pass_thru_view.select<32, 1>());
  acc_res = gather(local_acc, ioffset_n32, mask_n32,
                   pass_thru_view.select<32, 1>(), props_align4);
  acc_res = gather(local_acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                   pass_thru_view.select<32, 1>(), props_align4);

  // 12) gather(lacc, ...): same as (9), (10), (11) above, but with VS > 1.
  // CHECK-COUNT-39: call <32 x i32> @llvm.genx.lsc.load.merge.slm.v32i32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, i32 0, <32 x i32> {{[^)]+}})
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16, props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view, props_align4);

  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16, mask_n16);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view, mask_n16);
  acc_res =
      gather<float, 32, 2>(local_acc, ioffset_n16, mask_n16, props_align4);
  acc_res =
      gather<float, 32, 2>(local_acc, ioffset_n16_view, mask_n16, props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16, mask_n16, pass_thru);
  acc_res =
      gather<float, 32, 2>(local_acc, ioffset_n16_view, mask_n16, pass_thru);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16, mask_n16, pass_thru,
                                 props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view, mask_n16,
                                 pass_thru, props_align4);
  acc_res =
      gather<float, 32, 2>(local_acc, ioffset_n16, mask_n16, pass_thru_view);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view, mask_n16,
                                 pass_thru_view);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16, mask_n16,
                                 pass_thru_view, props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view, mask_n16,
                                 pass_thru_view, props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16);

  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>());
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                                 props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, pass_thru);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, pass_thru, props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16, mask_n16,
                                 pass_thru_view.select<32, 1>());
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, pass_thru_view.select<32, 1>());
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16, mask_n16,
                                 pass_thru_view.select<32, 1>(), props_align4);
  acc_res = gather<float, 32, 2>(local_acc, ioffset_n16_view.select<16, 1>(),
                                 mask_n16, pass_thru_view.select<32, 1>(),
                                 props_align4);

  acc_res = gather<2>(local_acc, ioffset_n16_view, mask_n16, pass_thru);
  acc_res =
      gather<2>(local_acc, ioffset_n16_view, mask_n16, pass_thru, props_align4);
  acc_res = gather<2>(local_acc, ioffset_n16, mask_n16, pass_thru_view);
  acc_res = gather<2>(local_acc, ioffset_n16_view, mask_n16, pass_thru_view);
  acc_res =
      gather<2>(local_acc, ioffset_n16, mask_n16, pass_thru_view, props_align4);
  acc_res = gather<2>(local_acc, ioffset_n16_view, mask_n16, pass_thru_view,
                      props_align4);

  acc_res = gather<2>(local_acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru);
  acc_res = gather<2>(local_acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru, props_align4);
  acc_res = gather<2>(local_acc, ioffset_n16, mask_n16,
                      pass_thru_view.select<32, 1>());
  acc_res = gather<2>(local_acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru_view.select<32, 1>());
  acc_res = gather<2>(local_acc, ioffset_n16, mask_n16,
                      pass_thru_view.select<32, 1>(), props_align4);
  acc_res = gather<2>(local_acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru_view.select<32, 1>(), props_align4);

  // Validate that a new API doesn't conflict with the old API.
  // CHECK-COUNT-2: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  acc_res = gather<float, 32>(local_acc, ioffset_n32, 0);
  acc_res = gather<float, 32>(local_acc, ioffset_n32, 0, mask_n32);
}

// CHECK-LABEL: define {{.*}} @_Z15test_slm_gather{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_slm_gather(int byte_offset32) {

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

  // Test SLM gather using this plan:
  // 1) slm_gather(offsets): offsets is simd or simd_view
  // 2) slm_gather(offsets, mask): offsets is simd or simd_view
  // 3) slm_gather( offsets, mask, pass_thru)
  // 4) slm_gather(...): same as (1), (2), (3) above, but with VS > 1.

  // 1) slm_gather(offsets): offsets is simd or simd_view
  // CHECK-COUNT-3: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  slm = slm_gather<float>(ioffset_n32);
  slm = slm_gather<float, 32>(ioffset_n32_view);
  slm = slm_gather<float, 32>(ioffset_n32_view.select<32, 1>());

  // CHECK-COUNT-3: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  slm = slm_gather<float>(ioffset_n32, props_align8);
  slm = slm_gather<float, 32>(ioffset_n32_view, props_align8);
  slm = slm_gather<float, 32>(ioffset_n32_view.select<32, 1>(), props_align8);

  // 2) slm_gather(offsets, mask): offsets is simd or simd_view
  // CHECK-COUNT-3: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  slm = slm_gather<float>(ioffset_n32, mask_n32);
  slm = slm_gather<float, 32>(ioffset_n32_view, mask_n32);
  slm = slm_gather<float, 32>(ioffset_n32_view.select<32, 1>(), mask_n32);

  // CHECK-COUNT-3: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  slm = slm_gather<float>(ioffset_n32, mask_n32, props_align8);
  slm = slm_gather<float, 32>(ioffset_n32_view, mask_n32, props_align8);
  slm = slm_gather<float, 32>(ioffset_n32_view.select<32, 1>(), mask_n32,
                              props_align8);

  // 3) slm_gather(offsets, mask, pass_thru)
  // CHECK-COUNT-13: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 4, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  slm = slm_gather<float>(ioffset_n32, mask_n32, pass_thru);
  slm = slm_gather<float, 32>(ioffset_n32_view, mask_n32, pass_thru);
  slm = slm_gather<float, 32>(ioffset_n32, mask_n32, pass_thru_view);
  slm = slm_gather<float, 32>(ioffset_n32_view, mask_n32, pass_thru_view);
  slm = slm_gather<float, 32>(ioffset_n32_view.select<32, 1>(), mask_n32,
                              pass_thru);
  slm = slm_gather<float, 32>(ioffset_n32, mask_n32,
                              pass_thru_view.select<32, 1>());
  slm = slm_gather<float, 32>(ioffset_n32_view.select<32, 1>(), mask_n32,
                              pass_thru_view.select<32, 1>());

  slm = slm_gather(ioffset_n32_view, mask_n32, pass_thru);
  slm = slm_gather(ioffset_n32, mask_n32, pass_thru_view);
  slm = slm_gather(ioffset_n32_view, mask_n32, pass_thru_view);
  slm = slm_gather(ioffset_n32_view.select<32, 1>(), mask_n32, pass_thru);
  slm = slm_gather(ioffset_n32, mask_n32, pass_thru_view.select<32, 1>());
  slm = slm_gather(ioffset_n32_view.select<32, 1>(), mask_n32,
                   pass_thru_view.select<32, 1>());

  // CHECK-COUNT-13: call <32 x float> @llvm.masked.gather.v32f32.v32p3(<32 x ptr addrspace(3)> {{[^)]+}}, i32 8, <32 x i1> {{[^)]+}}, <32 x float> {{[^)]+}})
  slm = slm_gather<float>(ioffset_n32, mask_n32, pass_thru, props_align8);
  slm = slm_gather<float, 32>(ioffset_n32_view, mask_n32, pass_thru,
                              props_align8);
  slm = slm_gather<float, 32>(ioffset_n32, mask_n32, pass_thru_view,
                              props_align8);
  slm = slm_gather<float, 32>(ioffset_n32_view, mask_n32, pass_thru_view,
                              props_align8);
  slm = slm_gather<float, 32>(ioffset_n32_view.select<32, 1>(), mask_n32,
                              pass_thru, props_align8);
  slm = slm_gather<float, 32>(ioffset_n32, mask_n32,
                              pass_thru_view.select<32, 1>(), props_align8);
  slm = slm_gather<float, 32>(ioffset_n32_view.select<32, 1>(), mask_n32,
                              pass_thru_view.select<32, 1>(), props_align8);

  slm = slm_gather(ioffset_n32_view, mask_n32, pass_thru, props_align8);
  slm = slm_gather(ioffset_n32, mask_n32, pass_thru_view, props_align8);
  slm = slm_gather(ioffset_n32_view, mask_n32, pass_thru_view, props_align8);
  slm = slm_gather(ioffset_n32_view.select<32, 1>(), mask_n32, pass_thru,
                   props_align8);
  slm = slm_gather(ioffset_n32, mask_n32, pass_thru_view.select<32, 1>(),
                   props_align8);
  slm = slm_gather(ioffset_n32_view.select<32, 1>(), mask_n32,
                   pass_thru_view.select<32, 1>(), props_align8);

  // 4) slm_gather(...): same as (1), (2), (3) above, but with VS > 1.
  // CHECK-COUNT-38: call <32 x i32> @llvm.genx.lsc.load.merge.slm.v32i32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, i32 0, <32 x i32> {{[^)]+}})
  // 4a) check VS > 1. no 'mask' operand first.
  slm = slm_gather<float, 32, 2>(ioffset_n16);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view.select<16, 1>());

  slm = slm_gather<float, 32, 2>(ioffset_n16, props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view, props_align4);
  slm =
      slm_gather<float, 32, 2>(ioffset_n16_view.select<16, 1>(), props_align4);

  // 4b) check VS > 1. Pass the 'mask' operand this time.
  slm = slm_gather<float, 32, 2>(ioffset_n16, mask_n16);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view, mask_n16);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view.select<16, 1>(), mask_n16);

  slm = slm_gather<float, 32, 2>(ioffset_n16, mask_n16, props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view, mask_n16, props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view.select<16, 1>(), mask_n16,
                                 props_align4);

  // 4c) check VS > 1. Pass the 'mask' and 'pass_thru' operands.
  slm = slm_gather<float, 32, 2>(ioffset_n16, mask_n16, pass_thru);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view, mask_n16, pass_thru);
  slm = slm_gather<float, 32, 2>(ioffset_n16, mask_n16, pass_thru_view);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view, mask_n16, pass_thru_view);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view.select<16, 1>(), mask_n16,
                                 pass_thru);
  slm = slm_gather<float, 32, 2>(ioffset_n16, mask_n16,
                                 pass_thru_view.select<32, 1>());
  slm = slm_gather<float, 32, 2>(ioffset_n16_view.select<16, 1>(), mask_n16,
                                 pass_thru_view.select<32, 1>());
  slm =
      slm_gather<float, 32, 2>(ioffset_n16, mask_n16, pass_thru, props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view, mask_n16, pass_thru,
                                 props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16, mask_n16, pass_thru_view,
                                 props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view, mask_n16, pass_thru_view,
                                 props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view.select<16, 1>(), mask_n16,
                                 pass_thru, props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16, mask_n16,
                                 pass_thru_view.select<32, 1>(), props_align4);
  slm = slm_gather<float, 32, 2>(ioffset_n16_view.select<16, 1>(), mask_n16,
                                 pass_thru_view.select<32, 1>(), props_align4);

  slm = slm_gather<2>(ioffset_n16_view, mask_n16, pass_thru);
  slm = slm_gather<2>(ioffset_n16, mask_n16, pass_thru_view);
  slm = slm_gather<2>(ioffset_n16_view, mask_n16, pass_thru_view);
  slm = slm_gather<2>(ioffset_n16_view.select<16, 1>(), mask_n16, pass_thru);
  slm = slm_gather<2>(ioffset_n16, mask_n16, pass_thru_view.select<32, 1>());
  slm = slm_gather<2>(ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru_view.select<32, 1>());
  slm = slm_gather<2>(ioffset_n16_view, mask_n16, pass_thru, props_align4);
  slm = slm_gather<2>(ioffset_n16, mask_n16, pass_thru_view, props_align4);
  slm = slm_gather<2>(ioffset_n16_view, mask_n16, pass_thru_view, props_align4);
  slm = slm_gather<2>(ioffset_n16_view.select<16, 1>(), mask_n16, pass_thru,
                      props_align4);
  slm = slm_gather<2>(ioffset_n16, mask_n16, pass_thru_view.select<32, 1>(),
                      props_align4);
  slm = slm_gather<2>(ioffset_n16_view.select<16, 1>(), mask_n16,
                      pass_thru_view.select<32, 1>(), props_align4);
}