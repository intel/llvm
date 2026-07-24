// Check that the sycl_ext_oneapi_dot_accumulate extension (dot_acc / dp4a)
// lowers to the hardware `v_dot4` family on AMDGCN via the amdgcn dot builtins.
//
// Same-signedness dot products use `__builtin_amdgcn_sdot4` (needs dot1-insts,
// available on CDNA/RDNA2 but *not* RDNA3) and `__builtin_amdgcn_udot4` (needs
// dot7-insts). Mixed-signedness dot products use `__builtin_amdgcn_sudot4`
// (`v_dot4_i32_iu8`, needs dot8-insts), which exists on gfx11 (RDNA3) and gfx12
// only -- CDNA, including the gfx94x MI300 parts, lacks it and uses the scalar
// fallback for the mixed cases. All paths are guarded by
// `__builtin_amdgcn_is_invocable` so unsupported architectures fall back.
//
// REQUIRES: hip
//
// gfx90a (CDNA2, e.g. MI210/MI250): sdot4 + udot4, mixed falls back to scalar.
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend --offload-arch=gfx90a -S -Xclang -emit-llvm %s \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,GCN-DOT4
//
// gfx942 (CDNA3, MI300): same as gfx90a -- no sudot4 for the mixed cases.
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend --offload-arch=gfx942 -S -Xclang -emit-llvm %s \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,GCN-DOT4
//
// gfx1030 (RDNA2): sdot4 + udot4, mixed falls back to scalar (no dot8-insts).
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend --offload-arch=gfx1030 -S -Xclang -emit-llvm %s \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,GCN-DOT4
//
// gfx1100 (RDNA3): all signed/mixed cases lower to sudot4 (v_dot4_i32_iu8).
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend --offload-arch=gfx1100 -S -Xclang -emit-llvm %s \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,GCN-SUDOT
//
// gfx1200 (RDNA4): same as RDNA3 -- signed/mixed cases lower to sudot4.
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend --offload-arch=gfx1200 -S -Xclang -emit-llvm %s \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,GCN-SUDOT

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>

using namespace sycl::ext::oneapi;

// On CDNA/RDNA2 the signed case uses sdot4; RDNA3+/gfx12 lack v_dot4_i32_i8 and
// perform the signed dot product via sudot4 with both operands marked signed.
// GCN-DOT4: call{{.*}}@llvm.amdgcn.sdot4
// GCN-SUDOT: call{{.*}}@llvm.amdgcn.sudot4(i1 true, i32 %{{.*}}, i1 true,
SYCL_EXTERNAL int32_t test_ss(int32_t a, int32_t b, int32_t c) {
  return dot_acc(a, b, c);
}

// CHECK: call{{.*}}@llvm.amdgcn.udot4
SYCL_EXTERNAL int32_t test_uu(uint32_t a, uint32_t b, int32_t c) {
  return dot_acc(a, b, c);
}

// GCN-DOT4: call{{.*}}@llvm.amdgcn.sdot4
// GCN-SUDOT: call{{.*}}@llvm.amdgcn.sudot4(i1 true, i32 %{{.*}}, i1 true,
SYCL_EXTERNAL int32_t test_vec(sycl::vec<int8_t, 4> a, sycl::vec<int8_t, 4> b,
                               int32_t c) {
  return dot_acc(a, b, c);
}

// GCN-SUDOT: call{{.*}}@llvm.amdgcn.sudot4(i1 true, i32 %{{.*}}, i1 false,
SYCL_EXTERNAL int32_t test_su(int32_t a, uint32_t b, int32_t c) {
  return dot_acc(a, b, c);
}

// GCN-SUDOT: call{{.*}}@llvm.amdgcn.sudot4(i1 false, i32 %{{.*}}, i1 true,
SYCL_EXTERNAL int32_t test_us(uint32_t a, int32_t b, int32_t c) {
  return dot_acc(a, b, c);
}
