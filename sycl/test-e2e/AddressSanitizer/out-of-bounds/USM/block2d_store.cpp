// REQUIRES: linux, gpu && level_zero && arch-intel_gpu_pvc
// RUN: %{build} %device_asan_flags -O2 -g -DOOB_SRC -Xspirv-translator -spirv-ext=+SPV_INTEL_2d_block_io -Xs "-options ' -cl-intel-enable-auto-large-GRF-mode'" -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-SRC %s
// RUN: %{build} %device_asan_flags -O2 -g -DOOB_DST -Xspirv-translator -spirv-ext=+SPV_INTEL_2d_block_io -Xs "-options ' -cl-intel-enable-auto-large-GRF-mode'" -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-DST %s

// Test that ASAN detects out-of-bounds access from 2D block store operations.

#include <sycl/detail/core.hpp>
#include <sycl/sub_group.hpp>
#include <sycl/usm.hpp>

typedef int int2 __attribute__((ext_vector_type(2)));

#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL __attribute__((convergent)) void
__spirv_Subgroup2DBlockStoreINTEL(int elem_size, int block_width,
                                  int block_height, int block_count,
                                  void *src_ptr, const void *base_ptr,
                                  int surface_width, int surface_height,
                                  int surface_pitch, int2 coord);
#else
void __spirv_Subgroup2DBlockStoreINTEL(int, int, int, int, void *, const void *,
                                       int, int, int, int2) {}
#endif

int main() {
  sycl::queue Q(sycl::gpu_selector_v);

  constexpr int Width = 16;
  constexpr int Height = 8;
#ifdef OOB_SRC
  auto *A = sycl::malloc_device<uint16_t>(Width * Height, Q);
#else
  auto *A = sycl::malloc_device<uint16_t>(Width * Height - 1, Q);
#endif

  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class block2d_oob_store>(
        sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> item) {
          if (item.get_sub_group().get_local_linear_id() == 0) {
#ifdef OOB_SRC
            uint16_t src_buf[Width * Height - 1] = {};
#else
            uint16_t src_buf[Width * Height] = {};
#endif
            // Store a 16-element-wide x 2-row block starting at coord (0, 0)
            // Row 1 at y=1 is out-of-bounds (surface_height=1, max y=0)
            int2 coord = {0, 0};
            __spirv_Subgroup2DBlockStoreINTEL(
                /*elem_size=*/2, /*block_width=*/16, /*block_height=*/8,
                /*block_count=*/1, /*src_ptr=*/src_buf, /*base_ptr=*/A,
                /*surface_width=*/Width * (int)sizeof(uint16_t),
                /*surface_height=*/Height,
                /*surface_pitch=*/Width * (int)sizeof(uint16_t), coord);
          }
        });
  });
  Q.wait();
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access
  // CHECK-SRC: {{READ of size .* at kernel <.*block2d_oob_store>}}
  // CHECK-DST: {{WRITE of size .* at kernel <.*block2d_oob_store>}}

  sycl::free(A, Q);
  return 0;
}
