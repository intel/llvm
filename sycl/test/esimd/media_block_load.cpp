// RUN: not %clangxx -fsycl -fsycl-device-only -S %s -o /dev/null 2>&1 | FileCheck %s
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

int main(void) {
  constexpr unsigned Size = 256;
  constexpr unsigned VL = 9; // change to 8 to make the test compile just fine

  constexpr unsigned GroupSize = 2;

  int A[Size];
  sycl::image<2> imgA(A, image_channel_order::rgba,
                      image_channel_type::unsigned_int32,
                      range<2>{Size / 4, 1});

  // We need that many workitems
  range<1> GlobalRange{(Size / VL)};

  // Number of workitems in a workgroup
  range<1> LocalRange{GroupSize};

  queue q;

  q.submit([&](handler &cgh) {
    auto accA = imgA.get_access<uint4, access::mode::read>(cgh);

    cgh.parallel_for<class Test>(
        GlobalRange * LocalRange, [=](id<1> i) SYCL_ESIMD_KERNEL {
          using namespace sycl::ext::intel::esimd;

          constexpr int ESIZE = sizeof(int);
          int x = i * ESIZE * VL;
          int y = 0;

          simd<int, VL> va;
          auto va_ref = va.bit_cast_view<int, 1, VL>();
          // CHECK: {{.*}}error:{{.*}}static assertion failed due to requirement 'detail::isPowerOf2(9)': N must be a power of 2{{.*}}
          va_ref = media_block_load<int, 1, VL>(accA, x, y);
        });
  });

  return 0;
}
