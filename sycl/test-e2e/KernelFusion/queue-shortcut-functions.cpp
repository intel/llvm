// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run} %t.out 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=ERROR

// Test fusion with queue shortcut functions being involved.

#include <sycl/sycl.hpp>

using namespace sycl;

template <int FusionStartPoint, int KernelNum> class Kernel;

template <int FusionStartPoint> void test() {
  static_assert(0 <= FusionStartPoint && FusionStartPoint < 3,
                "Invalid fusion start point");

  constexpr size_t size = 1024;
  constexpr float value = 10;
  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  std::array<float, size> h;
  h.fill(0);
  auto *ptr0 = sycl::malloc_device<float>(size, q);
  auto *ptr1 = sycl::malloc_device<float>(size, q);

  {
    range<1> r{size};

    ext::codeplay::experimental::fusion_wrapper fw{q};

    if constexpr (FusionStartPoint == 0) {
      fw.start_fusion();
    }

    // ptr0(x) = value
    auto e0 = q.parallel_for<Kernel<FusionStartPoint, 0>>(
        r, [=](sycl::id<1> i) { ptr0[i] = value; });
    // ptr1(x) = value / 2
    auto e1 = q.parallel_for<Kernel<FusionStartPoint, 1>>(
        r, [=](sycl::id<1> i) { ptr1[i] = value / 2; });

    if constexpr (FusionStartPoint == 1) {
      fw.start_fusion();
    }

    // ptr0(x) = value / 2 if x < size / 2 else value
    auto e2 = q.memcpy(ptr0, ptr1, sizeof(float) * size / 2, {e0, e1});

    if constexpr (FusionStartPoint == 2) {
      fw.start_fusion();
    }

    // ptr0(x) = value / 2 + 1 if x < size / 2 else value + 1
    auto e3 = q.parallel_for<Kernel<FusionStartPoint, 2>>(
        r, e2, [=](sycl::id<1> i) { ptr0[i]++; });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    // Copyback
    q.memcpy(h.data(), ptr0, sizeof(float) * size, e3).wait();
  }

  sycl::free(ptr0, q);
  sycl::free(ptr1, q);

  assert(std::all_of(h.begin(), h.begin() + size / 2,
                     [=](float f) { return f == value / 2 + 1; }) &&
         "ERROR");
  assert(std::all_of(h.begin() + size / 2, h.end(),
                     [=](float f) { return f == value + 1; }) &&
         "ERROR");
}

int main() {
  std::cerr << "FusionStartPoint = 0:\n";
  // COM: memcpy leads to a CG being created as it depends on CGs not producing
  // a PI event (coming from the CGs to be fused), so not safe to bypass. Fusion
  // should be cancelled as a dependency with an event to be fused is found.

  // CHECK:      FusionStartPoint = 0:
  // CHECK-NEXT: WARNING: Not fusing 'copy usm' command group. Can only fuse device kernel command groups.
  // CHECK-NEXT: WARNING: Aborting fusion because synchronization with one of the kernels in the fusion list was requested
  test<0>();

  std::cerr << "FusionStartPoint = 1:\n";
  // COM: memcpy does not create CG, memory manager handles the operation
  // instead. As no dependency with a CG to be fused is found, events are issued
  // as usual and fusion takes place.

  // CHECK-NEXT: FusionStartPoint = 1:
  // CHECK-NEXT: JIT DEBUG: Compiling new kernel, no suitable cached kernel found
  test<1>();

  std::cerr << "FusionStartPoint = 2:\n";
  // COM: Same as above.

  // CHECK-NEXT: FusionStartPoint = 2:
  // CHECK-NEXT: JIT DEBUG: Compiling new kernel, no suitable cached kernel found
  test<2>();
}
