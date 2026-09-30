// This test verifies that the unsupported combinations of the grf_size /
// grf_size_automatic kernel properties produce a compilation
// error.

// RUN: not %clangxx -fsycl -fsycl-device-only -fsycl-targets=intel_gpu_bmg_g21 -DCASE=0 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-AOT-512

// RUN: not %clangxx -fsycl -fsycl-device-only -DCASE=1 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ESIMD-512

// RUN: not %clangxx -fsycl -fsycl-device-only -fsycl-targets=intel_gpu_bmg_g21 -DCASE=2 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-AOT-ESIMD-256

// RUN: not %clangxx -fsycl -fsycl-device-only -fsycl-targets=intel_gpu_bmg_g21 -DCASE=3 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-AOT-ESIMD-AUTO

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace intelex = sycl::ext::intel::experimental;
namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  queue q;
  q.submit([&](handler &cgh) {
#if CASE == 0
    // grf_size<512> in a regular (non-ESIMD) kernel compiled AOT.
    syclexp::properties prop{intelex::grf_size<512>};
    cgh.parallel_for(range<1>(1), prop, [=](id<1>) {});
#elif CASE == 1
    // grf_size<512> in an ESIMD kernel.
    syclexp::properties prop{intelex::grf_size<512>};
    cgh.parallel_for(range<1>(1), prop, [=](id<1>) SYCL_ESIMD_KERNEL {});
#elif CASE == 2
    // grf_size<256> in an ESIMD kernel compiled AOT.
    syclexp::properties prop{intelex::grf_size<256>};
    cgh.parallel_for(range<1>(1), prop, [=](id<1>) SYCL_ESIMD_KERNEL {});
#elif CASE == 3
    // grf_size_automatic in an ESIMD kernel compiled AOT.
    syclexp::properties prop{intelex::grf_size_automatic};
    cgh.parallel_for(range<1>(1), prop, [=](id<1>) SYCL_ESIMD_KERNEL {});
#endif
  });
  return 0;
}

// CHECK-AOT-512: error: grf_size<512> is not supported with ahead-of-time compilation.{{.*}}Consider using the maximum_registers and maximum_registers_automatic properties.

// CHECK-ESIMD-512: error: grf_size<512> is not supported with ESIMD.{{.*}}Consider using the maximum_registers and maximum_registers_automatic properties.

// CHECK-AOT-ESIMD-256: error: grf_size<256> is not supported with ESIMD and ahead-of-time compilation.{{.*}}Consider using the maximum_registers and maximum_registers_automatic properties.

// CHECK-AOT-ESIMD-AUTO: error: grf_size_automatic is not supported with ESIMD and ahead-of-time-compilation.{{.*}}Consider using the maximum_registers and maximum_registers_automatic properties.
