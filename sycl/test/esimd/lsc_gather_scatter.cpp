// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -fsycl-esimd-force-stateless-mem -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
using namespace sycl;

void kernel(accessor<int, 1, access::mode::read_write, access::target::device>
                &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(int));
  simd<int, 32> v1(0, 1);

  lsc_prefetch<uint32_t, 1, lsc_data_size::default_size, cache_hint::cached,
               cache_hint::cached>(buf, offsets);

  auto v0 = lsc_gather<int>(buf, offsets);

  v0 = v0 + v1;

  lsc_scatter<int>(buf, offsets, v0);
}

// --- Negative tests.

// Incompatible mode (write).
SYCL_EXTERNAL void
kernel2(accessor<int, 1, access::mode::write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v;
  simd<uint32_t, 32> offset(0, 1);
  // CHECK: lsc_gather_scatter.cpp:38{{.*}}error: no matching function
  // function for call to 'lsc_gather'
  v = lsc_gather<int>(buf, offset);
}

// Incompatible mode (write).
SYCL_EXTERNAL void
kernel3(accessor<int, 1, access::mode::write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offset(0, 1);
  // CHECK: lsc_gather_scatter.cpp:48{{.*}}error: no matching function
  // function for call to 'lsc_prefetch'
  lsc_prefetch<int, 1, lsc_data_size::default_size, cache_hint::cached,
               cache_hint::cached>(buf, offset);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel4(accessor<int, 1, access::mode::read, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  simd<uint32_t, 32> offset(0, 1);
  // CHECK: lsc_gather_scatter.cpp:60{{.*}}error: no matching function
  // function for call to 'lsc_scatter'
  lsc_scatter<int>(buf, offset, v);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel5(local_accessor<const int, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  simd<uint32_t, 32> offset(0, 1);
  // CHECK: lsc_gather_scatter.cpp:70{{.*}}error: no matching function
  // function for call to 'lsc_scatter'
  lsc_scatter<int>(buf, offset, v);
}
