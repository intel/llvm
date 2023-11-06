// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -fsycl-esimd-force-stateless-mem -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl;

void kernel(accessor<int, 1, access::mode::read_write, access::target::device>
                &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(int));
  simd<int, 32> v1(0, 1);

  auto v0 = gather<int, 32>(buf, offsets);

  v0 = v0 + v1;

  scatter<int, 32>(buf, offsets, v0);
}

void kernel(
    accessor<uint8_t, 1, access::mode::read_write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(uint8_t));
  simd<uint8_t, 32> v1(0, 1);

  simd<uint8_t, 32> v0 = gather<uint8_t, 32>(buf, offsets);

  // We honor integer promotion rules: uint8_t + uint8_t yields int
  // So we need to convert it back to simd<uint8_t, 32>
  v0 = convert<uint8_t>(v0 + v1);

  scatter<uint8_t, 32>(buf, offsets, v0);
}

void kernel(accessor<uint16_t, 1, access::mode::read_write,
                     access::target::device> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(uint16_t));
  simd<uint16_t, 32> v1(0, 1);

  simd<uint16_t, 32> v0 = gather<uint16_t, 32>(buf, offsets);

  // We honor integer promotion rules: uint16_t + uint16_t yields int
  // So we need to convert it back to simd<uint16_t, 32>
  v0 = convert<uint16_t>(v0 + v1);

  scatter<uint16_t, 32>(buf, offsets, v0);
}

void conv_kernel(accessor<uint16_t, 1, access::mode::read_write,
                          access::target::device> &buf) SYCL_ESIMD_FUNCTION {

  // Make sure we can pass offsets as a scalar.
  simd<uint16_t, 32> v0 = gather<uint16_t, 32>(buf, 0);

  scatter<uint16_t, 32>(buf, 0, v0);
}

// --- Negative tests.

// Incompatible mode (write).
SYCL_EXTERNAL void
kernel2(accessor<int, 1, access::mode::write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v;
  simd<uint32_t, 32> offset(0, 1);
  // CHECK: gather_scatter.cpp:72{{.*}}error: no matching function
  // function for call to 'gather'
  v = gather<int, 32>(buf, offset);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel3(accessor<int, 1, access::mode::read, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  simd<uint32_t, 32> offset(0, 1);
  // CHECK: gather_scatter.cpp:83{{.*}}error: no matching function
  // function for call to 'scatter'
  scatter<int, 32>(buf, offset, v);
}

// Incompatible mode (write).
SYCL_EXTERNAL void
kernel4(accessor<int, 1, access::mode::write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32 * 4> v;
  simd<uint32_t, 32> offset(0, sizeof(int) * 4);
  // CHECK: gather_scatter.cpp:94{{.*}}error: no matching function
  // function for call to 'gather'
  v = gather_rgba(buf, offset);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel5(accessor<int, 1, access::mode::read, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32 * 4> v(0, 1);
  simd<uint32_t, 32> offset(0, sizeof(int) * 4);
  // CHECK: gather_scatter.cpp:105{{.*}}error: no matching function
  // function for call to 'scatter'
  scatter_rgba(buf, offset, v);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel6(local_accessor<const int, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  simd<uint32_t, 32> offset(0, 1);
  // CHECK: gather_scatter.cpp:115{{.*}}error: no matching function
  // function for call to 'scatter'
  scatter<int, 32>(buf, offset, v);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel7(local_accessor<const int, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<int, 32 * 4> v(0, 1);
  simd<uint32_t, 32> offset(0, sizeof(int) * 4);
  // CHECK: gather_scatter.cpp:125{{.*}}error: no matching function
  // function for call to 'scatter'
  scatter_rgba(buf, offset, v);
}
