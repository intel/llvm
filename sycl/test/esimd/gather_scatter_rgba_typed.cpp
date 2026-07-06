// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Wno-deprecated-declarations -Xclang -verify %s

// This test checks that the device compiler can:
// - successfully compile the typed-surface gather_rgba_typed/scatter_rgba_typed
//   APIs;
// - emit an error if some of the restrictions on template parameters are
//   violated.

#include <sycl/accessor_image.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl;

// Valid usage: read the RGBA channels of 16 pixels, add a constant and write
// them back.
void kernel(accessor<uint4, 2, access::mode::read, access::target::image> accIn,
            accessor<uint4, 2, access::mode::write, access::target::image>
                accOut) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 16> u(0, 1);
  simd<uint32_t, 16> v(0, 0);
  simd<uint32_t, 16> r(0, 0);

  auto px = gather_rgba_typed<uint32_t, 16>(accIn, u, v);
  px += 1;
  scatter_rgba_typed<uint32_t, 16>(accOut, u, v, r, px);

  // Single-channel R variant with N = 8.
  simd<uint32_t, 8> u8(0, 1);
  auto red = gather_rgba_typed<uint32_t, 8, rgba_channel_mask::R>(accIn, u8);
  scatter_rgba_typed<uint32_t, 8, rgba_channel_mask::R>(accOut, u8, u8, u8,
                                                        red);
}

// Invalid SIMD width: only 8, 16 and 32 are supported.
void kernel_bad_n(accessor<uint4, 2, access::mode::read, access::target::image>
                      acc) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 9> u(0, 1);
  // expected-error@* {{Unsupported value of N. Only 8, 16 or 32 are supported}}
  // expected-note@* {{check_rgba_typed_access}}
  // expected-note@+1 {{in instantiation }}
  auto px = gather_rgba_typed<uint32_t, 9>(acc, u);
  (void)px;
}

// Invalid element size: only 4-byte channel types are supported.
void kernel_bad_type(
    accessor<uint4, 2, access::mode::read, access::target::image> acc)
    SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 16> u(0, 1);
  // expected-error@* {{Unsupported size of type T}}
  // expected-note@* {{check_rgba_typed_access}}
  // expected-note@* {{expression evaluates to}}
  // expected-note@+1 {{in instantiation }}
  auto px = gather_rgba_typed<uint16_t, 16>(acc, u);
  (void)px;
}

// Invalid write channel mask: only masks covering consecutive channels starting
// from R (R, GR, BGR, ABGR) are supported for writes.
void kernel_bad_mask(
    accessor<uint4, 2, access::mode::write, access::target::image> acc,
    simd<uint32_t, 16 * 3> vals) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 16> u(0, 1);
  // expected-error-re@* {{static assertion failed{{.*}}rgba_channel_mask{{.*}}ABGR{{.*}}BGR{{.*}}GR{{.*}}R{{.*}}}}
  // expected-note@* {{validate_rgba_write_channel_mask}}
  // expected-note@+1 {{in instantiation }}
  scatter_rgba_typed<uint32_t, 16, rgba_channel_mask::AGR>(acc, u, u, u, vals);
}

// Typed surface access requires an image accessor.
void kernel_bad_accessor(local_accessor<uint32_t, 1> acc) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 16> u(0, 1);
  // expected-error@* {{Typed surface access requires an image accessor}}
  // expected-note@* {{check_rgba_typed_access}}
  // expected-note@+1 {{in instantiation }}
  auto px = gather_rgba_typed<uint32_t, 16>(acc, u);
  (void)px;
}
