// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace cl::sycl;

void kernel(
    accessor<int, 1, access::mode::read_write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
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
                     access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(uint16_t));
  simd<uint16_t, 32> v1(0, 1);

  simd<uint16_t, 32> v0 = gather<uint16_t, 32>(buf, offsets);

  // We honor integer promotion rules: uint16_t + uint16_t yields int
  // So we need to convert it back to simd<uint16_t, 32>
  v0 = convert<uint16_t>(v0 + v1);

  scatter<uint16_t, 32>(buf, offsets, v0);
}
