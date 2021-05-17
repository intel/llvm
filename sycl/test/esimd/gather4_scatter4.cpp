// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel(accessor<int, 1, access::mode::read_write,
                     access::target::global_buffer> &buf)
    __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, 1);
  simd<int, 32 * 4> v1(0, 1);

  auto v0 = gather4<int, 32, ESIMD_ABGR_ENABLE>(buf.get_pointer(), offsets);

  v0 = v0 + v1;

  scatter4<int, 32, ESIMD_ABGR_ENABLE>(buf.get_pointer(), v0, offsets);
}
