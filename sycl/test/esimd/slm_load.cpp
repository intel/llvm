// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel() __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, 1);
  simd<int, 32> v1(0, 1);

  auto v0 = slm_load<int, 32>(offsets);

  esimd_fence(3);
  esimd_barrier();

  v0 = v0 + v1;

  slm_store<int, 32>(v0, offsets);
}

void slm_supported_types() __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, 1);
  auto v1 = slm_load<char, 32>(offsets);
  auto v2 = slm_load<short, 32>(offsets);
  auto v3 = slm_load<int, 32>(offsets);
  auto v4 = slm_load<float, 32>(offsets);
  // expected-error@+2 {{no matching function for call to 'slm_load'}}
  // expected-note@sycl/ext/intel/experimental/esimd/memory.hpp:* 2 {{candidate template ignored}}
  auto v5 = slm_load<double, 32>(offsets);

  slm_store<char, 32>(v1, offsets);
  slm_store<short, 32>(v2, offsets);
  slm_store<int, 32>(v3, offsets);
  slm_store<float, 32>(v4, offsets);
  simd<double, 32> v6(0, 1);
  // expected-error@+2 {{no matching function for call to 'slm_store'}}
  // expected-note@sycl/ext/intel/experimental/esimd/memory.hpp:* 2 {{candidate template ignored}}
  slm_store<double, 32>(v6, offsets);
}