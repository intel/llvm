// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// This test checks that both host and device code can use simd::copy_from/to
// with alignment tags.

#include <sycl/ext/intel/experimental/esimd.hpp>
#include <stdio.h>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

// simd constructor
SYCL_EXTERNAL simd<float, 8> test_simd_constructor_element_aligned(float *ptr) {
  simd<float, 8> v(ptr, element_aligned);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_simd_constructor_vector_aligned(float *ptr) {
  simd<float, 8> v(ptr, vector_aligned);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_simd_constructor_overaligned(float *ptr) {
  simd<float, 8> v(ptr, overaligned<128>);
  return v;
}

// simd constructor accessor-based
SYCL_EXTERNAL simd<float, 8> test_simd_constructor_element_aligned(accessor<float, 1, access::mode::read_write, access::target::device> &acc) {
  simd<float, 8> v(acc, 0, element_aligned);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_simd_constructor_vector_aligned(accessor<float, 1, access::mode::read_write, access::target::device> &acc) {
  simd<float, 8> v(acc, 0, vector_aligned);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_simd_constructor_overaligned(accessor<float, 1, access::mode::read_write, access::target::device> &acc) {
  simd<float, 8> v(acc, 0, overaligned<128>);
  return v;
}

// simd copy_from/to
SYCL_EXTERNAL void test_simd_copy_default(float *src, float *dst) {
  simd<float, 8> v;
  v.copy_from(src);
  v.copy_to(dst);
}
SYCL_EXTERNAL void test_simd_copy_element_aligned(float *src, float *dst) {
  simd<float, 8> v;
  v.copy_from(src, element_aligned);
  v.copy_to(dst, element_aligned);
}
SYCL_EXTERNAL void test_simd_copy_vector_aligned(float *src, float *dst) {
  simd<float, 8> v;
  v.copy_from(src, vector_aligned);
  v.copy_to(dst, vector_aligned);
}
SYCL_EXTERNAL void test_simd_copy_overaligned(float *src, float *dst) {
  simd<float, 8> v;
  v.copy_from(src, overaligned<128>);
  v.copy_to(dst, overaligned<128>);
}

// simd copy_from/to accessor-based
SYCL_EXTERNAL void test_simd_copy_default(accessor<float, 1, access::mode::read, access::target::device> &src, accessor<float, 1, access::mode::write, access::target::device> &dst) {
  simd<float, 8> v;
  v.copy_from(src, 0);
  v.copy_to(dst, 0);
}
SYCL_EXTERNAL void test_simd_copy_element_aligned(accessor<float, 1, access::mode::read, access::target::device> &src, accessor<float, 1, access::mode::write, access::target::device> &dst) {
  simd<float, 8> v;
  v.copy_from(src, 0, element_aligned);
  v.copy_to(dst, 0, element_aligned);
}
SYCL_EXTERNAL void test_simd_copy_vector_aligned(accessor<float, 1, access::mode::read, access::target::device> &src, accessor<float, 1, access::mode::write, access::target::device> &dst) {
  simd<float, 8> v;
  v.copy_from(src, 0, vector_aligned);
  v.copy_to(dst, 0, vector_aligned);
}
SYCL_EXTERNAL void test_simd_copy_overaligned(accessor<float, 1, access::mode::read, access::target::device> &src, accessor<float, 1, access::mode::write, access::target::device> &dst) {
  simd<float, 8> v;
  v.copy_from(src, 0, overaligned<128>);
  v.copy_to(dst, 0, overaligned<128>);
}

// block_load
SYCL_EXTERNAL simd<float, 8> test_block_load_default(float *ptr) {
  simd<float, 8> v = block_load<float, 8>(ptr);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_block_load_element_aligned(float *ptr) {
  simd<float, 8> v = block_load<float, 8>(ptr, element_aligned);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_block_load_vector_aligned(float *ptr) {
  simd<float, 8> v = block_load<float, 8>(ptr, vector_aligned);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_block_load_overaligned(float *ptr) {
  simd<float, 8> v = block_load<float, 8>(ptr, overaligned<128>);
  return v;
}

// block_load accessor-based
SYCL_EXTERNAL simd<float, 8> test_block_load_default(accessor<float, 1, access::mode::read_write, access::target::device> &acc) {
  simd<float, 8> v = block_load<float, 8>(acc, 0);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_block_load_element_aligned(accessor<float, 1, access::mode::read_write, access::target::device> &acc) {
  simd<float, 8> v = block_load<float, 8>(acc, 0, element_aligned);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_block_load_vector_aligned(accessor<float, 1, access::mode::read_write, access::target::device> &acc) {
  simd<float, 8> v = block_load<float, 8>(acc, 0, vector_aligned);
  return v;
}
SYCL_EXTERNAL simd<float, 8> test_block_load_overaligned(accessor<float, 1, access::mode::read_write, access::target::device> &acc) {
  simd<float, 8> v = block_load<float, 8>(acc, 0, overaligned<128>);
  return v;
}
