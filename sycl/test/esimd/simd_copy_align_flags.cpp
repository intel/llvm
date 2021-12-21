// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// This test checks that both host and device code can use simd::copy_from/to
// with alignment tags.

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

// simd constructor
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_simd_constructor_default(T *ptr) {
  simd<T, N> v(ptr);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_simd_constructor_element_aligned(T *ptr) {
  simd<T, N> v(ptr, element_aligned);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_simd_constructor_vector_aligned(T *ptr) {
  simd<T, N> v(ptr, vector_aligned);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_simd_constructor_overaligned(T *ptr) {
  simd<T, N> v(ptr, overaligned<128>);
  return v;
}

#define TEST_SVM_CONSTRUCTOR(T, N) \
template simd<T, N> test_simd_constructor_default<T, N>(T*); \
template simd<T, N> test_simd_constructor_element_aligned<T, N>(T*); \
template simd<T, N> test_simd_constructor_vector_aligned<T, N>(T*); \
template simd<T, N> test_simd_constructor_overaligned<T, N>(T*);

TEST_SVM_CONSTRUCTOR(char, 2)
TEST_SVM_CONSTRUCTOR(char, 52)
TEST_SVM_CONSTRUCTOR(short, 5)
TEST_SVM_CONSTRUCTOR(short, 55)
TEST_SVM_CONSTRUCTOR(int, 7)
TEST_SVM_CONSTRUCTOR(int, 57)
TEST_SVM_CONSTRUCTOR(float, 14)
TEST_SVM_CONSTRUCTOR(float, 54)
TEST_SVM_CONSTRUCTOR(double, 16)
TEST_SVM_CONSTRUCTOR(double, 56)

#undef TEST_SVM_CONSTRUCTOR

// simd constructor accessor-based
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_simd_constructor_default(accessor<T, 1, access::mode::read_write, access::target::device> &acc) {
  simd<T, N> v(acc, 0);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_simd_constructor_element_aligned(accessor<T, 1, access::mode::read_write, access::target::device> &acc) {
  simd<T, N> v(acc, 0, element_aligned);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_simd_constructor_vector_aligned(accessor<T, 1, access::mode::read_write, access::target::device> &acc) {
  simd<T, N> v(acc, 0, vector_aligned);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_simd_constructor_overaligned(accessor<T, 1, access::mode::read_write, access::target::device> &acc) {
  simd<T, N> v(acc, 0, overaligned<128>);
  return v;
}

#define TEST_ACC_CONSTRUCTOR(T, N) \
template simd<T, N> test_simd_constructor_default<T, N>(accessor<T, 1, access::mode::read_write, access::target::device> &); \
template simd<T, N> test_simd_constructor_element_aligned<T, N>(accessor<T, 1, access::mode::read_write, access::target::device> &); \
template simd<T, N> test_simd_constructor_vector_aligned<T, N>(accessor<T, 1, access::mode::read_write, access::target::device> &); \
template simd<T, N> test_simd_constructor_overaligned<T, N>(accessor<T, 1, access::mode::read_write, access::target::device> &);

TEST_ACC_CONSTRUCTOR(char, 2)
TEST_ACC_CONSTRUCTOR(char, 52)
TEST_ACC_CONSTRUCTOR(short, 5)
TEST_ACC_CONSTRUCTOR(short, 55)
TEST_ACC_CONSTRUCTOR(int, 7)
TEST_ACC_CONSTRUCTOR(int, 57)
TEST_ACC_CONSTRUCTOR(float, 14)
TEST_ACC_CONSTRUCTOR(float, 54)
TEST_ACC_CONSTRUCTOR(double, 16)
TEST_ACC_CONSTRUCTOR(double, 56)

#undef TEST_ACC_CONSTRUCTOR

// simd copy_from/to
template <typename T, int N>
SYCL_EXTERNAL void test_simd_copy_default(T *src, T *dst) {
  simd<T, N> v;
  v.copy_from(src);
  v.copy_to(dst);
}
template <typename T, int N>
SYCL_EXTERNAL void test_simd_copy_element_aligned(T *src, T *dst) {
  simd<T, N> v;
  v.copy_from(src, element_aligned);
  v.copy_to(dst, element_aligned);
}
template <typename T, int N>
SYCL_EXTERNAL void test_simd_copy_vector_aligned(T *src, T *dst) {
  simd<T, N> v;
  v.copy_from(src, vector_aligned);
  v.copy_to(dst, vector_aligned);
}
template <typename T, int N>
SYCL_EXTERNAL void test_simd_copy_overaligned(T *src, T *dst) {
  simd<T, N> v;
  v.copy_from(src, overaligned<128>);
  v.copy_to(dst, overaligned<128>);
}

#define TEST_SVM_COPY(T, N) \
template void test_simd_copy_default<T, N>(T*, T*); \
template void test_simd_copy_element_aligned<T, N>(T*, T*); \
template void test_simd_copy_vector_aligned<T, N>(T*, T*); \
template void test_simd_copy_overaligned<T, N>(T*, T*);

TEST_SVM_COPY(char, 2)
TEST_SVM_COPY(char, 52)
TEST_SVM_COPY(short, 5)
TEST_SVM_COPY(short, 55)
TEST_SVM_COPY(int, 7)
TEST_SVM_COPY(int, 57)
TEST_SVM_COPY(float, 14)
TEST_SVM_COPY(float, 54)
TEST_SVM_COPY(double, 16)
TEST_SVM_COPY(double, 56)

#undef TEST_SVM_COPY

// simd copy_from/to accessor-based
template <typename T, int N>
SYCL_EXTERNAL void test_simd_copy_default(accessor<T, 1, access::mode::read, access::target::device> &src, accessor<T, 1, access::mode::write, access::target::device> &dst) {
  simd<T, N> v;
  v.copy_from(src, 0);
  v.copy_to(dst, 0);
}
template <typename T, int N>
SYCL_EXTERNAL void test_simd_copy_element_aligned(accessor<T, 1, access::mode::read, access::target::device> &src, accessor<T, 1, access::mode::write, access::target::device> &dst) {
  simd<T, N> v;
  v.copy_from(src, 0, element_aligned);
  v.copy_to(dst, 0, element_aligned);
}
template <typename T, int N>
SYCL_EXTERNAL void test_simd_copy_vector_aligned(accessor<T, 1, access::mode::read, access::target::device> &src, accessor<T, 1, access::mode::write, access::target::device> &dst) {
  simd<T, N> v;
  v.copy_from(src, 0, vector_aligned);
  v.copy_to(dst, 0, vector_aligned);
}
template <typename T, int N>
SYCL_EXTERNAL void test_simd_copy_overaligned(accessor<T, 1, access::mode::read, access::target::device> &src, accessor<T, 1, access::mode::write, access::target::device> &dst) {
  simd<T, N> v;
  v.copy_from(src, 0, overaligned<128>);
  v.copy_to(dst, 0, overaligned<128>);
}

#define TEST_ACC_COPY(T, N) \
template void test_simd_copy_default<T, N>(accessor<T, 1, access::mode::read, access::target::device> &, accessor<T, 1, access::mode::write, access::target::device> &); \
template void test_simd_copy_element_aligned<T, N>(accessor<T, 1, access::mode::read, access::target::device> &, accessor<T, 1, access::mode::write, access::target::device> &); \
template void test_simd_copy_vector_aligned<T, N>(accessor<T, 1, access::mode::read, access::target::device> &, accessor<T, 1, access::mode::write, access::target::device> &); \
template void test_simd_copy_overaligned<T, N>(accessor<T, 1, access::mode::read, access::target::device> &, accessor<T, 1, access::mode::write, access::target::device> &);

TEST_ACC_COPY(char, 2)
TEST_ACC_COPY(char, 52)
TEST_ACC_COPY(short, 5)
TEST_ACC_COPY(short, 55)
TEST_ACC_COPY(int, 7)
TEST_ACC_COPY(int, 57)
TEST_ACC_COPY(float, 14)
TEST_ACC_COPY(float, 54)
TEST_ACC_COPY(double, 16)
TEST_ACC_COPY(double, 56)

#undef TEST_ACC_COPY

// block_load
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_block_load_default(T *ptr) {
  simd<T, N> v = block_load<T, N>(ptr);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_block_load_element_aligned(T *ptr) {
  simd<T, N> v = block_load<T, N>(ptr, element_aligned);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_block_load_vector_aligned(T *ptr) {
  simd<T, N> v = block_load<T, N>(ptr, vector_aligned);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_block_load_overaligned(T *ptr) {
  simd<T, N> v = block_load<T, N>(ptr, overaligned<128>);
  return v;
}

#define TEST_SVM_BLOCK_LOAD(T, N) \
template simd<T, N> test_block_load_default<T, N>(T*); \
template simd<T, N> test_block_load_element_aligned<T, N>(T*); \
template simd<T, N> test_block_load_vector_aligned<T, N>(T*); \
template simd<T, N> test_block_load_overaligned<T, N>(T*);

TEST_SVM_BLOCK_LOAD(char, 16)
TEST_SVM_BLOCK_LOAD(short, 8)
TEST_SVM_BLOCK_LOAD(int, 16)
TEST_SVM_BLOCK_LOAD(float, 32)
TEST_SVM_BLOCK_LOAD(double, 16)

#undef TEST_SVM_BLOCK_LOAD

// block_load accessor-based
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_block_load_default(accessor<T, 1, access::mode::read_write, access::target::device> &acc) {
  simd<T, N> v = block_load<T, N>(acc, 0);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_block_load_element_aligned(accessor<T, 1, access::mode::read_write, access::target::device> &acc) {
  simd<T, N> v = block_load<T, N>(acc, 0, element_aligned);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_block_load_vector_aligned(accessor<T, 1, access::mode::read_write, access::target::device> &acc) {
  simd<T, N> v = block_load<T, N>(acc, 0, vector_aligned);
  return v;
}
template <typename T, int N>
SYCL_EXTERNAL simd<T, N> test_block_load_overaligned(accessor<T, 1, access::mode::read_write, access::target::device> &acc) {
  simd<T, N> v = block_load<T, N>(acc, 0, overaligned<128>);
  return v;
}

#define TEST_ACC_BLOCK_LOAD(T, N) \
template simd<T, N> test_block_load_default<T, N>(accessor<T, 1, access::mode::read_write, access::target::device> &); \
template simd<T, N> test_block_load_element_aligned<T, N>(accessor<T, 1, access::mode::read_write, access::target::device> &); \
template simd<T, N> test_block_load_vector_aligned<T, N>(accessor<T, 1, access::mode::read_write, access::target::device> &); \
template simd<T, N> test_block_load_overaligned<T, N>(accessor<T, 1, access::mode::read_write, access::target::device> &);

TEST_ACC_BLOCK_LOAD(char, 16)
TEST_ACC_BLOCK_LOAD(short, 8)
TEST_ACC_BLOCK_LOAD(int, 16)
TEST_ACC_BLOCK_LOAD(float, 32)
TEST_ACC_BLOCK_LOAD(double, 16)

#undef TEST_ACC_BLOCK_LOAD
