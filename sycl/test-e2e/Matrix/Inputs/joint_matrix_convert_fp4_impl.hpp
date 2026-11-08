//==-- joint_matrix_convert_fp4_impl.hpp - DPC++ joint_matrix --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Bit exact coverage of joint_matrix_convert to and from fp4_e2m1_x, for the A
// and B operands, and for both 16 bit element types.

#include <sycl/usm.hpp>
#include <vector>

constexpr unsigned int numElems = 2;
using fp4 = syclex::fp4_e2m1_x<numElems>;
static_assert(sizeof(fp4) == 1, "an fp4_e2m1_x<2> is one packed byte");

// Host reference, sharing no code with the device conversions under test.
static uint8_t encode_pair(float lo, float hi) {
  const fp4 packed(marray<float, numElems>{lo, hi});
  return sycl::bit_cast<uint8_t>(packed);
}

static marray<float, numElems> decode_byte(uint8_t byte) {
  return (marray<float, numElems>)sycl::bit_cast<fp4>(byte);
}

template <typename T16> class cvt_up_a;
template <typename T16> class cvt_down_a;
template <typename T16> class cvt_up_b;
template <typename T16> class cvt_down_b;

constexpr size_t ARows = 8;
constexpr size_t ACols = 64;
constexpr size_t APackedCols = ACols / numElems;

// fp4 -> 16 bit, row major A operand.
template <typename T16> void test_up_a(queue q) {
  std::cout << "Test A up " << ARows << "x" << ACols << "\n";
  const size_t count = ARows * ACols;
  fp4 *src = malloc_shared<fp4>(count / numElems, q);
  T16 *dst = malloc_shared<T16>(count, q);
  uint8_t *bytes = reinterpret_cast<uint8_t *>(src);
  // Every nibble value in both nibble positions.
  for (size_t i = 0; i < count / numElems; i++)
    bytes[i] = i % 256;

  auto pSrc = address_space_cast<sycl::access::address_space::global_space,
                                 access::decorated::no>(src);
  auto pDst = address_space_cast<sycl::access::address_space::global_space,
                                 access::decorated::no>(dst);
  size_t sg_size = get_sg_size<cvt_up_a<T16>>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<cvt_up_a<T16>>(
         nd_range<2>({1, sg_size}, {1, sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, fp4, use::a, ARows, ACols, layout::row_major>
               m4;
           joint_matrix<sub_group, T16, use::a, ARows, ACols, layout::row_major>
               m16;
           joint_matrix_load(sg, m4, pSrc, APackedCols);
           joint_matrix_convert(sg, m4, m16);
           syclintelex::matrix::joint_matrix_store(sg, m16, pDst, ACols);
         });
   }).wait();

  std::vector<T16> expected(count);
  for (size_t i = 0; i < count / numElems; i++) {
    const marray<float, numElems> pair = decode_byte(bytes[i]);
    expected[numElems * i] = (T16)pair[0];
    expected[numElems * i + 1] = (T16)pair[1];
  }

  assert((matrix_compare<T16, T16, true>(ARows, ACols, dst, expected.data())));
  free(src, q);
  free(dst, q);
}

// 16 bit -> fp4, row major A operand.
template <typename T16>
void test_down_a(queue q, const char *what, const float *values,
                 size_t numValues) {
  std::cout << "Test A down " << ARows << "x" << ACols << ", " << what << "\n";
  const size_t count = ARows * ACols;
  T16 *src = malloc_shared<T16>(count, q);
  fp4 *dst = malloc_shared<fp4>(count / numElems, q);
  for (size_t i = 0; i < count; i++)
    src[i] = (T16)values[i % numValues];

  auto pSrc = address_space_cast<sycl::access::address_space::global_space,
                                 access::decorated::no>(src);
  auto pDst = address_space_cast<sycl::access::address_space::global_space,
                                 access::decorated::no>(dst);
  size_t sg_size = get_sg_size<cvt_down_a<T16>>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<cvt_down_a<T16>>(
         nd_range<2>({1, sg_size}, {1, sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T16, use::a, ARows, ACols, layout::row_major>
               m16;
           joint_matrix<sub_group, fp4, use::a, ARows, ACols, layout::row_major>
               m4;
           joint_matrix_load(sg, m16, pSrc, ACols);
           joint_matrix_convert(sg, m16, m4);
           syclintelex::matrix::joint_matrix_store(sg, m4, pDst, APackedCols);
         });
   }).wait();

  uint8_t *bytes = reinterpret_cast<uint8_t *>(dst);
  std::vector<uint8_t> expected(count / numElems);
  for (size_t i = 0; i < expected.size(); i++)
    expected[i] = encode_pair(src[numElems * i], src[numElems * i + 1]);

  assert(matrix_compare(ARows, APackedCols, bytes, expected.data()));
  free(src, q);
  free(dst, q);
}

// A B operand is VNNI packed along K, by 8 at 4 bits and by 2 at 16 bits, so
// the two sides of a conversion have different memory images of one tile.
constexpr size_t BRows = 64;
constexpr size_t BCols = 16;
constexpr size_t BVnni4 = 8;
constexpr size_t BVnni16 = 2;
constexpr size_t BStride4 = BCols / numElems * BVnni4;
constexpr size_t BStride16 = BCols * BVnni16;

// fp4 -> 16 bit, packed B operand.
template <typename T16> void test_up_b(queue q) {
  std::cout << "Test B up " << BRows << "x" << BCols << "\n";
  const size_t count = BRows * BCols;
  std::vector<float> logical(count), vnni4(count), vnni16(count);
  // Every code, shifted by row so the two nibbles of a byte differ.
  for (size_t i = 0; i < count; i++)
    logical[i] = decode_byte((i + i / BCols) % 16)[0];
  matrix_vnni<float>(BRows, BCols, logical.data(), vnni4.data(), BVnni4);
  matrix_vnni<float>(BRows, BCols, logical.data(), vnni16.data(), BVnni16);

  fp4 *src = malloc_shared<fp4>(count / numElems, q);
  T16 *dst = malloc_shared<T16>(count, q);
  uint8_t *bytes = reinterpret_cast<uint8_t *>(src);
  for (size_t i = 0; i < count / numElems; i++)
    bytes[i] = encode_pair(vnni4[numElems * i], vnni4[numElems * i + 1]);

  auto pSrc = address_space_cast<sycl::access::address_space::global_space,
                                 access::decorated::no>(src);
  auto pDst = address_space_cast<sycl::access::address_space::global_space,
                                 access::decorated::no>(dst);
  size_t sg_size = get_sg_size<cvt_up_b<T16>>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<cvt_up_b<T16>>(
         nd_range<2>({1, sg_size}, {1, sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, fp4, use::b, BRows, BCols,
                        layout::ext_intel_packed>
               m4;
           joint_matrix<sub_group, T16, use::b, BRows, BCols,
                        layout::ext_intel_packed>
               m16;
           joint_matrix_load(sg, m4, pSrc, BStride4);
           joint_matrix_convert(sg, m4, m16);
           syclintelex::matrix::joint_matrix_store(sg, m16, pDst, BStride16);
         });
   }).wait();

  std::vector<T16> expected(count);
  for (size_t i = 0; i < count; i++)
    expected[i] = (T16)vnni16[i];

  assert((matrix_compare<T16, T16, true>(BRows / BVnni16, BStride16, dst,
                                         expected.data())));
  free(src, q);
  free(dst, q);
}

// 16 bit -> fp4, packed B operand. Values cycle over the tile, so a pattern
// shorter than a column still covers both nibble positions.
template <typename T16>
void test_down_b(queue q, const char *what, const float *values,
                 size_t numValues) {
  std::cout << "Test B down " << BRows << "x" << BCols << ", " << what << "\n";
  const size_t count = BRows * BCols;
  std::vector<float> logical(count), vnni4(count), vnni16(count);
  // Round to T16 first: that is the value the device converts.
  for (size_t i = 0; i < count; i++)
    logical[i] = (float)(T16)values[i % numValues];
  matrix_vnni<float>(BRows, BCols, logical.data(), vnni4.data(), BVnni4);
  matrix_vnni<float>(BRows, BCols, logical.data(), vnni16.data(), BVnni16);

  T16 *src = malloc_shared<T16>(count, q);
  fp4 *dst = malloc_shared<fp4>(count / numElems, q);
  for (size_t i = 0; i < count; i++)
    src[i] = (T16)vnni16[i];

  auto pSrc = address_space_cast<sycl::access::address_space::global_space,
                                 access::decorated::no>(src);
  auto pDst = address_space_cast<sycl::access::address_space::global_space,
                                 access::decorated::no>(dst);
  size_t sg_size = get_sg_size<cvt_down_b<T16>>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<cvt_down_b<T16>>(
         nd_range<2>({1, sg_size}, {1, sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T16, use::b, BRows, BCols,
                        layout::ext_intel_packed>
               m16;
           joint_matrix<sub_group, fp4, use::b, BRows, BCols,
                        layout::ext_intel_packed>
               m4;
           joint_matrix_load(sg, m16, pSrc, BStride16);
           joint_matrix_convert(sg, m16, m4);
           syclintelex::matrix::joint_matrix_store(sg, m4, pDst, BStride4);
         });
   }).wait();

  uint8_t *bytes = reinterpret_cast<uint8_t *>(dst);
  std::vector<uint8_t> expected(count / numElems);
  for (size_t i = 0; i < expected.size(); i++)
    expected[i] = encode_pair(vnni4[2 * i], vnni4[2 * i + 1]);

  assert(matrix_compare(BRows / BVnni4, BStride4, bytes, expected.data()));
  free(src, q);
  free(dst, q);
}

// Nearer one neighbour than the other, plus the magnitudes that have to clamp.
static constexpr float Rounded[12] = {0.2f, 0.6f, 1.4f,  1.6f,  2.4f,  2.6f,
                                      3.4f, 4.9f, -0.6f, -2.6f, 10.0f, -10.0f};

// Halfway cases: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2,
// 3.5 -> 4, 5.0 -> 4.
static constexpr float Ties[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

int main() {
  queue q;
  if (!is_type_supported_by_device(q, matrix_type::fp4_e2m1)) {
    std::cout << "fp4_e2m1 type not supported on this device" << std::endl;
    return 0;
  }

  // Every representable value, in code order, taken from the type itself.
  float Representable[16];
  for (uint8_t code = 0; code < 16; code++)
    Representable[code] = decode_byte(code)[0];

  test_up_a<sycl::half>(q);
  test_up_a<bfloat16>(q);
  test_down_a<sycl::half>(q, "representable", Representable, 16);
  test_down_a<bfloat16>(q, "representable", Representable, 16);
  test_up_b<sycl::half>(q);
  test_up_b<bfloat16>(q);
  test_down_b<sycl::half>(q, "representable", Representable, 16);
  test_down_b<bfloat16>(q, "representable", Representable, 16);
  test_down_b<sycl::half>(q, "rounded and clamped", Rounded, 12);
  test_down_b<bfloat16>(q, "rounded and clamped", Rounded, 12);
  test_down_b<sycl::half>(q, "ties to even", Ties, 7);
  test_down_b<bfloat16>(q, "ties to even", Ties, 7);

  std::cout << "Passed\n";
  return 0;
}
