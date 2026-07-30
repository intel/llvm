// The test verifies 4-bit LUT upconversion on Xe3+.

// RUN: %clangxx -fsycl -fsycl-device-only -S %s -o %t.ll
// -O0 lowering, requires `-force-disable-esimd-opt` to disable all
// optimizations.
// RUN: sycl-post-link -O0 -force-disable-esimd-opt -lower-esimd -split-esimd -S %t.ll -o %t.table
// RUN: FileCheck %s -input-file=%t_0.esimd.ll

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

SYCL_EXTERNAL void fourtosixteen() SYCL_ESIMD_FUNCTION {
  // CHECK: [[RD0:%.*]] = call <16 x i8> @llvm.genx.rdregioni.v16i8.v64i8.i16(<64 x i8> {{.*}}, i32 4, i32 1, i32 0, i16 0, i32 0)
  // CHECK: = call <16 x i32> @llvm.genx.packed.4bit.upconvert.lut.v16i32.v16i8(<16 x i32> {{.*}}, <16 x i8> [[RD0]])
  // CHECK: [[RD1:%.*]] = call <16 x i8> @llvm.genx.rdregioni.v16i8.v64i8.i16(<64 x i8> {{.*}}, i32 4, i32 1, i32 0, i16 1, i32 0)
  // CHECK: = call <16 x i32> @llvm.genx.packed.4bit.upconvert.lut.v16i32.v16i8(<16 x i32> {{.*}}, <16 x i8> [[RD1]])
  simd<uint32_t, 16> lut = 0;

  simd<uint32_t, 16> src(0, 0);

  simd<uint8_t, 16 * 4> src_byte = src.bit_cast_view<uint8_t>();

  simd<uint32_t, 16> res =
      experimental::esimd::packed_4bit_upconvert_lut<0>(lut, src_byte);

  res = experimental::esimd::packed_4bit_upconvert_lut<1>(lut, src_byte);
}

SYCL_EXTERNAL void fourtoeight() SYCL_ESIMD_FUNCTION {
  // CHECK: [[RD2:%.*]] = call <16 x i16> @llvm.genx.rdregioni.v16i16.v32i16.i16(<32 x i16> {{.*}}, i32 2, i32 1, i32 0, i16 0, i32 0)
  // CHECK: = call <16 x i32> @llvm.genx.packed.4bit.upconvert.lut.v16i32.v16i16(<16 x i32> {{.*}}, <16 x i16> [[RD2]])
  // CHECK: [[RD3:%.*]] = call <16 x i16> @llvm.genx.rdregioni.v16i16.v32i16.i16(<32 x i16> {{.*}}, i32 2, i32 1, i32 0, i16 2, i32 0)
  // CHECK: = call <16 x i32> @llvm.genx.packed.4bit.upconvert.lut.v16i32.v16i16(<16 x i32> {{.*}}, <16 x i16> [[RD3]])
  simd<uint32_t, 16> lut = 0;

  simd<uint32_t, 16> src(0, 0);

  simd<uint16_t, 16 * 2> src_byte = src.bit_cast_view<uint16_t>();

  simd<uint32_t, 16> res =
      experimental::esimd::packed_4bit_upconvert_lut<0>(lut, src_byte);
  res = experimental::esimd::packed_4bit_upconvert_lut<1>(lut, src_byte);
}

SYCL_EXTERNAL void fourtosixteen_VL32() SYCL_ESIMD_FUNCTION {
  // CHECK: [[RD4:%.*]] = call <32 x i8> @llvm.genx.rdregioni.v32i8.v128i8.i16(<128 x i8> {{.*}}, i32 4, i32 1, i32 0, i16 0, i32 0)
  // CHECK: = call <32 x i32> @llvm.genx.packed.4bit.upconvert.lut.v32i32.v32i8(<16 x i32> {{.*}}, <32 x i8> [[RD4]])
  // CHECK: [[RD5:%.*]] = call <32 x i8> @llvm.genx.rdregioni.v32i8.v128i8.i16(<128 x i8> {{.*}}, i32 4, i32 1, i32 0, i16 1, i32 0)
  // CHECK: = call <32 x i32> @llvm.genx.packed.4bit.upconvert.lut.v32i32.v32i8(<16 x i32> {{.*}}, <32 x i8> [[RD5]])
  simd<uint32_t, 16> lut = 0;

  simd<uint32_t, 32> src(0, 0);

  simd<uint8_t, 32 * 4> src_byte = src.bit_cast_view<uint8_t>();

  simd<uint32_t, 32> res =
      experimental::esimd::packed_4bit_upconvert_lut<0, uint8_t, 32>(lut,
                                                                     src_byte);

  res = experimental::esimd::packed_4bit_upconvert_lut<1, uint8_t, 32>(
      lut, src_byte);
}

SYCL_EXTERNAL void fourtoeight_VL32() SYCL_ESIMD_FUNCTION {
  // CHECK: [[RD6:%.*]] = call <32 x i16> @llvm.genx.rdregioni.v32i16.v64i16.i16(<64 x i16> {{.*}}, i32 2, i32 1, i32 0, i16 0, i32 0)
  // CHECK: = call <32 x i32> @llvm.genx.packed.4bit.upconvert.lut.v32i32.v32i16(<16 x i32> {{.*}}, <32 x i16> [[RD6]])
  // CHECK: [[RD7:%.*]] = call <32 x i16> @llvm.genx.rdregioni.v32i16.v64i16.i16(<64 x i16> {{.*}}, i32 2, i32 1, i32 0, i16 2, i32 0)
  // CHECK: = call <32 x i32> @llvm.genx.packed.4bit.upconvert.lut.v32i32.v32i16(<16 x i32> {{.*}}, <32 x i16> [[RD7]])
  simd<uint32_t, 16> lut = 0;

  simd<uint32_t, 32> src(0, 0);

  simd<uint16_t, 32 * 2> src_byte = src.bit_cast_view<uint16_t>();

  simd<uint32_t, 32> res =
      experimental::esimd::packed_4bit_upconvert_lut<0, uint16_t, 32>(lut,
                                                                      src_byte);
  res = experimental::esimd::packed_4bit_upconvert_lut<1, uint16_t, 32>(
      lut, src_byte);
}
