// REQUIRES: gpu
// REQUIRES: aspect-fp16

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -fno-builtin -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda, hip

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>

int main() {

  sycl::queue device_queue(sycl::default_selector_v);
  std::cout << "Running on "
            << device_queue.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // half2int tests
  {
    std::initializer_list<uint16_t> input_vals = {
        0x8001, 0x83FF, 0x1,    0x3FF,  0x8000, 0x0,   0x0256,
        0x3800, 0xBE00, 0x7C00, 0xFC00, 0x7E00, 0x7bff};
    std::initializer_list<int> ref_vals_rd = {
        -1, -1, 0, 0, 0, 0, 0, 0, -2, 2147483647, -2147483648, 0, 65504};
    std::initializer_list<int> ref_vals_rn = {
        0, 0, 0, 0, 0, 0, 0, 0, -2, 2147483647, -2147483648, 0, 65504};
    std::initializer_list<int> ref_vals_ru = {
        0, 0, 1, 1, 0, 0, 1, 1, -1, 2147483647, -2147483648, 0, 65504};
    std::initializer_list<int> ref_vals_rz = {
        0, 0, 0, 0, 0, 0, 0, 0, -1, 2147483647, -2147483648, 0, 65504};

    test_host(input_vals, ref_vals_rd,
              FT1(sycl::half, sycl::ext::intel::math::half2int_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT1(sycl::half, sycl::ext::intel::math::half2int_rd));

    test_host(input_vals, ref_vals_rn,
              FT1(sycl::half, sycl::ext::intel::math::half2int_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT1(sycl::half, sycl::ext::intel::math::half2int_rn));

    test_host(input_vals, ref_vals_ru,
              FT1(sycl::half, sycl::ext::intel::math::half2int_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT1(sycl::half, sycl::ext::intel::math::half2int_ru));

    test_host(input_vals, ref_vals_rz,
              FT1(sycl::half, sycl::ext::intel::math::half2int_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT1(sycl::half, sycl::ext::intel::math::half2int_rz));
  }

  {
    std::initializer_list<uint16_t> input_vals = {
        0x8001, // max negative subnormal
        0x83FF, // min negative subnormal
        0x1,    // min positive subnormal
        0x3FF,  // max positive subnormal
        0x0,
        0x8000, // -0
        0x0256, // subnormal half-precision value
        0x5648, // 100.5
        0x5658, // 101.5
        0x564C, // 100.75
        0x7C00, // +infinity
        0x7E00, // NAN
        0x7bFF, // maximum half-precision value
        0xD648, // -100.5
    };
    std::initializer_list<unsigned int> ref_vals_rd = {
        0, 0, 0, 0, 0, 0, 0, 100, 101, 100, 4294967295, 0, 65504, 0};
    std::initializer_list<unsigned int> ref_vals_rn = {
        0, 0, 0, 0, 0, 0, 0, 100, 102, 101, 4294967295, 0, 65504, 0};
    std::initializer_list<unsigned int> ref_vals_ru = {
        0, 0, 1, 1, 0, 0, 1, 101, 102, 101, 4294967295, 0, 65504, 0};
    std::initializer_list<unsigned int> ref_vals_rz = {
        0, 0, 0, 0, 0, 0, 0, 100, 101, 100, 4294967295, 0, 65504, 0};

    test_host(input_vals, ref_vals_rd,
              FT1(sycl::half, sycl::ext::intel::math::half2uint_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT1(sycl::half, sycl::ext::intel::math::half2uint_rd));

    test_host(input_vals, ref_vals_rn,
              FT1(sycl::half, sycl::ext::intel::math::half2uint_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT1(sycl::half, sycl::ext::intel::math::half2uint_rn));

    test_host(input_vals, ref_vals_ru,
              FT1(sycl::half, sycl::ext::intel::math::half2uint_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT1(sycl::half, sycl::ext::intel::math::half2uint_ru));

    test_host(input_vals, ref_vals_rz,
              FT1(sycl::half, sycl::ext::intel::math::half2uint_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT1(sycl::half, sycl::ext::intel::math::half2uint_rz));
  }

  {
    std::initializer_list<uint16_t> input_vals = {
        0x8001, // max negative subnormal
        0x83FF, // min negative subnormal
        0x1,    // min positive subnormal
        0x3FF,  // max positive subnormal
        0x0,
        0x8000, // -0
        0x0256, // subnormal half-precision value
        0x5648, // 100.5
        0x5658, // 101.5
        0x564C, // 100.75
        0x7C00, // +infinity
        0xFC00, // -infinity
        0x7E00, // NAN
        0x7BFF, // maximum half-precision value
        0x7BEE, // 64960
        0xFBEE, // -64960
        0xD648, // -100.5
    };

    std::initializer_list<short> ref_vals_rd = {
        -1,  -1,    0,      0, 0,     0,     0,      100, 101,
        100, 32767, -32768, 0, 32767, 32767, -32768, -101};
    std::initializer_list<short> ref_vals_rn = {
        0,   0,     0,      0, 0,     0,     0,      100, 102,
        101, 32767, -32768, 0, 32767, 32767, -32768, -100};
    std::initializer_list<short> ref_vals_ru = {
        0,   0,     1,      1, 0,     0,     1,      101, 102,
        101, 32767, -32768, 0, 32767, 32767, -32768, -100};
    std::initializer_list<short> ref_vals_rz = {
        0,   0,     0,      0, 0,     0,     0,      100, 101,
        100, 32767, -32768, 0, 32767, 32767, -32768, -100};

    test_host(input_vals, ref_vals_rd,
              FT1(sycl::half, sycl::ext::intel::math::half2short_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT1(sycl::half, sycl::ext::intel::math::half2short_rd));

    test_host(input_vals, ref_vals_rn,
              FT1(sycl::half, sycl::ext::intel::math::half2short_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT1(sycl::half, sycl::ext::intel::math::half2short_rn));

    test_host(input_vals, ref_vals_ru,
              FT1(sycl::half, sycl::ext::intel::math::half2short_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT1(sycl::half, sycl::ext::intel::math::half2short_ru));

    test_host(input_vals, ref_vals_rz,
              FT1(sycl::half, sycl::ext::intel::math::half2short_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT1(sycl::half, sycl::ext::intel::math::half2short_rz));
  }

  {
    std::initializer_list<uint16_t> input_vals = {
        0x8001, // max negative subnormal
        0x83FF, // min negative subnormal
        0x1,    // min positive subnormal
        0x3FF,  // max positive subnormal
        0x0,
        0x8000, // -0
        0x0256, // subnormal half-precision value
        0x5648, // 100.5
        0x5658, // 101.5
        0x564C, // 100.75
        0x7C00, // +infinity
        0x7E00, // NAN
        0x7BFF, // maximum half-precision value
        0x7BEE, // 64960
        0x7BAA, // 62784
        0xD648, // -100.5
        0x51D9, // 46.789
    };

    std::initializer_list<unsigned short> ref_vals_rd = {
        0,   0,     0, 0,     0,     0,     0, 100, 101,
        100, 65535, 0, 65504, 64960, 62784, 0, 46};
    std::initializer_list<unsigned short> ref_vals_rn = {
        0,   0,     0, 0,     0,     0,     0, 100, 102,
        101, 65535, 0, 65504, 64960, 62784, 0, 47};
    std::initializer_list<unsigned short> ref_vals_ru = {
        0,   0,     1, 1,     0,     0,     1, 101, 102,
        101, 65535, 0, 65504, 64960, 62784, 0, 47};
    std::initializer_list<unsigned short> ref_vals_rz = {
        0,   0,     0, 0,     0,     0,     0, 100, 101,
        100, 65535, 0, 65504, 64960, 62784, 0, 46};

    test_host(input_vals, ref_vals_rd,
              FT1(sycl::half, sycl::ext::intel::math::half2ushort_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT1(sycl::half, sycl::ext::intel::math::half2ushort_rd));

    test_host(input_vals, ref_vals_rn,
              FT1(sycl::half, sycl::ext::intel::math::half2ushort_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT1(sycl::half, sycl::ext::intel::math::half2ushort_rn));

    test_host(input_vals, ref_vals_ru,
              FT1(sycl::half, sycl::ext::intel::math::half2ushort_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT1(sycl::half, sycl::ext::intel::math::half2ushort_ru));

    test_host(input_vals, ref_vals_rz,
              FT1(sycl::half, sycl::ext::intel::math::half2ushort_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT1(sycl::half, sycl::ext::intel::math::half2ushort_rz));
  }

  {
    std::initializer_list<uint16_t> input_vals = {
        0x8001, 0x83FF, 0x1, 0x3FF, 0x0,
        0x8000, // -0
        0x0256, // subnormal half-precision value
        0x5648, // 100.5
        0x5658, // 101.5
        0x564C, // 100.75
        0x7C00, // +infinity
        0xFC00, // -infinity
        0x7E00, // NAN
        0x7BFF, // maximum half-precision value
        0xFBFF, // minimum half-precision value
        0xFBEE, // -64960
        0xD648, // -100.5
        0x70E3, // 10008
        0xCCE3, // -19.546875
    };

    std::initializer_list<long long int> ref_vals_rd = {
        -1,
        -1,
        0,
        0,
        0,
        0,
        0,
        100,
        101,
        100,
        std::numeric_limits<long long>::max(),
        std::numeric_limits<long long>::min(),
        0,
        65504,
        -65504,
        -64960,
        -101,
        10008,
        -20};
    std::initializer_list<long long int> ref_vals_rn = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        100,
        102,
        101,
        std::numeric_limits<long long>::max(),
        std::numeric_limits<long long>::min(),
        0,
        65504,
        -65504,
        -64960,
        -100,
        10008,
        -20};
    std::initializer_list<long long int> ref_vals_ru = {
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        101,
        102,
        101,
        std::numeric_limits<long long>::max(),
        std::numeric_limits<long long>::min(),
        0,
        65504,
        -65504,
        -64960,
        -100,
        10008,
        -19};
    std::initializer_list<long long int> ref_vals_rz = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        100,
        101,
        100,
        std::numeric_limits<long long>::max(),
        std::numeric_limits<long long>::min(),
        0,
        65504,
        -65504,
        -64960,
        -100,
        10008,
        -19};

    test_host(input_vals, ref_vals_rd,
              FT1(sycl::half, sycl::ext::intel::math::half2ll_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT1(sycl::half, sycl::ext::intel::math::half2ll_rd));

    test_host(input_vals, ref_vals_rn,
              FT1(sycl::half, sycl::ext::intel::math::half2ll_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT1(sycl::half, sycl::ext::intel::math::half2ll_rn));

    test_host(input_vals, ref_vals_ru,
              FT1(sycl::half, sycl::ext::intel::math::half2ll_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT1(sycl::half, sycl::ext::intel::math::half2ll_ru));

    test_host(input_vals, ref_vals_rz,
              FT1(sycl::half, sycl::ext::intel::math::half2ll_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT1(sycl::half, sycl::ext::intel::math::half2ll_rz));
  }

  {
    std::initializer_list<uint16_t> input_vals = {
        0x8001, // max negative subnormal
        0x83FF, // min negative subnormal
        0x1,    // min positive subnormal
        0x3FF,  // max positive subnormal
        0x0,
        0x8000, // -0
        0x0256, // subnormal half-precision value
        0x5648, // 100.5
        0x5658, // 101.5
        0x564C, // 100.75
        0x7C00, // +infinity
        0x7E00, // NAN
        0x7BFF, // maximum half-precision value
        0x73DF, // 16120
        0x7AEE, // 56768
        0xD648, // -100.5
        0x70E3, // 10008
        0x4CE3, // 19.546875
    };

    std::initializer_list<unsigned long long> ref_vals_rd = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        100,
        101,
        100,
        std::numeric_limits<unsigned long long>::max(),
        0,
        65504,
        16120,
        56768,
        0,
        10008,
        19};
    std::initializer_list<unsigned long long> ref_vals_rn = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        100,
        102,
        101,
        std::numeric_limits<unsigned long long>::max(),
        0,
        65504,
        16120,
        56768,
        0,
        10008,
        20};
    std::initializer_list<unsigned long long> ref_vals_ru = {
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        101,
        102,
        101,
        std::numeric_limits<unsigned long long>::max(),
        0,
        65504,
        16120,
        56768,
        0,
        10008,
        20};
    std::initializer_list<unsigned long long> ref_vals_rz = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        100,
        101,
        100,
        std::numeric_limits<unsigned long long>::max(),
        0,
        65504,
        16120,
        56768,
        0,
        10008,
        19};

    test_host(input_vals, ref_vals_rd,
              FT1(sycl::half, sycl::ext::intel::math::half2ull_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT1(sycl::half, sycl::ext::intel::math::half2ull_rd));

    test_host(input_vals, ref_vals_rn,
              FT1(sycl::half, sycl::ext::intel::math::half2ull_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT1(sycl::half, sycl::ext::intel::math::half2ull_rn));

    test_host(input_vals, ref_vals_ru,
              FT1(sycl::half, sycl::ext::intel::math::half2ull_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT1(sycl::half, sycl::ext::intel::math::half2ull_ru));

    test_host(input_vals, ref_vals_rz,
              FT1(sycl::half, sycl::ext::intel::math::half2ull_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT1(sycl::half, sycl::ext::intel::math::half2ull_rz));
  }

  {
    std::initializer_list<short> input_vals = {
        0, 7, 49, -323, 2999, -32768, 32767, 32111, -11111, 16383};
    std::initializer_list<uint16_t> ref_vals_rd = {
        0,      0x4700, 0x5220, 0xDD0C, 0x69DB,
        0xF800, 0x77FF, 0x77D6, 0xF16D, 0x73FF};
    std::initializer_list<uint16_t> ref_vals_rn = {
        0,      0x4700, 0x5220, 0xDD0C, 0x69DC,
        0xF800, 0x7800, 0x77D7, 0xF16D, 0x7400};
    std::initializer_list<uint16_t> ref_vals_ru = {
        0,      0x4700, 0x5220, 0xDD0C, 0x69DC,
        0xF800, 0x7800, 0x77D7, 0xF16C, 0x7400};
    std::initializer_list<uint16_t> ref_vals_rz = {
        0,      0x4700, 0x5220, 0xDD0C, 0x69DB,
        0xF800, 0x77FF, 0x77D6, 0xF16C, 0x73FF};

    test_host(input_vals, ref_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::short2half_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::short2half_rd));

    test_host(input_vals, ref_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::short2half_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::short2half_rn));

    test_host(input_vals, ref_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::short2half_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::short2half_ru));

    test_host(input_vals, ref_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::short2half_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::short2half_rz));
  }

  {
    std::initializer_list<unsigned short> input_vals = {
        0, 7, 49, 2999, 32777, 11111, 16383, 60000, 65535, 64101, 51111, 39999};
    std::initializer_list<uint16_t> ref_vals_rd = {
        0,      0x4700, 0x5220, 0x69DB, 0x7800, 0x716C,
        0x73FF, 0x7B53, 0x7BFF, 0x7BD3, 0x7A3D, 0x78E1};
    std::initializer_list<uint16_t> ref_vals_rn = {
        0,      0x4700, 0x5220, 0x69DC, 0x7800, 0x716D,
        0x7400, 0x7B53, 0x7C00, 0x7BD3, 0x7A3D, 0x78E2};
    std::initializer_list<uint16_t> ref_vals_ru = {
        0,      0x4700, 0x5220, 0x69DC, 0x7801, 0x716D,
        0x7400, 0x7B53, 0x7C00, 0x7BD4, 0x7A3E, 0x78E2};
    std::initializer_list<uint16_t> ref_vals_rz = {
        0,      0x4700, 0x5220, 0x69DB, 0x7800, 0x716C,
        0x73FF, 0x7B53, 0x7BFF, 0x7BD3, 0x7A3D, 0x78E1};

    test_host(input_vals, ref_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::ushort2half_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::ushort2half_rd));

    test_host(input_vals, ref_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::ushort2half_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::ushort2half_rn));

    test_host(input_vals, ref_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::ushort2half_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::ushort2half_ru));

    test_host(input_vals, ref_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::ushort2half_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::ushort2half_rz));
  }

  {
    std::initializer_list<int> input_vals = {
        0,     -100,  137,    2187,  -3845, 10001,  -21283,
        16383, 65504, -64404, 65503, 65505, -80001, 1000021};
    std::initializer_list<uint16_t> ref_vals_rd = {
        0,      0xD640, 0x5848, 0x6845, 0xEB83, 0x70E2, 0xF533,
        0x73FF, 0x7BFF, 0xFBDD, 0x7BFE, 0x7BFF, 0xFC00, 0x7BFF};
    std::initializer_list<uint16_t> ref_vals_rn = {
        0,      0xD640, 0x5848, 0x6846, 0xEB82, 0x70E2, 0xF532,
        0x7400, 0x7BFF, 0xFBDD, 0x7BFF, 0x7BFF, 0xFC00, 0x7C00};
    std::initializer_list<uint16_t> ref_vals_ru = {
        0,      0xD640, 0x5848, 0x6846, 0xEB82, 0x70E3, 0xF532,
        0x7400, 0x7BFF, 0xFBDC, 0x7BFF, 0x7C00, 0xFBFF, 0x7C00};
    std::initializer_list<uint16_t> ref_vals_rz = {
        0,      0xD640, 0x5848, 0x6845, 0xEB82, 0x70E2, 0xF532,
        0x73FF, 0x7BFF, 0xFBDC, 0x7BFE, 0x7BFF, 0xFBFF, 0x7BFF};

    test_host(input_vals, ref_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::int2half_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::int2half_rd));

    test_host(input_vals, ref_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::int2half_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::int2half_rn));

    test_host(input_vals, ref_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::int2half_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::int2half_ru));

    test_host(input_vals, ref_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::int2half_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::int2half_rz));
  }

  {
    std::initializer_list<unsigned int> input_vals = {
        0,     100,   137,   2187,  3845,  10001,    21283,
        16383, 65504, 64404, 65503, 65505, 23220001, 1000021};
    std::initializer_list<uint16_t> ref_vals_rd = {
        0,      0x5640, 0x5848, 0x6845, 0x6B82, 0x70E2, 0x7532,
        0x73FF, 0x7BFF, 0x7BDC, 0x7BFE, 0x7BFF, 0x7BFF, 0x7BFF};
    std::initializer_list<uint16_t> ref_vals_rn = {
        0,      0x5640, 0x5848, 0x6846, 0x6B82, 0x70E2, 0x7532,
        0x7400, 0x7BFF, 0x7BDD, 0x7BFF, 0x7BFF, 0x7C00, 0x7C00};
    std::initializer_list<uint16_t> ref_vals_ru = {
        0,      0x5640, 0x5848, 0x6846, 0x6B83, 0x70E3, 0x7533,
        0x7400, 0x7BFF, 0x7BDD, 0x7BFF, 0x7C00, 0x7C00, 0x7C00};
    std::initializer_list<uint16_t> ref_vals_rz = {
        0,      0x5640, 0x5848, 0x6845, 0x6B82, 0x70E2, 0x7532,
        0x73FF, 0x7BFF, 0x7BDC, 0x7BFE, 0x7BFF, 0x7BFF, 0x7BFF};

    test_host(input_vals, ref_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::uint2half_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::uint2half_rd));

    test_host(input_vals, ref_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::uint2half_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::uint2half_rn));

    test_host(input_vals, ref_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::uint2half_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::uint2half_ru));

    test_host(input_vals, ref_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::uint2half_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::uint2half_rz));
  }

  {
    std::initializer_list<long long> input_vals = {
        0,     -100,  137,    2187,  -3845, 10001,          -21283,
        16383, 65504, -64404, 65503, 65505, -80023120101LL, 79921000021LL};
    std::initializer_list<uint16_t> ref_vals_rd = {
        0,      0xD640, 0x5848, 0x6845, 0xEB83, 0x70E2, 0xF533,
        0x73FF, 0x7BFF, 0xFBDD, 0x7BFE, 0x7BFF, 0xFC00, 0x7BFF};
    std::initializer_list<uint16_t> ref_vals_rn = {
        0,      0xD640, 0x5848, 0x6846, 0xEB82, 0x70E2, 0xF532,
        0x7400, 0x7BFF, 0xFBDD, 0x7BFF, 0x7BFF, 0xFC00, 0x7C00};
    std::initializer_list<uint16_t> ref_vals_ru = {
        0,      0xD640, 0x5848, 0x6846, 0xEB82, 0x70E3, 0xF532,
        0x7400, 0x7BFF, 0xFBDC, 0x7BFF, 0x7C00, 0xFBFF, 0x7C00};
    std::initializer_list<uint16_t> ref_vals_rz = {
        0,      0xD640, 0x5848, 0x6845, 0xEB82, 0x70E2, 0xF532,
        0x73FF, 0x7BFF, 0xFBDC, 0x7BFE, 0x7BFF, 0xFBFF, 0x7BFF};

    test_host(input_vals, ref_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::ll2half_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::ll2half_rd));

    test_host(input_vals, ref_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::ll2half_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::ll2half_rn));

    test_host(input_vals, ref_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::ll2half_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::ll2half_ru));

    test_host(input_vals, ref_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::ll2half_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::ll2half_rz));
  }

  {
    std::initializer_list<unsigned long long> input_vals = {
        0,     100,   137,   2187,  3845,  10001,       21283,
        16383, 65504, 64404, 65503, 65505, 77223220001, 99991000021};
    std::initializer_list<uint16_t> ref_vals_rd = {
        0,      0x5640, 0x5848, 0x6845, 0x6B82, 0x70E2, 0x7532,
        0x73FF, 0x7BFF, 0x7BDC, 0x7BFE, 0x7BFF, 0x7BFF, 0x7BFF};
    std::initializer_list<uint16_t> ref_vals_rn = {
        0,      0x5640, 0x5848, 0x6846, 0x6B82, 0x70E2, 0x7532,
        0x7400, 0x7BFF, 0x7BDD, 0x7BFF, 0x7BFF, 0x7C00, 0x7C00};
    std::initializer_list<uint16_t> ref_vals_ru = {
        0,      0x5640, 0x5848, 0x6846, 0x6B83, 0x70E3, 0x7533,
        0x7400, 0x7BFF, 0x7BDD, 0x7BFF, 0x7C00, 0x7C00, 0x7C00};
    std::initializer_list<uint16_t> ref_vals_rz = {
        0,      0x5640, 0x5848, 0x6845, 0x6B82, 0x70E2, 0x7532,
        0x73FF, 0x7BFF, 0x7BDC, 0x7BFE, 0x7BFF, 0x7BFF, 0x7BFF};

    test_host(input_vals, ref_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::ull2half_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::ull2half_rd));

    test_host(input_vals, ref_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::ull2half_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::ull2half_rn));

    test_host(input_vals, ref_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::ull2half_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::ull2half_ru));

    test_host(input_vals, ref_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::ull2half_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::ull2half_rz));
  }

  {
    std::initializer_list<unsigned int> input_vals = {
        0x807FFFFF, // min negative subnormnl
        0x80000001, // max negative subnormal
        0x1,        // min positive subnormal
        0x7FFFFF,   // max positive subnormal
        0x0,        // 0
        0x80000000, // -0
        0x7F800000, // +infinity
        0xFF800000, // -infinity
        // 0x7FC00000, // NAN
        0x41340000, // 11.25
        0x44812b21, // 1033.3478
        0x477FE000, // 65504
        0xC77FE000, // -65504
        0xC69C40A1, // -20000.3154
        0x44fCDC29, // 2022.88
        0x47709000, // 61584
        0xC770B000, // -61616
        0x47D54980, // 109203
        0xCD3EC5C0, // -200039423.564234
        0x21B877AA, // 1.25e-18
        0xC01D3A93, // -2.4567e-16
        0x0F81B224, // 1.2789e-29
        0x8CE08054, // -3.45899e-31
    };

    std::initializer_list<uint16_t> ref_vals_rd = {
        0x8001, 0x8001, 0,      0,      0,      0x8000, 0x7C00, 0xFC00,
        0x49A0, 0x6409, 0x7BFF, 0xFBFF, 0xF4E3, 0x67E6, 0x7B84, 0xFB86,
        0x7BFF, 0xFC00, 0,      0xC0EA, 0,      0x8001};

    std::initializer_list<uint16_t> ref_vals_rn = {
        0x8000, 0x8000, 0,      0,      0,      0x8000, 0x7C00, 0xFC00,
        0x49A0, 0x6409, 0x7BFF, 0xFBFF, 0xF4E2, 0x67E7, 0x7B84, 0xFB86,
        0x7C00, 0xFC00, 0,      0xC0EA, 0,      0x8000};

    std::initializer_list<uint16_t> ref_vals_ru = {
        0x8000, 0x8000, 1,      1,      0,      0x8000, 0x7C00, 0xFC00,
        0x49A0, 0x640A, 0x7BFF, 0xFBFF, 0xF4E2, 0x67E7, 0x7B85, 0xFB85,
        0x7C00, 0xFBFF, 1,      0xC0E9, 1,      0x8000};

    std::initializer_list<uint16_t> ref_vals_rz = {
        0x8000, 0x8000, 0,      0,      0,      0x8000, 0x7C00, 0xFC00,
        0x49A0, 0x6409, 0x7BFF, 0xFBFF, 0xF4E2, 0x67E6, 0x7B84, 0xFB85,
        0x7BFF, 0xFBFF, 0,      0xC0E9, 0,      0x8000};

    test_host(input_vals, ref_vals_rd,
              FT2(uint16_t, float, sycl::ext::intel::math::float2half_rd));
    test(device_queue, input_vals, ref_vals_rd,
         FT2(uint16_t, float, sycl::ext::intel::math::float2half_rd));

    test_host(input_vals, ref_vals_rn,
              FT2(uint16_t, float, sycl::ext::intel::math::float2half_rn));
    test(device_queue, input_vals, ref_vals_rn,
         FT2(uint16_t, float, sycl::ext::intel::math::float2half_rn));

    test_host(input_vals, ref_vals_ru,
              FT2(uint16_t, float, sycl::ext::intel::math::float2half_ru));
    test(device_queue, input_vals, ref_vals_ru,
         FT2(uint16_t, float, sycl::ext::intel::math::float2half_ru));

    test_host(input_vals, ref_vals_rz,
              FT2(uint16_t, float, sycl::ext::intel::math::float2half_rz));
    test(device_queue, input_vals, ref_vals_rz,
         FT2(uint16_t, float, sycl::ext::intel::math::float2half_rz));
  }

  return 0;
}
