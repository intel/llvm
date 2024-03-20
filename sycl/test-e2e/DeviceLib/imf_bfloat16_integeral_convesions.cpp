// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -fno-builtin -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: cuda || hip

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>
#include <sycl/detail/core.hpp>

int main() {
  sycl::queue device_queue(sycl::default_selector_v);
  std::cout << "Running on "
            << device_queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  {
    std::initializer_list<uint16_t> input_vals = {
        0x0,    0x8000, 0x4020, 0xC020, 0x447A, 0x4700, 0xC700,
        0x3FC0, 0xBFC0, 0x4780, 0x7F80, 0xFF80, 0x7FC0};
    std::initializer_list<unsigned short> uoutput_vals_rd = {
        0, 0, 2, 0, 1000, 32768, 0, 1, 0, 65535, 65535, 0, 0};
    std::initializer_list<unsigned short> uoutput_vals_rn = {
        0, 0, 2, 0, 1000, 32768, 0, 2, 0, 65535, 65535, 0, 0};
    std::initializer_list<unsigned short> uoutput_vals_ru = {
        0, 0, 3, 0, 1000, 32768, 0, 2, 0, 65535, 65535, 0, 0};
    std::initializer_list<unsigned short> uoutput_vals_rz = {
        0, 0, 2, 0, 1000, 32768, 0, 1, 0, 65535, 65535, 0, 0};
    std::initializer_list<short> soutput_vals_rd = {
        0, 0, 2, -3, 1000, 32767, -32768, 1, -2, 32767, 32767, -32768, 0};
    std::initializer_list<short> soutput_vals_rn = {
        0, 0, 2, -2, 1000, 32767, -32768, 2, -2, 32767, 32767, -32768, 0};
    std::initializer_list<short> soutput_vals_ru = {
        0, 0, 3, -2, 1000, 32767, -32768, 2, -1, 32767, 32767, -32768, 0};
    std::initializer_list<short> soutput_vals_rz = {
        0, 0, 2, -2, 1000, 32767, -32768, 1, -1, 32767, 32767, -32768, 0};
    test_host(input_vals, uoutput_vals_rd,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ushort_rd));
    test_host(input_vals, uoutput_vals_rn,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ushort_rn));
    test_host(input_vals, uoutput_vals_ru,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ushort_ru));
    test_host(input_vals, uoutput_vals_rz,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ushort_rz));
    test(device_queue, input_vals, uoutput_vals_rd,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ushort_rd));
    test(device_queue, input_vals, uoutput_vals_rn,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ushort_rn));
    test(device_queue, input_vals, uoutput_vals_ru,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ushort_ru));
    test(device_queue, input_vals, uoutput_vals_rz,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ushort_rz));
    test_host(input_vals, soutput_vals_rd,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162short_rd));
    test_host(input_vals, soutput_vals_rn,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162short_rn));
    test_host(input_vals, soutput_vals_ru,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162short_ru));
    test_host(input_vals, soutput_vals_rz,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162short_rz));
    test(device_queue, input_vals, soutput_vals_rd,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162short_rd));
    test(device_queue, input_vals, soutput_vals_rn,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162short_rn));
    test(device_queue, input_vals, soutput_vals_ru,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162short_ru));
    test(device_queue, input_vals, soutput_vals_rz,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162short_rz));
  }

  {
    std::initializer_list<uint16_t> input_vals = {
        0x0,    0x8000, 0x4020, 0xC020, 0x447A, 0x4700, 0xC700,
        0x3FC0, 0xBFC0, 0x4780, 0x7F80, 0xFF80, 0x7FC0, 0x4F00,
        0xCF00, 0x4F80, 0x4E93, 0xA,    0x800A, 0x4D9F, 0xCC54};
    std::initializer_list<unsigned int> uoutput_vals_rd = {
        0, 0,          2,          0,          1000, 32768,     0,
        1, 0,          65536,      4294967295, 0,    0,         2147483648,
        0, 4294967295, 1233125376, 0,          0,    333447168, 0};
    std::initializer_list<unsigned int> uoutput_vals_rn = {
        0, 0,          2,          0,          1000, 32768,     0,
        2, 0,          65536,      4294967295, 0,    0,         2147483648,
        0, 4294967295, 1233125376, 0,          0,    333447168, 0};
    std::initializer_list<unsigned int> uoutput_vals_ru = {
        0, 0,          3,          0,          1000, 32768,     0,
        2, 0,          65536,      4294967295, 0,    0,         2147483648,
        0, 4294967295, 1233125376, 1,          0,    333447168, 0};
    std::initializer_list<unsigned int> uoutput_vals_rz = {
        0, 0,          2,          0,          1000, 32768,     0,
        1, 0,          65536,      4294967295, 0,    0,         2147483648,
        0, 4294967295, 1233125376, 0,          0,    333447168, 0};
    std::initializer_list<int> soutput_vals_rd = {
        0,      0,          2,           -3,         1000,       32768,
        -32768, 1,          -2,          65536,      2147483647, -2147483648,
        0,      2147483647, -2147483648, 2147483647, 1233125376, 0,
        -1,     333447168,  -55574528};
    std::initializer_list<int> soutput_vals_rn = {
        0,      0,          2,           -2,         1000,       32768,
        -32768, 2,          -2,          65536,      2147483647, -2147483648,
        0,      2147483647, -2147483648, 2147483647, 1233125376, 0,
        0,      333447168,  -55574528};
    std::initializer_list<int> soutput_vals_ru = {
        0,      0,          3,           -2,         1000,       32768,
        -32768, 2,          -1,          65536,      2147483647, -2147483648,
        0,      2147483647, -2147483648, 2147483647, 1233125376, 1,
        0,      333447168,  -55574528};
    std::initializer_list<int> soutput_vals_rz = {
        0,      0,          2,           -2,         1000,       32768,
        -32768, 1,          -1,          65536,      2147483647, -2147483648,
        0,      2147483647, -2147483648, 2147483647, 1233125376, 0,
        0,      333447168,  -55574528};
    test_host(input_vals, uoutput_vals_rd,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162uint_rd));
    test_host(input_vals, uoutput_vals_rn,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162uint_rn));
    test_host(input_vals, uoutput_vals_ru,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162uint_ru));
    test_host(input_vals, uoutput_vals_rz,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162uint_rz));
    test_host(input_vals, soutput_vals_rd,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162int_rd));
    test_host(input_vals, soutput_vals_rn,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162int_rn));
    test_host(input_vals, soutput_vals_ru,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162int_ru));
    test_host(input_vals, soutput_vals_rz,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162int_rz));
    test(device_queue, input_vals, uoutput_vals_rd,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162uint_rd));
    test(device_queue, input_vals, uoutput_vals_rn,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162uint_rn));
    test(device_queue, input_vals, uoutput_vals_ru,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162uint_ru));
    test(device_queue, input_vals, uoutput_vals_rz,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162uint_rz));
    test(device_queue, input_vals, soutput_vals_rd,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162int_rd));
    test(device_queue, input_vals, soutput_vals_rn,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162int_rn));
    test(device_queue, input_vals, soutput_vals_ru,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162int_ru));
    test(device_queue, input_vals, soutput_vals_rz,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162int_rz));
  }

  {
    std::initializer_list<uint16_t> input_vals = {
        0x0,    0x8000, 0x4020, 0xC020, 0x447A, 0x4700, 0xC700,
        0x3FC0, 0xBFC0, 0x4780, 0x7F80, 0xFF80, 0x7FC0, 0x4F00,
        0xCF00, 0x4F80, 0x4E93, 0xA,    0x800A, 0x4D9F, 0xCC54};
    std::initializer_list<unsigned long long> uoutput_vals_rd = {
        0,
        0,
        2,
        0,
        1000,
        32768,
        0,
        1,
        0,
        65536,
        18446744073709551615ULL,
        0,
        9223372036854775808ULL,
        2147483648,
        0,
        4294967296,
        1233125376,
        0,
        0,
        333447168,
        0};
    std::initializer_list<unsigned long long> uoutput_vals_rn = {
        0,
        0,
        2,
        0,
        1000,
        32768,
        0,
        2,
        0,
        65536,
        18446744073709551615ULL,
        0,
        9223372036854775808ULL,
        2147483648,
        0,
        4294967296,
        1233125376,
        0,
        0,
        333447168,
        0};
    std::initializer_list<unsigned long long> uoutput_vals_ru = {
        0,
        0,
        3,
        0,
        1000,
        32768,
        0,
        2,
        0,
        65536,
        18446744073709551615ULL,
        0,
        9223372036854775808ULL,
        2147483648,
        0,
        4294967296,
        1233125376,
        1,
        0,
        333447168,
        0};
    std::initializer_list<unsigned long long> uoutput_vals_rz = {
        0,
        0,
        2,
        0,
        1000,
        32768,
        0,
        1,
        0,
        65536,
        18446744073709551615ULL,
        0,
        9223372036854775808ULL,
        2147483648,
        0,
        4294967296,
        1233125376,
        0,
        0,
        333447168,
        0};
    std::initializer_list<long long> soutput_vals_rd = {
        0,         0,          2,           -3,         1000,       32768,
        -32768,    1,          -2,          65536,      LLONG_MAX,  LLONG_MIN,
        LLONG_MIN, 2147483648, -2147483648, 4294967296, 1233125376, 0,
        -1,        333447168,  -55574528};
    std::initializer_list<long long> soutput_vals_rn = {
        0,         0,          2,           -2,         1000,       32768,
        -32768,    2,          -2,          65536,      LLONG_MAX,  LLONG_MIN,
        LLONG_MIN, 2147483648, -2147483648, 4294967296, 1233125376, 0,
        0,         333447168,  -55574528};
    std::initializer_list<long long> soutput_vals_ru = {
        0,         0,          3,           -2,         1000,       32768,
        -32768,    2,          -1,          65536,      LLONG_MAX,  LLONG_MIN,
        LLONG_MIN, 2147483648, -2147483648, 4294967296, 1233125376, 1,
        0,         333447168,  -55574528};
    std::initializer_list<long long> soutput_vals_rz = {
        0,         0,          2,           -2,         1000,       32768,
        -32768,    1,          -1,          65536,      LLONG_MAX,  LLONG_MIN,
        LLONG_MIN, 2147483648, -2147483648, 4294967296, 1233125376, 0,
        0,         333447168,  -55574528};

    test_host(input_vals, uoutput_vals_rd,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ull_rd));
    test_host(input_vals, uoutput_vals_rn,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ull_rn));
    test_host(input_vals, uoutput_vals_ru,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ull_ru));
    test_host(input_vals, uoutput_vals_rz,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ull_rz));
    test_host(input_vals, soutput_vals_rd,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ll_rd));
    test_host(input_vals, soutput_vals_rn,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ll_rn));
    test_host(input_vals, soutput_vals_ru,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ll_ru));
    test_host(input_vals, soutput_vals_rz,
              FT1(sycl::ext::oneapi::bfloat16,
                  sycl::ext::intel::math::bfloat162ll_rz));
    test(device_queue, input_vals, uoutput_vals_rd,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ull_rd));
    test(device_queue, input_vals, uoutput_vals_rn,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ull_rn));
    test(device_queue, input_vals, uoutput_vals_ru,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ull_ru));
    test(device_queue, input_vals, uoutput_vals_rz,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ull_rz));
    test(device_queue, input_vals, soutput_vals_rd,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ll_rd));
    test(device_queue, input_vals, soutput_vals_rn,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ll_rn));
    test(device_queue, input_vals, soutput_vals_ru,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ll_ru));
    test(device_queue, input_vals, soutput_vals_rz,
         FT1(sycl::ext::oneapi::bfloat16,
             sycl::ext::intel::math::bfloat162ll_rz));
  }

  {
    std::initializer_list<unsigned short> input_vals = {
        0,    1,    4,    13,    64,    234,   498,   512,   853,   999,  1024,
        2048, 4078, 9999, 18983, 20000, 39932, 41120, 65535, 55532, 32768};
    std::initializer_list<uint16_t> output_vals_rd = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9,
        0x4400, 0x4455, 0x4479, 0x4480, 0x4500, 0x457E, 0x461C,
        0x4694, 0x469C, 0x471B, 0x4720, 0x477F, 0x4758, 0x4700};
    std::initializer_list<uint16_t> output_vals_rn = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9,
        0x4400, 0x4455, 0x447A, 0x4480, 0x4500, 0x457F, 0x461C,
        0x4694, 0x469C, 0x471C, 0x4721, 0x4780, 0x4759, 0x4700};
    std::initializer_list<uint16_t> output_vals_ru = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9,
        0x4400, 0x4456, 0x447A, 0x4480, 0x4500, 0x457F, 0x461D,
        0x4695, 0x469D, 0x471C, 0x4721, 0x4780, 0x4759, 0x4700};
    std::initializer_list<uint16_t> output_vals_rz = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9,
        0x4400, 0x4455, 0x4479, 0x4480, 0x4500, 0x457E, 0x461C,
        0x4694, 0x469C, 0x471B, 0x4720, 0x477F, 0x4758, 0x4700};
    test_host(input_vals, output_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::ushort2bfloat16_rd));
    test_host(input_vals, output_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::ushort2bfloat16_rn));
    test_host(input_vals, output_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::ushort2bfloat16_ru));
    test_host(input_vals, output_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::ushort2bfloat16_rz));
    test(device_queue, input_vals, output_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::ushort2bfloat16_rd));
    test(device_queue, input_vals, output_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::ushort2bfloat16_rn));
    test(device_queue, input_vals, output_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::ushort2bfloat16_ru));
    test(device_queue, input_vals, output_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::ushort2bfloat16_rz));
  }

  {
    std::initializer_list<unsigned> input_vals = {
        0,          1,         4,       13,       64,        234,
        498,        512,       853,     999,      1024,      2048,
        4078,       9999,      18983,   20000,    39932,     41120,
        65535,      55532,     125539,  999999,   382728456, 4294967295,
        3248548722, 128322354, 5638921, 11112222, 4284837219};
    std::initializer_list<uint16_t> output_vals_rd = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9, 0x4400,
        0x4455, 0x4479, 0x4480, 0x4500, 0x457E, 0x461C, 0x4694, 0x469C,
        0x471B, 0x4720, 0x477F, 0x4758, 0x47F5, 0x4974, 0x4DB6, 0x4F7F,
        0x4F41, 0x4CF4, 0x4AAC, 0x4B29, 0x4F7F};
    std::initializer_list<uint16_t> output_vals_rn = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9, 0x4400,
        0x4455, 0x447A, 0x4480, 0x4500, 0x457F, 0x461C, 0x4694, 0x469C,
        0x471C, 0x4721, 0x4780, 0x4759, 0x47F5, 0x4974, 0x4DB6, 0x4F80,
        0x4F42, 0x4CF5, 0x4AAC, 0x4B2A, 0x4F7F};
    std::initializer_list<uint16_t> output_vals_ru = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9, 0x4400,
        0x4456, 0x447A, 0x4480, 0x4500, 0x457F, 0x461D, 0x4695, 0x469D,
        0x471C, 0x4721, 0x4780, 0x4759, 0x47F6, 0x4975, 0x4DB7, 0x4F80,
        0x4F42, 0x4CF5, 0x4AAD, 0x4B2A, 0x4F80};
    std::initializer_list<uint16_t> output_vals_rz = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9, 0x4400,
        0x4455, 0x4479, 0x4480, 0x4500, 0x457E, 0x461C, 0x4694, 0x469C,
        0x471B, 0x4720, 0x477F, 0x4758, 0x47F5, 0x4974, 0x4DB6, 0x4F7F,
        0x4F41, 0x4CF4, 0x4AAC, 0x4B29, 0x4F7F};
    test_host(input_vals, output_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::uint2bfloat16_rd));
    test_host(input_vals, output_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::uint2bfloat16_rn));
    test_host(input_vals, output_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::uint2bfloat16_ru));
    test_host(input_vals, output_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::uint2bfloat16_rz));
    test(device_queue, input_vals, output_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::uint2bfloat16_rd));
    test(device_queue, input_vals, output_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::uint2bfloat16_rn));
    test(device_queue, input_vals, output_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::uint2bfloat16_ru));
    test(device_queue, input_vals, output_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::uint2bfloat16_rz));
  }

  {
    std::initializer_list<unsigned long long> input_vals = {0,
                                                            1,
                                                            4,
                                                            13,
                                                            64,
                                                            234,
                                                            498,
                                                            512,
                                                            853,
                                                            999,
                                                            1024,
                                                            2048,
                                                            4078,
                                                            9999,
                                                            18983,
                                                            20000,
                                                            39932,
                                                            41120,
                                                            65535,
                                                            55532,
                                                            125539,
                                                            999999,
                                                            382728456,
                                                            4294967295,
                                                            3248548722,
                                                            128322354,
                                                            5638921,
                                                            11112222,
                                                            4284837219,
                                                            99999911919,
                                                            28372717371112,
                                                            1990112719930525,
                                                            ULLONG_MAX};
    std::initializer_list<uint16_t> output_vals_rd = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9, 0x4400, 0x4455,
        0x4479, 0x4480, 0x4500, 0x457E, 0x461C, 0x4694, 0x469C, 0x471B, 0x4720,
        0x477F, 0x4758, 0x47F5, 0x4974, 0x4DB6, 0x4F7F, 0x4F41, 0x4CF4, 0x4AAC,
        0x4B29, 0x4F7F, 0x51BA, 0x55CE, 0x58E2, 0x5F7F};
    std::initializer_list<uint16_t> output_vals_rn = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9, 0x4400, 0x4455,
        0x447A, 0x4480, 0x4500, 0x457F, 0x461C, 0x4694, 0x469C, 0x471C, 0x4721,
        0x4780, 0x4759, 0x47F5, 0x4974, 0x4DB6, 0x4F80, 0x4F42, 0x4CF5, 0x4AAC,
        0x4B2A, 0x4F7F, 0x51BA, 0x55CE, 0x58E2, 0x5F80};
    std::initializer_list<uint16_t> output_vals_ru = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9, 0x4400, 0x4456,
        0x447A, 0x4480, 0x4500, 0x457F, 0x461D, 0x4695, 0x469D, 0x471C, 0x4721,
        0x4780, 0x4759, 0x47F6, 0x4975, 0x4DB7, 0x4F80, 0x4F42, 0x4CF5, 0x4AAD,
        0x4B2A, 0x4F80, 0x51BB, 0x55CF, 0x58E3, 0x5F80};
    std::initializer_list<uint16_t> output_vals_rz = {
        0,      0x3F80, 0x4080, 0x4150, 0x4280, 0x436A, 0x43F9, 0x4400, 0x4455,
        0x4479, 0x4480, 0x4500, 0x457E, 0x461C, 0x4694, 0x469C, 0x471B, 0x4720,
        0x477F, 0x4758, 0x47F5, 0x4974, 0x4DB6, 0x4F7F, 0x4F41, 0x4CF4, 0x4AAC,
        0x4B29, 0x4F7F, 0x51BA, 0x55CE, 0x58E2, 0x5F7F};
    test_host(input_vals, output_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::ull2bfloat16_rd));
    test_host(input_vals, output_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::ull2bfloat16_rn));
    test_host(input_vals, output_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::ull2bfloat16_ru));
    test_host(input_vals, output_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::ull2bfloat16_rz));
    test(device_queue, input_vals, output_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::ull2bfloat16_rd));
    test(device_queue, input_vals, output_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::ull2bfloat16_rn));
    test(device_queue, input_vals, output_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::ull2bfloat16_ru));
    test(device_queue, input_vals, output_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::ull2bfloat16_rz));
  }

  {
    std::initializer_list<short> input_vals = {
        0,   1,    -2,  4,    -7,   13,    24,   68,    -77,   128,    -127,
        345, -498, 982, -888, 1888, -2048, 4099, -8877, 12345, -21234, 32767,
        /*-32768*/};
    std::initializer_list<uint16_t> output_vals_rd = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288,
        0xC29A, 0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4475, 0xC45E, 0x44EC,
        0xC500, 0x4580, 0xC60B, 0x4640, 0xC6A6, 0x46FF, /*0xC700*/};
    std::initializer_list<uint16_t> output_vals_rn = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288,
        0xC29A, 0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4476, 0xC45E, 0x44EC,
        0xC500, 0x4580, 0xC60B, 0x4641, 0xC6A6, 0x4700, /*0xC700*/};
    std::initializer_list<uint16_t> output_vals_ru = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288,
        0xC29A, 0x4300, 0xC2FE, 0x43AD, 0xC3F9, 0x4476, 0xC45E, 0x44EC,
        0xC500, 0x4581, 0xC60A, 0x4641, 0xC6A5, 0x4700, /*0xC700*/};
    std::initializer_list<uint16_t> output_vals_rz = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288,
        0xC29A, 0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4475, 0xC45E, 0x44EC,
        0xC500, 0x4580, 0xC60A, 0x4640, 0xC6A5, 0x46FF, /*0xC700*/};
    test_host(input_vals, output_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::short2bfloat16_rd));
    test_host(input_vals, output_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::short2bfloat16_rn));
    test_host(input_vals, output_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::short2bfloat16_ru));
    test_host(input_vals, output_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::short2bfloat16_rz));
    test(device_queue, input_vals, output_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::short2bfloat16_rd));
    test(device_queue, input_vals, output_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::short2bfloat16_rn));
    test(device_queue, input_vals, output_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::short2bfloat16_ru));
    test(device_queue, input_vals, output_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::short2bfloat16_rz));
  }

  {
    std::initializer_list<int> input_vals = {
        0,         1,          -2,          4,          -7,        13,
        24,        68,         -77,         128,        -127,      345,
        -498,      982,        -888,        1888,       -2048,     4099,
        -8877,     12345,      -21234,      32767,      -32768,    69891,
        -72000,    141234,     -239999,     599192,     -612934,   2223456,
        -3424578,  6888888,    -7951238,    12004541,   -19889875, 40001122,
        -78410987, 1147483647, -2147483647, 2147483647, /*-2147483648*/};
    std::initializer_list<uint16_t> output_vals_rd = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288,
        0xC29A, 0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4475, 0xC45E, 0x44EC,
        0xC500, 0x4580, 0xC60B, 0x4640, 0xC6A6, 0x46FF, 0xC700, 0x4788,
        0xC78D, 0x4809, 0xC86B, 0x4912, 0xC916, 0x4A07, 0xCA52, 0x4AD2,
        0xCAF3, 0x4B37, 0xCB98, 0x4C18, 0xCC96, 0x4E88, 0xCF00, 0x4EFF,
        /*0xCF00*/};
    std::initializer_list<uint16_t> output_vals_rn = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288,
        0xC29A, 0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4476, 0xC45E, 0x44EC,
        0xC500, 0x4580, 0xC60B, 0x4641, 0xC6A6, 0x4700, 0xC700, 0x4789,
        0xC78D, 0x480A, 0xC86A, 0x4912, 0xC916, 0x4A08, 0xCA51, 0x4AD2,
        0xCAF3, 0x4B37, 0xCB98, 0x4C19, 0xCC96, 0x4E89, 0xCF00, 0x4F00,
        /*0xCF00*/};
    std::initializer_list<uint16_t> output_vals_ru = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288,
        0xC29A, 0x4300, 0xC2FE, 0x43AD, 0xC3F9, 0x4476, 0xC45E, 0x44EC,
        0xC500, 0x4581, 0xC60A, 0x4641, 0xC6A5, 0x4700, 0xC700, 0x4789,
        0xC78C, 0x480A, 0xC86A, 0x4913, 0xC915, 0x4A08, 0xCA51, 0x4AD3,
        0xCAF2, 0x4B38, 0xCB97, 0x4C19, 0xCC95, 0x4E89, 0xCEFF, 0x4F00,
        /*0xCF00*/};
    std::initializer_list<uint16_t> output_vals_rz = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288,
        0xC29A, 0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4475, 0xC45E, 0x44EC,
        0xC500, 0x4580, 0xC60A, 0x4640, 0xC6A5, 0x46FF, 0xC700, 0x4788,
        0xC78C, 0x4809, 0xC86A, 0x4912, 0xC915, 0x4A07, 0xCA51, 0x4AD2,
        0xCAF2, 0x4B37, 0xCB97, 0x4C18, 0xCC95, 0x4E88, 0xCEFF, 0x4EFF,
        /*0xCF00*/};
    test_host(input_vals, output_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::int2bfloat16_rd));
    test_host(input_vals, output_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::int2bfloat16_rn));
    test_host(input_vals, output_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::int2bfloat16_ru));
    test_host(input_vals, output_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::int2bfloat16_rz));
    test(device_queue, input_vals, output_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::int2bfloat16_rd));
    test(device_queue, input_vals, output_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::int2bfloat16_rn));
    test(device_queue, input_vals, output_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::int2bfloat16_ru));
    test(device_queue, input_vals, output_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::int2bfloat16_rz));
  }

  {
    std::initializer_list<long long int> input_vals = {0,
                                                       1,
                                                       -2,
                                                       4,
                                                       -7,
                                                       13,
                                                       24,
                                                       68,
                                                       -77,
                                                       128,
                                                       -127,
                                                       345,
                                                       -498,
                                                       982,
                                                       -888,
                                                       1888,
                                                       -2048,
                                                       4099,
                                                       -8877,
                                                       12345,
                                                       -21234,
                                                       32767,
                                                       -32768,
                                                       69891,
                                                       -72000,
                                                       141234,
                                                       -239999,
                                                       599192,
                                                       -612934,
                                                       2223456,
                                                       -3424578,
                                                       6888888,
                                                       -7951238,
                                                       12004541,
                                                       -19889875,
                                                       40001122,
                                                       -78410987,
                                                       1147483647,
                                                       -2147483647,
                                                       2147483647,
                                                       -2147483648,
                                                       LLONG_MAX,
                                                       -834859343834504,
                                                       287532947093842,
                                                       1990112719930525};
    std::initializer_list<uint16_t> output_vals_rd = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288, 0xC29A,
        0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4475, 0xC45E, 0x44EC, 0xC500, 0x4580,
        0xC60B, 0x4640, 0xC6A6, 0x46FF, 0xC700, 0x4788, 0xC78D, 0x4809, 0xC86B,
        0x4912, 0xC916, 0x4A07, 0xCA52, 0x4AD2, 0xCAF3, 0x4B37, 0xCB98, 0x4C18,
        0xCC96, 0x4E88, 0xCF00, 0x4EFF, 0xCF00, 0x5EFF, 0xD83E, 0x5782, 0x58E2};
    std::initializer_list<uint16_t> output_vals_rn = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288, 0xC29A,
        0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4476, 0xC45E, 0x44EC, 0xC500, 0x4580,
        0xC60B, 0x4641, 0xC6A6, 0x4700, 0xC700, 0x4789, 0xC78D, 0x480A, 0xC86A,
        0x4912, 0xC916, 0x4A08, 0xCA51, 0x4AD2, 0xCAF3, 0x4B37, 0xCB98, 0x4C19,
        0xCC96, 0x4E89, 0xCF00, 0x4F00, 0xCF00, 0x5F00, 0xD83E, 0x5783, 0x58E2};
    std::initializer_list<uint16_t> output_vals_ru = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288, 0xC29A,
        0x4300, 0xC2FE, 0x43AD, 0xC3F9, 0x4476, 0xC45E, 0x44EC, 0xC500, 0x4581,
        0xC60A, 0x4641, 0xC6A5, 0x4700, 0xC700, 0x4789, 0xC78C, 0x480A, 0xC86A,
        0x4913, 0xC915, 0x4A08, 0xCA51, 0x4AD3, 0xCAF2, 0x4B38, 0xCB97, 0x4C19,
        0xCC95, 0x4E89, 0xCEFF, 0x4F00, 0xCF00, 0x5F00, 0xD83D, 0x5783, 0x58E3};
    std::initializer_list<uint16_t> output_vals_rz = {
        0,      0x3F80, 0xC000, 0x4080, 0xC0E0, 0x4150, 0x41C0, 0x4288, 0xC29A,
        0x4300, 0xC2FE, 0x43AC, 0xC3F9, 0x4475, 0xC45E, 0x44EC, 0xC500, 0x4580,
        0xC60A, 0x4640, 0xC6A5, 0x46FF, 0xC700, 0x4788, 0xC78C, 0x4809, 0xC86A,
        0x4912, 0xC915, 0x4A07, 0xCA51, 0x4AD2, 0xCAF2, 0x4B37, 0xCB97, 0x4C18,
        0xCC95, 0x4E88, 0xCEFF, 0x4EFF, 0xCF00, 0x5EFF, 0xD83D, 0x5782, 0x58E2};
    test_host(input_vals, output_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::ll2bfloat16_rd));
    test_host(input_vals, output_vals_rn,
              FT(uint16_t, sycl::ext::intel::math::ll2bfloat16_rn));
    test_host(input_vals, output_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::ll2bfloat16_ru));
    test_host(input_vals, output_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::ll2bfloat16_rz));
    test(device_queue, input_vals, output_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::ll2bfloat16_rd));
    test(device_queue, input_vals, output_vals_rn,
         FT(uint16_t, sycl::ext::intel::math::ll2bfloat16_rn));
    test(device_queue, input_vals, output_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::ll2bfloat16_ru));
    test(device_queue, input_vals, output_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::ll2bfloat16_rz));
  }

  return 0;
}
