// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

// RUN: %{build} -fno-builtin %if target-spir %{ -fsycl-device-lib-jit-link -Wno-deprecated %} -o %t2.out
// RUN: %{run} %t2.out

#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>
namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

int main(int, char **) {
  s::queue device_queue(s::default_selector_v);
  std::cout << "Running on "
            << device_queue.get_device().get_info<s::info::device::name>()
            << "\n";

  {
    std::initializer_list<float> input_vals = {
        0.0,           2.5,           23.34577463, 12387737.23232,
        -41.324938243, -9.1232335E+8, 9.54E-20,    -3.98726272938E+8};

    std::initializer_list<int> ref_vals_rd = {0,   2,          23, 12387737,
                                              -42, -912323328, 0,  -398726272};
    std::initializer_list<int> ref_vals_rn = {0,   2,          23, 12387737,
                                              -41, -912323328, 0,  -398726272};
    std::initializer_list<int> ref_vals_ru = {0,   3,          24, 12387737,
                                              -41, -912323328, 1,  -398726272};
    std::initializer_list<int> ref_vals_rz = {0,   2,          23, 12387737,
                                              -41, -912323328, 0,  -398726272};
    test(device_queue, input_vals, ref_vals_rd,
         F(s::ext::intel::math::float2int_rd));
    std::cout << "float2int_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         F(s::ext::intel::math::float2int_rn));
    std::cout << "float2int_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         F(s::ext::intel::math::float2int_ru));
    std::cout << "float2int_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         F(s::ext::intel::math::float2int_rz));
    std::cout << "float2int_rz passes." << std::endl;
  }

  {
    std::initializer_list<float> input_vals = {0.0,
                                               2.5,
                                               23.3457746363636,
                                               12387737.2323233232E+8,
                                               -41.3249382432435,
                                               -9.123233527373837E+17,
                                               9.54E-20,
                                               -3.9872627293918E+16};

    std::initializer_list<long long> ref_vals_rd = {0,   2,
                                                    23,  1238773660647424,
                                                    -42, -912323346559926272,
                                                    0,   -39872625245159424};
    std::initializer_list<long long> ref_vals_rn = {0,   2,
                                                    23,  1238773660647424,
                                                    -41, -912323346559926272,
                                                    0,   -39872625245159424};
    std::initializer_list<long long> ref_vals_ru = {0,   3,
                                                    24,  1238773660647424,
                                                    -41, -912323346559926272,
                                                    1,   -39872625245159424};
    std::initializer_list<long long> ref_vals_rz = {0,   2,
                                                    23,  1238773660647424,
                                                    -41, -912323346559926272,
                                                    0,   -39872625245159424};
    test(device_queue, input_vals, ref_vals_rd,
         F(s::ext::intel::math::float2ll_rd));
    std::cout << "float2ll_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         F(s::ext::intel::math::float2ll_rn));
    std::cout << "float2ll_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         F(s::ext::intel::math::float2ll_ru));
    std::cout << "float2ll_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         F(s::ext::intel::math::float2ll_rz));
    std::cout << "float2ll_rz passes." << std::endl;
  }

  {
    std::initializer_list<float> input_vals = {0.0,
                                               2.5,
                                               23.3457746363636,
                                               12387.23232,
                                               -41.3249382432435,
                                               3.038789876983E+9,
                                               9.54E-20,
                                               4.10487585E+3};

    std::initializer_list<unsigned int> ref_vals_rd = {0, 2,          23, 12387,
                                                       0, 3038789888, 0,  4104};
    std::initializer_list<unsigned int> ref_vals_rn = {0, 2,          23, 12387,
                                                       0, 3038789888, 0,  4105};
    std::initializer_list<unsigned int> ref_vals_ru = {0, 3,          24, 12388,
                                                       0, 3038789888, 1,  4105};
    std::initializer_list<unsigned int> ref_vals_rz = {0, 2,          23, 12387,
                                                       0, 3038789888, 0,  4104};
    test(device_queue, input_vals, ref_vals_rd,
         F(s::ext::intel::math::float2uint_rd));
    std::cout << "float2uint_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         F(s::ext::intel::math::float2uint_rn));
    std::cout << "float2uint_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         F(s::ext::intel::math::float2uint_ru));
    std::cout << "float2uint_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         F(s::ext::intel::math::float2uint_rz));
    std::cout << "float2uint_rz passes." << std::endl;
  }

  {
    std::initializer_list<float> input_vals = {0.0,
                                               2.5,
                                               23.3457746363636,
                                               12387737.2323233232E+8,
                                               -41.3249382432435,
                                               1.7665949E+4,
                                               9.54E-20,
                                               9.9872627232E+16};
    std::initializer_list<unsigned long long> ref_vals_rd = {
        0, 2, 23, 1238773660647424, 0, 17665, 0, 99872626880544768};
    std::initializer_list<unsigned long long> ref_vals_rn = {
        0, 2, 23, 1238773660647424, 0, 17666, 0, 99872626880544768};
    std::initializer_list<unsigned long long> ref_vals_ru = {
        0, 3, 24, 1238773660647424, 0, 17666, 1, 99872626880544768};
    std::initializer_list<unsigned long long> ref_vals_rz = {
        0, 2, 23, 1238773660647424, 0, 17665, 0, 99872626880544768};
    test(device_queue, input_vals, ref_vals_rd,
         F(s::ext::intel::math::float2ull_rd));
    std::cout << "float2ull_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         F(s::ext::intel::math::float2ull_rn));
    std::cout << "float2ull_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         F(s::ext::intel::math::float2ull_ru));
    std::cout << "float2ull_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         F(s::ext::intel::math::float2ull_rz));
    std::cout << "float2ull_rz passes." << std::endl;
  }

  {
    std::initializer_list<float> input_vals = {
        0,           2.5,          23.56774234, -12.43332432434, 9.564324E+25,
        -6.3212E-19, -9.49823E-20, 4.321993E+18};
    std::initializer_list<int> ref_vals = {0,           1075838976, 1102875324,
                                           -1052315931, 1788754557, -1589997149,
                                           -1612692153, 1584393015};
    test(device_queue, input_vals, ref_vals,
         F(s::ext::intel::math::float_as_int));
    std::cout << "float_as_int passes." << std::endl;
  }

  {
    std::initializer_list<float> input_vals = {
        0,           2.5,          23.56774234, -12.43332432434, 9.564324E+25,
        -6.3212E-19, -9.49823E-20, 4.321993E+18};
    std::initializer_list<unsigned int> ref_vals = {
        0,          1075838976, 1102875324, 3242651365,
        1788754557, 2704970147, 2682275143, 1584393015};
    test(device_queue, input_vals, ref_vals,
         F(s::ext::intel::math::float_as_uint));
    std::cout << "float_as_uint passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals = {
        0, 6364, -9383, -783, 958584758, -944866283, 2147483647, -2147483647};
    std::initializer_list<unsigned int> ref_vals_rd = {
        0,          0x45c6e000, 0xc6129c00, 0xc443c000,
        0x4e648b5e, 0xce614610, 0x4effffff, 0xcf000000};
    std::initializer_list<unsigned int> ref_vals_rn = {
        0,          0x45c6e000, 0xc6129c00, 0xc443c000,
        0x4e648b5f, 0xce614610, 0x4f000000, 0xcf000000};
    std::initializer_list<unsigned int> ref_vals_ru = {
        0,          0x45c6e000, 0xc6129c00, 0xc443c000,
        0x4e648b5f, 0xce61460f, 0x4f000000, 0xceffffff};
    std::initializer_list<unsigned int> ref_vals_rz = {
        0,          0x45c6e000, 0xc6129c00, 0xc443c000,
        0x4e648b5e, 0xce61460f, 0x4effffff, 0xceffffff};
    test(device_queue, input_vals, ref_vals_rd,
         FT(unsigned, s::ext::intel::math::int2float_rd));
    std::cout << "int2float_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         FT(unsigned, s::ext::intel::math::int2float_rn));
    std::cout << "int2float_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         FT(unsigned, s::ext::intel::math::int2float_ru));
    std::cout << "int2float_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         FT(unsigned, s::ext::intel::math::int2float_rz));
    std::cout << "int2float_rz passes." << std::endl;
  }

  {
    std::initializer_list<long long> input_vals = {0,
                                                   9383,
                                                   -783,
                                                   2258584758,
                                                   -4284866283,
                                                   9223372036854775807,
                                                   -9223372036854775807,
                                                   9197372036855684605};
    std::initializer_list<unsigned int> ref_vals_rd = {
        0,          0x46129c00, 0xc443c000, 0x4f069f44,
        0xcf7f65df, 0x5effffff, 0xdf000000, 0x5eff4742};
    std::initializer_list<unsigned int> ref_vals_rn = {
        0,          0x46129c00, 0xc443c000, 0x4f069f45,
        0xcf7f65df, 0x5f000000, 0xdf000000, 0x5eff4742};
    std::initializer_list<unsigned int> ref_vals_ru = {
        0,          0x46129c00, 0xc443c000, 0x4f069f45,
        0xcf7f65de, 0x5f000000, 0xdeffffff, 0x5eff4743};
    std::initializer_list<unsigned int> ref_vals_rz = {
        0,          0x46129c00, 0xc443c000, 0x4f069f44,
        0xcf7f65de, 0x5effffff, 0xdeffffff, 0x5eff4742};
    test(device_queue, input_vals, ref_vals_rd,
         FT(unsigned, s::ext::intel::math::ll2float_rd));
    std::cout << "ll2float_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         FT(unsigned, s::ext::intel::math::ll2float_rn));
    std::cout << "ll2float_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         FT(unsigned, s::ext::intel::math::ll2float_ru));
    std::cout << "ll2float_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         FT(unsigned, s::ext::intel::math::ll2float_rz));
    std::cout << "ll2float_rz passes." << std::endl;
  }

  {
    std::initializer_list<unsigned long long> input_vals = {
        0,
        9383,
        783,
        2258584758,
        9284866283883838383U,
        12223372036854775807U,
        18446744073709551615U,
        9197372036855684605U};
    std::initializer_list<unsigned int> ref_vals_rd = {
        0,          0x46129c00, 0x4443c000, 0x4f069f44,
        0x5f00da78, 0x5f29a224, 0x5f7fffff, 0x5eff4742};
    std::initializer_list<unsigned int> ref_vals_rn = {
        0,          0x46129c00, 0x4443c000, 0x4f069f45,
        0x5f00da79, 0x5f29a224, 0x5f800000, 0x5eff4742};
    std::initializer_list<unsigned int> ref_vals_ru = {
        0,          0x46129c00, 0x4443c000, 0x4f069f45,
        0x5f00da79, 0x5f29a225, 0x5f800000, 0x5eff4743};
    std::initializer_list<unsigned int> ref_vals_rz = {
        0,          0x46129c00, 0x4443c000, 0x4f069f44,
        0x5f00da78, 0x5f29a224, 0x5f7fffff, 0x5eff4742};
    test(device_queue, input_vals, ref_vals_rd,
         FT(unsigned, s::ext::intel::math::ull2float_rd));
    std::cout << "ull2float_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         FT(unsigned, s::ext::intel::math::ull2float_rn));
    std::cout << "ull2float_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         FT(unsigned, s::ext::intel::math::ull2float_ru));
    std::cout << "ull2float_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         FT(unsigned, s::ext::intel::math::ull2float_rz));
    std::cout << "ull2float_rz passes." << std::endl;
  }

  {
    std::initializer_list<unsigned int> input_vals = {
        0,         6364,       983283,     7373783,
        958584758, 2844866283, 3947483647, 4294967295};
    std::initializer_list<unsigned int> ref_vals_rd = {
        0,          0x45c6e000, 0x49700f30, 0x4ae107ae,
        0x4e648b5e, 0x4f299136, 0x4f6b49d1, 0x4f7fffff};
    std::initializer_list<unsigned int> ref_vals_rn = {
        0,          0x45c6e000, 0x49700f30, 0x4ae107ae,
        0x4e648b5f, 0x4f299137, 0x4f6b49d2, 0x4f800000};
    std::initializer_list<unsigned int> ref_vals_ru = {
        0,          0x45c6e000, 0x49700f30, 0x4ae107ae,
        0x4e648b5f, 0x4f299137, 0x4f6b49d2, 0x4f800000};
    std::initializer_list<unsigned int> ref_vals_rz = {
        0,          0x45c6e000, 0x49700f30, 0x4ae107ae,
        0x4e648b5e, 0x4f299136, 0x4f6b49d1, 0x4f7fffff};
    test(device_queue, input_vals, ref_vals_rd,
         FT(unsigned, s::ext::intel::math::uint2float_rd));
    std::cout << "uint2float_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         FT(unsigned, s::ext::intel::math::uint2float_rn));
    std::cout << "uint2float_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         FT(unsigned, s::ext::intel::math::uint2float_ru));
    std::cout << "uint2float_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         FT(unsigned, s::ext::intel::math::uint2float_rz));
    std::cout << "uint2float_rz passes." << std::endl;
  }

  {
    std::initializer_list<unsigned int> input_vals = {
        0,          1075838976, 1120485376, 3296331776,
        1094254592, 3255377920, 1120290816, 3319004448};
    std::initializer_list<float> ref_vals = {0,         2.5,         100.625,
                                             -1000.125, 11.5625,     -34.28125,
                                             99.140625, -6783.640625};
    test(device_queue, input_vals, ref_vals,
         F(s::ext::intel::math::uint_as_float));
    std::cout << "uint_as_float passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals = {
        0,          1075838976,  1120485376, -998635520,
        1094254592, -1039589376, 1120290816, -975962848};
    std::initializer_list<float> ref_vals = {0,         2.5,         100.625,
                                             -1000.125, 11.5625,     -34.28125,
                                             99.140625, -6783.640625};
    test(device_queue, input_vals, ref_vals,
         F(s::ext::intel::math::int_as_float));
    std::cout << "int_as_float passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals = {
        0,          1,          0x98324,    0xFFFFFFFF,
        0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
    std::initializer_list<unsigned> ref_vals = {
        0,          0x80000000, 0x24c19000, 0xFFFFFFFF,
        0x19D57B7F, 0x1E6A2C48, 0xE49933DD, 0x6E83D905};
    test(device_queue, input_vals, ref_vals, F(s::ext::intel::math::brev));
    std::cout << "brev passes." << std::endl;
  }

  {
    std::initializer_list<unsigned long long> input_vals = {0,
                                                            1,
                                                            0x98324,
                                                            0xFFFFFFFFFFFFFFFF,
                                                            0xF0CAFEDEAB98DB1C,
                                                            0x6364123456789093,
                                                            0xAAFFBBCC9927DDEE,
                                                            0x35CCA09BC176EE87};
    std::initializer_list<unsigned long long> ref_vals = {0,
                                                          0x8000000000000000,
                                                          0x24C1900000000000,
                                                          0xFFFFFFFFFFFFFFFF,
                                                          0x38DB19D57B7F530F,
                                                          0xC9091E6A2C4826C6,
                                                          0x77BBE49933DDFF55,
                                                          0xE1776E83D90533AC};
    test(device_queue, input_vals, ref_vals, F(s::ext::intel::math::brevll));
    std::cout << "brevll passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals = {
        0, 1, 98737322, -102838382, 2147483647, -2147483648, 840193321, 36361};
    std::initializer_list<int> ref_vals = {32, 31, 5, 0, 1, 0, 2, 16};
    test(device_queue, input_vals, ref_vals, F(s::ext::intel::math::clz));
    std::cout << "clz passes." << std::endl;
  }

  {
    std::initializer_list<long long> input_vals = {0,
                                                   1,
                                                   9223372036854775807,
                                                   -102838382,
                                                   214748364992,
                                                   7882147483648,
                                                   840193321938383338,
                                                   36361};
    std::initializer_list<int> ref_vals = {64, 63, 1, 0, 26, 21, 4, 48};
    test(device_queue, input_vals, ref_vals, F(s::ext::intel::math::clzll));
    std::cout << "clzll passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals = {
        0, 1, 98737328, -102838382, -2147483648, -2047441324, 840193321, 36360};
    std::initializer_list<int> ref_vals = {0, 1, 5, 2, 32, 3, 1, 4};
    test(device_queue, input_vals, ref_vals, F(s::ext::intel::math::ffs));
    std::cout << "ffs passes." << std::endl;
  }

  {
    std::initializer_list<long long> input_vals = {0,
                                                   1,
                                                   9223372036854775807,
                                                   -102838382,
                                                   214748364992,
                                                   7882147483648,
                                                   840193321938383338,
                                                   -92233720368547808};
    std::initializer_list<int> ref_vals = {0, 1, 1, 2, 7, 13, 2, 6};
    test(device_queue, input_vals, ref_vals, F(s::ext::intel::math::ffsll));
    std::cout << "ffsll passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals = {
        0, 1, 98737328, 102838382, 3147483648, 2947441324, 840193321, 36360};
    std::initializer_list<int> ref_vals = {0, 1, 13, 11, 14, 18, 12, 5};
    test(device_queue, input_vals, ref_vals, F(s::ext::intel::math::popc));
    std::cout << "popc passes." << std::endl;
  }

  {
    std::initializer_list<unsigned long long> input_vals = {
        0,
        1,
        10223372036854775807ULL,
        102838382,
        214748364992,
        7882147483648,
        840193321938383338,
        92233720368547808};
    std::initializer_list<int> ref_vals = {0, 1, 42, 11, 5, 17, 37, 28};
    test(device_queue, input_vals, ref_vals, F(s::ext::intel::math::popcll));
    std::cout << "popcll passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals1 = {0,    999,      8283838, -166,
                                              8324, -1123492, 7646211, -654212};
    std::initializer_list<int> input_vals2 = {100,   819933,  -322,  6832322,
                                              88324, 6666483, 92212, 100};
    std::initializer_list<int> ref_vals = {0,           819113067, 1627571460,
                                           -1134165452, 735208976, 682645588,
                                           697772188,   -65421200};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(s::ext::intel::math::mul24));
    std::cout << "mul24 passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals1 = {
        0, 999, 982838387, -166, 8324, -1123492, 7646211, -165421294};
    std::initializer_list<int> input_vals2 = {
        100, 819933, -99222322, 6832322, 88324, 6666483, 92212, 100};
    std::initializer_list<int> ref_vals = {0, 0,     -22705530, -1,
                                           0, -1744, 164,       -4};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(s::ext::intel::math::mulhi));
    std::cout << "mulhi passes." << std::endl;
  }

  {
    std::initializer_list<long long> input_vals1 = {0,
                                                    999,
                                                    8198283838781,
                                                    -1668897765548765876,
                                                    832499998311,
                                                    -1123492,
                                                    76462116766547976,
                                                    -54541165421294};
    std::initializer_list<long long> input_vals2 = {
        100,       819933,       -99222322,          911268323228583722,
        788321204, 111236666483, 992212711123456789, 9100000};
    std::initializer_list<long long> ref_vals = {
        0, 0, -45, -82443474164041221, 35, -1, 4112741190099812, -27};
    std::initializer_list<long long> ref_vals1 = {
        100,          819933,       8198283838781,      911268323228583722,
        832499998311, 111236666483, 992212711123456789, 9100000};
    std::initializer_list<long long> ref_vals2 = {
        0,         999,      -99222322,         -1668897765548765876,
        788321204, -1123492, 76462116766547976, -54541165421294};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(s::ext::intel::math::mul64hi));
    std::cout << "mul64hi passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals1,
          F2(s::ext::intel::math::llmax));
    std::cout << "llmax passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(s::ext::intel::math::llmin));
    std::cout << "llmin passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals1 = {
        0, 999, 8283835, -166, 1832499282, -1123492, -1117646211, 2147483647};
    std::initializer_list<int> input_vals2 = {99,          819933,     -322,
                                              6832322,     1992983587, 6666483,
                                              -2000002194, 2147483646};
    std::initializer_list<int> ref_vals1 = {99,          819933,     8283835,
                                            6832322,     1992983587, 6666483,
                                            -1117646211, 2147483647};
    std::initializer_list<int> ref_vals2 = {
        0, 999, -322, -166, 1832499282, -1123492, -2000002194, 2147483646};
    std::initializer_list<int> ref_vals3 = {50,          410466,     4141757,
                                            3416078,     1912741435, 2771496,
                                            -1558824202, 2147483647};
    std::initializer_list<int> ref_vals4 = {49,          410466,     4141756,
                                            3416078,     1912741434, 2771495,
                                            -1558824203, 2147483646};
    test2(device_queue, input_vals1, input_vals2, ref_vals1,
          F2(s::ext::intel::math::max));
    std::cout << "max passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(s::ext::intel::math::min));
    std::cout << "min passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals3,
          F2(s::ext::intel::math::rhadd));
    std::cout << "rhadd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals4,
          F2(s::ext::intel::math::hadd));
    std::cout << "hadd passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals1 = {
        0,          999,        8283835,    4294967289,
        1832499282, 3991123492, 1117646211, 2147483647};
    std::initializer_list<unsigned> input_vals2 = {
        99,         819933,     99898322,   6832322,
        1992983587, 3896666483, 2000002194, 2147483646};
    std::initializer_list<unsigned> ref_vals = {
        49,         410466,     54091078,   2150899805,
        1912741434, 3943894987, 1558824202, 2147483646};
    std::initializer_list<unsigned int> ref_vals1 = {
        99,         819933,     99898322,   4294967289,
        1992983587, 3991123492, 2000002194, 2147483647};
    std::initializer_list<unsigned int> ref_vals2 = {
        0,          999,        8283835,    6832322,
        1832499282, 3896666483, 1117646211, 2147483646};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(s::ext::intel::math::uhadd));
    std::cout << "uhadd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals1,
          F2(s::ext::intel::math::umax));
    std::cout << "umax passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(s::ext::intel::math::umin));
    std::cout << "umin passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals1 = {
        0, 999, 828388, 16666107, 8324, 1123492, 7646211, 16777215};
    std::initializer_list<unsigned> input_vals2 = {
        100, 819933, 8333107, 6832322, 88324, 6666483, 92212, 100};
    std::initializer_list<unsigned> ref_vals = {
        0,         819113067,  1033396844, 36558902,
        735208976, 3612321708, 697772188,  1677721500};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(s::ext::intel::math::umul24));
    std::cout << "umul24 passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals1 = {
        0, 999, 982838387, 9166, 8324, 1123492, 7646211, 165421294};
    std::initializer_list<unsigned> input_vals2 = {
        100, 78199338, 99222322, 796832322, 88324, 6666483, 92212, 4294967295};
    std::initializer_list<unsigned> ref_vals = {0, 18,   22705529, 1700,
                                                0, 1743, 164,      165421293};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(s::ext::intel::math::umulhi));
    std::cout << "umulhi passes." << std::endl;
  }

  {
    std::initializer_list<unsigned long long> input_vals1 = {
        0,
        999,
        819828383878191124,
        12668897765548765876ULL,
        832499998311,
        1123492,
        76462116766547976,
        9769854541165421294ULL};
    std::initializer_list<unsigned long long> input_vals2 = {
        100,       819933,       1199217699222322,   911268323228583722,
        788321204, 111236666483, 992212711123456789, 910008700};
    std::initializer_list<unsigned long long> ref_vals = {
        0,  0, 53296815109651,   625842976832950555,
        35, 0, 4112741190099812, 481963244};
    std::initializer_list<unsigned long long> ref_vals1 = {
        100,          819933,       819828383878191124, 12668897765548765876ULL,
        832499998311, 111236666483, 992212711123456789, 9769854541165421294ULL};
    std::initializer_list<unsigned long long> ref_vals2 = {
        0,         999,     1199217699222322,  911268323228583722,
        788321204, 1123492, 76462116766547976, 910008700};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(s::ext::intel::math::umul64hi));
    std::cout << "umul64hi passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals1,
          F2(s::ext::intel::math::ullmax));
    std::cout << "ullmax passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(s::ext::intel::math::ullmin));
    std::cout << "ullmin passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals1 = {
        0,          999,     1828383509, 4294967295,
        1832499282, 1123492, 3117646211, 3147483647};
    std::initializer_list<unsigned> input_vals2 = {
        99,         819933,  1322009876, 16832322,
        3992983587, 6666483, 2000002194, 4147483646};
    std::initializer_list<unsigned> ref_vals = {
        50,         410466,  1575196693, 2155899809,
        2912741435, 3894988, 2558824203, 3647483647};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(s::ext::intel::math::urhadd));
    std::cout << "urhadd passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals1 = {
        0,          1,          0x98324,    0xFFFFFFFF,
        0xFEDEAB98, 0x12345678, 0xBBCC9927, 0xA09BC176};
    std::initializer_list<unsigned> input_vals2 = {
        0,          1,          0x8372833,  0xFFFFFFFF,
        0xA8372992, 0xBCCFED83, 0x9AA337BE, 0x73821207};
    std::initializer_list<unsigned> input_vals3 = {
        0x98C688A1, 0x9912FFEA, 0x0,        0xABCD9871,
        0x8976BAC1, 0xCC901ABC, 0xBCEFF681, 0x99AACEDF};
    std::initializer_list<unsigned> ref_vals = {
        0,          0,          0x24242424, 0xFFFFFFFF,
        0xFEDE92AB, 0x56341283, 0x9aa32799, 0x7821273};
    test3(device_queue, input_vals1, input_vals2, input_vals3, ref_vals,
          F3(s::ext::intel::math::byte_perm));
    std::cout << "byte_perm passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals1 = {
        0, 999, 8283835, -166, 1832499282, -1123492, -1117646211, 2147483647};
    std::initializer_list<int> input_vals2 = {99,          819933,     -322,
                                              6832322,     1992983587, 6666483,
                                              -2000002194, 2147483646};
    std::initializer_list<unsigned> input_vals3 = {
        2147483648, 3000000100, 200, 10224, 3200000001, 4000109871, 0, 1};
    std::initializer_list<unsigned> ref_vals = {
        2147483747, 3000819034, 8284357,   6842712,
        3360484306, 4007899846, 882355983, 2};
    test3(device_queue, input_vals1, input_vals2, input_vals3, ref_vals,
          F3(s::ext::intel::math::sad));
    std::cout << "sad passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals1 = {
        0,          999,     8283835,    3987654166,
        1832499282, 1123492, 1117646211, 3147483647};
    std::initializer_list<unsigned> input_vals2 = {
        99,         4294967295, 322,        6832322,
        1992983587, 6666483,    2000002194, 2147483646};
    std::initializer_list<unsigned> input_vals3 = {
        2147483648, 12, 200, 10224, 3200000001, 4000109871, 0, 1};
    std::initializer_list<unsigned> ref_vals = {
        2147483747, 4294966308, 8283713,   3980832068,
        3360484306, 4005652862, 882355983, 1000000002};
    test3(device_queue, input_vals1, input_vals2, input_vals3, ref_vals,
          F3(s::ext::intel::math::usad));
    std::cout << "usad passes." << std::endl;
  }

  return 0;
}
