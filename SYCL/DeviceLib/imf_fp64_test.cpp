// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fno-builtin -fsycl-device-lib-jit-link %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// UNSUPPORTED: cuda || hip

#include "imf_utils.hpp"

namespace s = sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

extern "C" {
float __imf_double2float_rd(double);
float __imf_double2float_rn(double);
float __imf_double2float_ru(double);
float __imf_double2float_rz(double);
int __imf_double2hiint(double);
int __imf_double2loint(double);
int __imf_double2int_rd(double);
int __imf_double2int_rn(double);
int __imf_double2int_ru(double);
int __imf_double2int_rz(double);
long long __imf_double2ll_rd(double);
long long __imf_double2ll_rn(double);
long long __imf_double2ll_ru(double);
long long __imf_double2ll_rz(double);
unsigned int __imf_double2uint_rd(double);
unsigned int __imf_double2uint_rn(double);
unsigned int __imf_double2uint_ru(double);
unsigned int __imf_double2uint_rz(double);
unsigned long long __imf_double2ull_rd(double);
unsigned long long __imf_double2ull_rn(double);
unsigned long long __imf_double2ull_ru(double);
unsigned long long __imf_double2ull_rz(double);
long long __imf_double_as_longlong(double);
double __imf_hiloint2double(int, int);
double __imf_int2double_rn(int);
double __imf_ll2double_rd(long long);
double __imf_ll2double_rn(long long);
double __imf_ll2double_ru(long long);
double __imf_ll2double_rz(long long);
double __imf_longlong_as_double(long long);
double __imf_uint2double_rn(unsigned);
double __imf_ull2double_rd(unsigned long long);
double __imf_ull2double_rn(unsigned long long);
double __imf_ull2double_ru(unsigned long long);
double __imf_ull2double_rz(unsigned long long);
}

int main(int, char **) {
  s::default_selector device_selector;
  s::queue device_queue(device_selector);
  std::cout << "Running on "
            << device_queue.get_device().get_info<s::info::device::name>()
            << "\n";
  if (!device_queue.get_device().has(sycl::aspect::fp64)) {
    std::cout << "Test skipped on platform without fp64 support." << std::endl;
    return 0;
  }

  {
    std::initializer_list<double> input_vals = {0.0,
                                                2.5,
                                                23.3457746363636,
                                                123877377737.2323233232,
                                                -41.3249382432435,
                                                -6.7543E+25,
                                                9.54E-20,
                                                -3.12E+19};
    std::initializer_list<unsigned> ref_vals_rd = {
        0,          0x40200000, 0x41bac425, 0x51e6bd56,
        0xc2254cbd, 0xea5f7b26, 0x1fe141c2, 0xdfd87e56};
    std::initializer_list<unsigned> ref_vals_rn = {
        0,          0x40200000, 0x41bac425, 0x51e6bd56,
        0xc2254cbd, 0xea5f7b25, 0x1fe141c3, 0xdfd87e55};
    std::initializer_list<unsigned> ref_vals_ru = {
        0,          0x40200000, 0x41bac426, 0x51e6bd57,
        0xc2254cbc, 0xea5f7b25, 0x1fe141c3, 0xdfd87e55};
    std::initializer_list<unsigned> ref_vals_rz = {
        0,          0x40200000, 0x41bac425, 0x51e6bd56,
        0xc2254cbc, 0xea5f7b25, 0x1fe141c2, 0xdfd87e55};
    test(device_queue, input_vals, ref_vals_rd,
         FT(unsigned, __imf_double2float_rd));
    std::cout << "double2float_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         FT(unsigned, __imf_double2float_rn));
    std::cout << "double2float_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         FT(unsigned, __imf_double2float_ru));
    std::cout << "double2float_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         FT(unsigned, __imf_double2float_rz));
    std::cout << "double2float_rz passes." << std::endl;
  }

  {
    std::initializer_list<double> input_vals = {
        0,           2.5,           23.56774234, -12.43332432434, 9.564324E+255,
        -6.3212E-99, -9.49823E-201, 4.321993E+88};
    std::initializer_list<unsigned> ref_vals_hi = {
        0,          0x40040000, 0x40379157, 0xc028dddc,
        0x7514622a, 0xab8ba6ef, 0x966743ee, 0x5255b9ed};
    std::initializer_list<unsigned> ref_vals_lo = {
        0,          0,          0x8fdedac1, 0xaf933404,
        0x90d084df, 0xee71d6f2, 0xa5a053df, 0x4db8d884};
    test(device_queue, input_vals, ref_vals_hi, F(__imf_double2hiint));
    std::cout << "double2hiint passes." << std::endl;
    test(device_queue, input_vals, ref_vals_lo, F(__imf_double2loint));
    std::cout << "double2loint passes." << std::endl;
  }

  {
    std::initializer_list<double> input_vals = {0.0,
                                                2.5,
                                                23.3457746363636,
                                                12387737.2323233232,
                                                -41.3249382432435,
                                                -9.123233526E+8,
                                                9.54E-20,
                                                -3.9872627293918E+8};
    std::initializer_list<int> ref_vals_rd = {0,   2,          23, 12387737,
                                              -42, -912323353, 0,  -398726273};
    std::initializer_list<int> ref_vals_rn = {0,   2,          23, 12387737,
                                              -41, -912323353, 0,  -398726273};
    std::initializer_list<int> ref_vals_ru = {0,   3,          24, 12387738,
                                              -41, -912323352, 1,  -398726272};
    std::initializer_list<int> ref_vals_rz = {0,   2,          23, 12387737,
                                              -41, -912323352, 0,  -398726272};
    test(device_queue, input_vals, ref_vals_rd, F(__imf_double2int_rd));
    std::cout << "double2int_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn, F(__imf_double2int_rn));
    std::cout << "double2int_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru, F(__imf_double2int_ru));
    std::cout << "double2int_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz, F(__imf_double2int_rz));
    std::cout << "double2int_rz passes." << std::endl;
  }

  {
    std::initializer_list<double> input_vals = {0.0,
                                                2.5,
                                                23.3457746363636,
                                                12387737.2323233232E+8,
                                                -41.3249382432435,
                                                -9.123233527373837E+17,
                                                9.54E-20,
                                                -3.9872627293918E+16};
    std::initializer_list<long long> ref_vals_rd = {0,   2,
                                                    23,  1238773723232332,
                                                    -42, -912323352737383680,
                                                    0,   -39872627293918000};
    std::initializer_list<long long> ref_vals_rn = {0,   2,
                                                    23,  1238773723232332,
                                                    -41, -912323352737383680,
                                                    0,   -39872627293918000};
    std::initializer_list<long long> ref_vals_ru = {0,   3,
                                                    24,  1238773723232333,
                                                    -41, -912323352737383680,
                                                    1,   -39872627293918000};
    std::initializer_list<long long> ref_vals_rz = {0,   2,
                                                    23,  1238773723232332,
                                                    -41, -912323352737383680,
                                                    0,   -39872627293918000};
    test(device_queue, input_vals, ref_vals_rd, F(__imf_double2ll_rd));
    std::cout << "double2ll_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn, F(__imf_double2ll_rn));
    std::cout << "double2ll_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru, F(__imf_double2ll_ru));
    std::cout << "double2ll_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz, F(__imf_double2ll_rz));
    std::cout << "double2ll_rz passes." << std::endl;
  }

  {
    std::initializer_list<double> input_vals = {0.0,
                                                2.5,
                                                23.3457746363636,
                                                12387737.2323233232,
                                                -41.3249382432435,
                                                3.038789876983E+9,
                                                9.54E-20,
                                                4.1048756785435E+9};
    std::initializer_list<unsigned> ref_vals_rd = {
        0, 2, 23, 12387737, 0, 3038789876, 0, 4104875678};
    std::initializer_list<unsigned> ref_vals_rn = {
        0, 2, 23, 12387737, 0, 3038789877, 0, 4104875679};
    std::initializer_list<unsigned> ref_vals_ru = {
        0, 3, 24, 12387738, 0, 3038789877, 1, 4104875679};
    std::initializer_list<unsigned> ref_vals_rz = {
        0, 2, 23, 12387737, 0, 3038789876, 0, 4104875678};
    test(device_queue, input_vals, ref_vals_rd, F(__imf_double2uint_rd));
    std::cout << "double2uint_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn, F(__imf_double2uint_rn));
    std::cout << "double2uint_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru, F(__imf_double2uint_ru));
    std::cout << "double2uint_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz, F(__imf_double2uint_rz));
    std::cout << "double2uint_rz passes." << std::endl;
  }

  {
    std::initializer_list<double> input_vals = {0.0,
                                                2.5,
                                                23.3457746363636,
                                                12387737.2323233232E+8,
                                                -41.3249382432435,
                                                1.766559499488933883949E+19,
                                                9.54E-20,
                                                9.987262729391848483932E+18};
    std::initializer_list<unsigned long long> ref_vals_rd = {
        0,  2,
        23, 1238773723232332,
        0,  17665594994889338880ULL,
        0,  9987262729391849472ULL};
    std::initializer_list<unsigned long long> ref_vals_rn = {
        0,  2,
        23, 1238773723232332,
        0,  17665594994889338880ULL,
        0,  9987262729391849472ULL};
    std::initializer_list<unsigned long long> ref_vals_ru = {
        0,  3,
        24, 1238773723232333,
        0,  17665594994889338880ULL,
        1,  9987262729391849472ULL};
    std::initializer_list<unsigned long long> ref_vals_rz = {
        0,  2,
        23, 1238773723232332,
        0,  17665594994889338880ULL,
        0,  9987262729391849472ULL};
    test(device_queue, input_vals, ref_vals_rd, F(__imf_double2ull_rd));
    std::cout << "double2ull_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn, F(__imf_double2ull_rn));
    std::cout << "double2ull_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru, F(__imf_double2ull_ru));
    std::cout << "double2ull_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz, F(__imf_double2ull_rz));
    std::cout << "double2ull_rz passes." << std::endl;
  }

  {
    std::initializer_list<double> input_vals = {
        0,           2.5,           23.56774234, -12.43332432434, 9.564324E+255,
        -6.3212E-99, -9.49823E-201, 4.321993E+88};
    std::initializer_list<long long> ref_vals = {0,
                                                 4612811918334230528,
                                                 4627326947408403137,
                                                 -4600183079450758140,
                                                 8436475936929514719,
                                                 -6085586922033260814,
                                                 -7609038353159138337,
                                                 5932852512992843908};
    test(device_queue, input_vals, ref_vals, F(__imf_double_as_longlong));
    std::cout << "double_as_longlong passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals1 = {1074003968, -1059083256,
                                              1100450857, -1022596250};
    std::initializer_list<int> input_vals2 = {0, 0, -528482304, 696011200};
    std::initializer_list<double> ref_vals = {2.5, -32432.125, 98765432.125,
                                              -999923234343224};
    test2(device_queue, input_vals1, input_vals2, ref_vals,
          F2(__imf_hiloint2double));
    std::cout << "hiloint2double passes." << std::endl;
  }

  {
    std::initializer_list<int> input_vals = {
        0, 254, -245, 98773733, -85747332, 2147483647, -2147483648, 1023241323};
    std::initializer_list<unsigned long long> ref_vals = {0,
                                                          0x406fc00000000000,
                                                          0xc06ea00000000000,
                                                          0x41978cab94000000,
                                                          0xc194719a10000000,
                                                          0x41dfffffffc00000,
                                                          0xc1e0000000000000,
                                                          0x41ce7eb635800000};
    test(device_queue, input_vals, ref_vals,
         FT(unsigned long long, __imf_int2double_rn));
    std::cout << "int2double_rn passes." << std::endl;
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
    std::initializer_list<unsigned long long> ref_vals_rd = {
        0,
        0x40c2538000000000,
        0xc088780000000000,
        0x41e0d3e896c00000,
        0xc1efecbbdd600000,
        0x43dfffffffffffff,
        0xc3e0000000000000,
        0x43dfe8e848d0f937};
    std::initializer_list<unsigned long long> ref_vals_rn = {
        0,
        0x40c2538000000000,
        0xc088780000000000,
        0x41e0d3e896c00000,
        0xc1efecbbdd600000,
        0x43e0000000000000,
        0xc3e0000000000000,
        0x43dfe8e848d0f937};
    std::initializer_list<unsigned long long> ref_vals_ru = {
        0,
        0x40c2538000000000,
        0xc088780000000000,
        0x41e0d3e896c00000,
        0xc1efecbbdd600000,
        0x43e0000000000000,
        0xc3dfffffffffffff,
        0x43dfe8e848d0f938};
    std::initializer_list<unsigned long long> ref_vals_rz = {
        0,
        0x40c2538000000000,
        0xc088780000000000,
        0x41e0d3e896c00000,
        0xc1efecbbdd600000,
        0x43dfffffffffffff,
        0xc3dfffffffffffff,
        0x43dfe8e848d0f937};
    test(device_queue, input_vals, ref_vals_rd,
         FT(unsigned long long, __imf_ll2double_rd));
    std::cout << "ll2double_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         FT(unsigned long long, __imf_ll2double_rn));
    std::cout << "ll2double_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         FT(unsigned long long, __imf_ll2double_ru));
    std::cout << "ll2double_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         FT(unsigned long long, __imf_ll2double_rz));
    std::cout << "ll2double_rz passes." << std::endl;
  }

  {
    std::initializer_list<long long> input_vals = {0,
                                                   4612811918334230528,
                                                   4636781271819747328,
                                                   -4571363628501958656,
                                                   4622698726891192320,
                                                   -4593350562522595328,
                                                   4636676818215108608,
                                                   -4559191330407841792};
    std::initializer_list<double> ref_vals = {0,         2.5,         100.625,
                                              -1000.125, 11.5625,     -34.28125,
                                              99.140625, -6783.640625};
    test(device_queue, input_vals, ref_vals, F(__imf_longlong_as_double));
    std::cout << "longlong_as_double passes." << std::endl;
  }

  {
    std::initializer_list<unsigned> input_vals = {
        0, 254, 10233, 98773733, 989299343, 2147483647, 4294967295, 3023241323};
    std::initializer_list<unsigned long long> ref_vals = {0,
                                                          0x406fc00000000000,
                                                          0x40c3fc8000000000,
                                                          0x41978cab94000000,
                                                          0x41cd7bc147800000,
                                                          0x41dfffffffc00000,
                                                          0x41efffffffe00000,
                                                          0x41e686600d600000};
    test(device_queue, input_vals, ref_vals,
         FT(unsigned long long, __imf_uint2double_rn));
    std::cout << "uint2double_rn passes." << std::endl;
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
    std::initializer_list<unsigned long long> ref_vals_rd = {
        0,
        0x40c2538000000000,
        0x4088780000000000,
        0x41e0d3e896c00000,
        0x43e01b4f1623bcb7,
        0x43e53444835ec57f,
        0x43efffffffffffff,
        0x43dfe8e848d0f937};
    std::initializer_list<unsigned long long> ref_vals_rn = {
        0,
        0x40c2538000000000,
        0x4088780000000000,
        0x41e0d3e896c00000,
        0x43e01b4f1623bcb7,
        0x43e53444835ec580,
        0x43f0000000000000,
        0x43dfe8e848d0f937};
    std::initializer_list<unsigned long long> ref_vals_ru = {
        0,
        0x40c2538000000000,
        0x4088780000000000,
        0x41e0d3e896c00000,
        0x43e01b4f1623bcb8,
        0x43e53444835ec580,
        0x43f0000000000000,
        0x43dfe8e848d0f938};
    std::initializer_list<unsigned long long> ref_vals_rz = {
        0,
        0x40c2538000000000,
        0x4088780000000000,
        0x41e0d3e896c00000,
        0x43e01b4f1623bcb7,
        0x43e53444835ec57f,
        0x43efffffffffffff,
        0x43dfe8e848d0f937};
    test(device_queue, input_vals, ref_vals_rd,
         FT(unsigned long long, __imf_ull2double_rd));
    std::cout << "ull2double_rd passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rn,
         FT(unsigned long long, __imf_ull2double_rn));
    std::cout << "ull2double_rn passes." << std::endl;
    test(device_queue, input_vals, ref_vals_ru,
         FT(unsigned long long, __imf_ull2double_ru));
    std::cout << "ull2double_ru passes." << std::endl;
    test(device_queue, input_vals, ref_vals_rz,
         FT(unsigned long long, __imf_ull2double_rz));
    std::cout << "ull2double_rz passes." << std::endl;
  }

  return 0;
}
