// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out


// RUN: %{build} -DDES -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -DDES -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out


// RUN: %{build} -DSPREAD -o %t.out
// RUN: %{run} %t.out

// RUN: %{build}  -DSPREAD -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DDES -DSPREAD -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -DDES -DSPREAD -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda || hip || cpu

#include "group_private_KV_sort_p1p1_p3.hpp"

int main() {
  queue q;

  {
    constexpr static int NUM = 35;
    uint16_t ikeys[NUM] = {1,   11,  1,   9,   3,  100, 34,  8,  121,
                          77,  125, 23,  36,  2,  111, 91,  88, 2,
                          51,  213, 181, 183, 31, 142, 216, 1,  199,
                          124, 12,  0,   181, 17, 15,  101, 44};
    uint32_t ivals[NUM] = {99,   32,    1,    2,      67,   9123, 453,
                           435,  91111, 777,  165,    145,  2456, 88811,
                           761,  96,    765,  10000,  6364, 90,   525,
                           882,  1,     2423, 9,      4324, 9123, 0,
                           1232, 777,   555,  314159, 905,  9831, 84341};
    uint8_t ivals1[NUM] = {99, 32,  1,   2,  67, 91,  45, 43,  91,  77,  16, 14,
                           24, 88,  76,  96, 76, 100, 63, 90,  52,  82,  1,  22,
                           9,  225, 127, 0,  12, 128, 3,  102, 200, 111, 123};
    int8_t ivals2[NUM] = {-99, 127, -121, 100, 9,  5,  12,  35,  -98,
                          77,  112, -91,  11,  12, 3,  71,  -66, 121,
                          18,  14,  21,   -22, 54, 88, -81, 31,  23,
                          53,  97,  103,  71,  83, 97, 37,  -41};
    uint16_t ivals3[NUM] = {28831, 23870, 54250, 5022,  9571,  60147, 9554,
                            18818, 28689, 18229, 40512, 23200, 40454, 24841,
                            43251, 63264, 29448, 45917, 882,   30788, 7586,
                            57541, 22108, 59535, 31880, 7152,  63919, 58703,
                            14686, 29914, 5872,  35868, 51479, 22721, 50927};
    int16_t ivals4[NUM] = {
        2798,  -13656, 1592,   3992,  -25870, 25172,  7761,   -18347, 1617,
        25472, 26763,  -5982,  24791, 27189,  22911,  22502,  15801,  25326,
        -2196, 9205,   -10418, 20464, -16616, -11285, 7249,   22866,  30574,
        -1298, 31351,  28252,  21322, -10072, 7874,   -26785, 22016};

    uint32_t ivals5[NUM] = {
        2238578408, 102907035,  2316773768, 617902655,  532045482,  73173328,
        1862406505, 142735533,  3494078873, 610196959,  4210902254, 1863122236,
        1257721692, 30008197,   3199012044, 3503276708, 3504950001, 1240383071,
        2463430884, 904104390,  4044803029, 3164373711, 1586440767, 1999536602,
        3377662770, 927279985,  1740225703, 1133653675, 3975816601, 260339911,
        1115507520, 2279020820, 4289105012, 692964674,  53775301};

    int32_t ivals6[NUM] = {
        507394811,   1949685322, 1624859474, -940434061,  -1440675113,
        -2002743224, 369969519,  840772268,  224522238,   296113452,
        -714007528,  480713824,  665592454,  1696360848,  780843358,
        -1901994531, 1667711523, 1390737696, 1357434904,  -290165630,
        305128121,   1301489180, 630469211,  -1385846315, 809333959,
        1098974670,  56900257,   876775101,  -1496897817, 1172877939,
        1528916082,  559152364,  749878571,  2071902702,  -430851798};

    uint64_t ivals7[NUM] = {7916688577774406903ULL,  16873442000174444235ULL,
                            5261140449171682429ULL,  6274209348061756377ULL,
                            17881284978944229367ULL, 4701456380424752599ULL,
                            6241062870187348613ULL,  8972524466137433448ULL,
                            468629112944127776ULL,   17523909311582893643ULL,
                            17447772191733931166ULL, 14311152396797789854ULL,
                            9265327272409079662ULL,  9958475911404242556ULL,
                            15359829357736068242ULL, 11416531655738519189ULL,
                            16839972510321195914ULL, 1927049095689256442ULL,
                            3356565661065236628ULL,  1065114285796701562ULL,
                            7071763288904613033ULL,  16473053015775147286ULL,
                            10317354477399696817ULL, 16005969584273256379ULL,
                            15391010921709289298ULL, 17671303749287233862ULL,
                            8028596930095411867ULL,  10265936863337610975ULL,
                            17403745948359398749ULL, 8504886230707230194ULL,
                            12855085916215721214ULL, 5562885793068933146ULL,
                            1508385574711135517ULL,  5953119477818575536ULL,
                            9165320150094769334ULL};

    int64_t ivals8[NUM] = {
        2944696543084623337,  137239101631340692,  4370169869966467498,
        3842452903439153631,  -795080033670806202, 3023506421574592237,
        -4142692575864168559, 1716333381567689984, 1591746912204250089,
        -1974664662220599925, 3144022139297218102, -371429365537296255,
        4202906659701034264,  3878513012313576184, -3425767072006791628,
        -2929337291418891626, 1880013370888913338, 1498977159463939728,
        -2928775660744278650, 4074214991200977615, 4291797122649374026,
        -763110360214750992,  2883673064242977727, 4270151072450399778,
        1408696225027958214,  1264214335825459628, -4152065441956669638,
        2684706424226400837,  569335474272794084,  -2798088842177577224,
        814002749054152728,   2517003920904582842, 4089891582575745386,
        705067059635512048,   -2500935118374519236};

    auto work_group_sorter = [](uint16_t *keys, uint32_t *vals, uint32_t n,
                                uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u32_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1u32_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u32_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u32_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    auto work_group_sorter1 = [](uint16_t *keys, uint8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u8_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1u8_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u8_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u8_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    auto work_group_sorter2 = [](uint16_t *keys, int8_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1i8_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1i8_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1i8_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1i8_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    auto work_group_sorter3 = [](uint16_t *keys, uint16_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u16_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1u16_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u16_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u16_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    auto work_group_sorter4 = [](uint16_t *keys, int16_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1i16_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1i16_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1i16_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1i16_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    auto work_group_sorter5 = [](uint16_t *keys, uint32_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u32_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1u32_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u32_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u32_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    auto work_group_sorter6 = [](uint16_t *keys, int32_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1i32_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1i32_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1i32_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1i32_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    auto work_group_sorter7 = [](uint16_t *keys, uint64_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1u64_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1u64_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1u64_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1u64_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    auto work_group_sorter8 = [](uint16_t *keys, int64_t *vals, uint32_t n,
                                 uint8_t *scratch) {
#if __DEVICE_CODE
#ifdef DES
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_descending_p1u16_p1i64_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_descending_p1u16_p1i64_u32_p3i8(
          keys, vals, n, scratch);
#endif
#else
#ifdef SPREAD
      __devicelib_default_work_group_private_sort_spread_ascending_p1u16_p1i64_u32_p3i8(
          keys, vals, n, scratch);
#else
      __devicelib_default_work_group_private_sort_close_ascending_p1u16_p1i64_u32_p3i8(
          keys, vals, n, scratch);
#endif
#endif
#endif
    };

    constexpr static int NUM1 = 32;
    test_work_group_KV_private_sort<uint16_t, uint8_t, 1, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint16_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint8_t, 2, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint16_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint8_t, 4, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint16_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 4 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint8_t, 8, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint16_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 8 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint8_t, 16, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint16_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 16 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint8_t, 32, NUM1,
                                    decltype(work_group_sorter1)>(
        q, ikeys, ivals1, work_group_sorter1);
    std::cout << "KV private sort <Key: uint16_t, Val: uint8_t> NUM = " << NUM1
              << ", WG = 32 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int8_t, 1, NUM,
                                    decltype(work_group_sorter2)>(
        q, ikeys, ivals2, work_group_sorter2);
    std::cout << "KV private sort <Key: uint16_t, Val: int8_t> NUM = " << NUM
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int8_t, 5, NUM,
                                    decltype(work_group_sorter2)>(
        q, ikeys, ivals2, work_group_sorter2);
    std::cout << "KV private sort <Key: uint16_t, Val: int8_t> NUM = " << NUM
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int8_t, 7, NUM,
                                    decltype(work_group_sorter2)>(
        q, ikeys, ivals2, work_group_sorter2);
    std::cout << "KV private sort <Key: uint16_t, Val: int8_t> NUM = " << NUM
              << ", WG = 7 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int8_t, 35, NUM,
                                    decltype(work_group_sorter2)>(
        q, ikeys, ivals2, work_group_sorter2);
    std::cout << "KV private sort <Key: uint16_t, Val: int8_t> NUM = " << NUM
              << ", WG = 35 pass." << std::endl;

    constexpr static int NUM2 = 24;
    test_work_group_KV_private_sort<uint16_t, uint16_t, 1, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint16_t, 2, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint16_t, 3, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 3 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint16_t, 4, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 4 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint16_t, 6, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 6 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint16_t, 8, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 8 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint16_t, 12, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 12 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint16_t, 24, NUM2,
                                    decltype(work_group_sorter3)>(
        q, ikeys, ivals3, work_group_sorter3);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM2
              << ", WG = 24 pass." << std::endl;

    constexpr static int NUM3 = 20;
    test_work_group_KV_private_sort<uint16_t, int16_t, 1, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint16_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int16_t, 2, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint16_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int16_t, 4, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint16_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 4 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int16_t, 5, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint16_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int16_t, 10, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint16_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 10 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int16_t, 20, NUM3,
                                    decltype(work_group_sorter4)>(
        q, ikeys, ivals4, work_group_sorter4);
    std::cout << "KV private sort <Key: uint16_t, Val: int16_t> NUM = " << NUM3
              << ", WG = 20 pass." << std::endl;

    constexpr static int NUM4 = 30;
    test_work_group_KV_private_sort<uint16_t, uint32_t, 1, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint16_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint32_t, 2, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint16_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint32_t, 3, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint16_t, Val: uint16_t> NUM = " << NUM4
              << ", WG = 3 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint32_t, 5, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint16_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint32_t, 6, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint16_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 6 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint32_t, 10, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint16_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 10 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint32_t, 15, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint16_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 15 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint32_t, 30, NUM4,
                                    decltype(work_group_sorter5)>(
        q, ikeys, ivals5, work_group_sorter5);
    std::cout << "KV private sort <Key: uint16_t, Val: uint32_t> NUM = " << NUM4
              << ", WG = 30 pass." << std::endl;

    constexpr static int NUM5 = 25;
    test_work_group_KV_private_sort<uint16_t, int32_t, 1, NUM5,
                                    decltype(work_group_sorter6)>(
        q, ikeys, ivals6, work_group_sorter6);
    std::cout << "KV private sort <Key: uint16_t, Val: int32_t> NUM = " << NUM5
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int32_t, 5, NUM5,
                                    decltype(work_group_sorter6)>(
        q, ikeys, ivals6, work_group_sorter6);
    std::cout << "KV private sort <Key: uint16_t, Val: int32_t> NUM = " << NUM5
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int32_t, 25, NUM5,
                                    decltype(work_group_sorter6)>(
        q, ikeys, ivals6, work_group_sorter6);
    std::cout << "KV private sort <Key: uint16_t, Val: int32_t> NUM = " << NUM5
              << ", WG = 25 pass." << std::endl;

    constexpr static int NUM6 = 30;
    test_work_group_KV_private_sort<uint16_t, uint64_t, 1, NUM6,
                                    decltype(work_group_sorter7)>(
        q, ikeys, ivals7, work_group_sorter7);
    std::cout << "KV private sort <Key: uint16_t, Val: uint64_t> NUM = " << NUM6
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint64_t, 2, NUM6,
                                    decltype(work_group_sorter7)>(
        q, ikeys, ivals7, work_group_sorter7);
    std::cout << "KV private sort <Key: uint16_t, Val: uint64_t> NUM = " << NUM6
              << ", WG = 2 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint64_t, 3, NUM6,
                                    decltype(work_group_sorter7)>(
        q, ikeys, ivals7, work_group_sorter7);
    std::cout << "KV private sort <Key: uint16_t, Val: uint64_t> NUM = " << NUM6
              << ", WG = 3 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint64_t, 5, NUM6,
                                    decltype(work_group_sorter7)>(
        q, ikeys, ivals7, work_group_sorter7);
    std::cout << "KV private sort <Key: uint16_t, Val: uint64_t> NUM = " << NUM6
              << ", WG = 5 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint64_t, 6, NUM6,
                                    decltype(work_group_sorter7)>(
        q, ikeys, ivals7, work_group_sorter7);
    std::cout << "KV private sort <Key: uint16_t, Val: uint64_t> NUM = " << NUM6
              << ", WG = 6 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint64_t, 10, NUM6,
                                    decltype(work_group_sorter7)>(
        q, ikeys, ivals7, work_group_sorter7);
    std::cout << "KV private sort <Key: uint16_t, Val: uint64_t> NUM = " << NUM6
              << ", WG = 10 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint64_t, 15, NUM6,
                                    decltype(work_group_sorter7)>(
        q, ikeys, ivals7, work_group_sorter7);
    std::cout << "KV private sort <Key: uint16_t, Val: uint64_t> NUM = " << NUM6
              << ", WG = 15 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, uint64_t, 30, NUM6,
                                    decltype(work_group_sorter7)>(
        q, ikeys, ivals7, work_group_sorter7);
    std::cout << "KV private sort <Key: uint16_t, Val: uint64_t> NUM = " << NUM6
              << ", WG = 30 pass." << std::endl;

    constexpr static int NUM7 = 21;
    test_work_group_KV_private_sort<uint16_t, int64_t, 1, NUM7,
                                    decltype(work_group_sorter8)>(
        q, ikeys, ivals8, work_group_sorter8);
    std::cout << "KV private sort <Key: uint16_t, Val: int64_t> NUM = " << NUM7
              << ", WG = 1 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int64_t, 3, NUM7,
                                    decltype(work_group_sorter8)>(
        q, ikeys, ivals8, work_group_sorter8);
    std::cout << "KV private sort <Key: uint16_t, Val: int64_t> NUM = " << NUM7
              << ", WG = 3 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int64_t, 7, NUM7,
                                    decltype(work_group_sorter8)>(
        q, ikeys, ivals8, work_group_sorter8);
    std::cout << "KV private sort <Key: uint16_t, Val: int64_t> NUM = " << NUM7
              << ", WG = 7 pass." << std::endl;

    test_work_group_KV_private_sort<uint16_t, int64_t, 21, NUM7,
                                    decltype(work_group_sorter8)>(
        q, ikeys, ivals8, work_group_sorter8);
    std::cout << "KV private sort <Key: uint16_t, Val: int64_t> NUM = " << NUM7
              << ", WG = 21 pass." << std::endl;
  }
}
