// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// UNSUPPORTED: cuda || hip

#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>

namespace sycl_imf = sycl::ext::intel::math;

int main(int, char **) {
  sycl::queue device_queue(sycl::default_selector_v);
  std::cout << "Running on "
            << device_queue.get_device().get_info<sycl::info::device::name>()
            << "\n";

  if (!device_queue.get_device().has(sycl::aspect::fp16)) {
    std::cout << "Test skipped on platform without fp16 support." << std::endl;
    return 0;
  }

  {
    std::initializer_list<sycl::half> input_vals1 = {0.5f, -1.125f, 100.5f,
                                                     0.f,  0.125f,  0.25f};
    std::initializer_list<sycl::half> input_vals2 = {-6.625f, -11.25f, -88.125f,
                                                     0.f,     0.625f,  0.f};
    std::initializer_list<sycl::half> ref_vals = {-6.125f, -12.375f, 12.375f,
                                                  0,       0.75f,    0.25f};
    std::initializer_list<sycl::half> sat_ref_vals = {0.f, 0.f,   1.f,
                                                      0.f, 0.75f, 0.25f};
    std::initializer_list<sycl::half> ref_vals1 = {7.125f, 10.125f, 188.625f,
                                                   0.f,    -0.5f,   0.25f};
    std::initializer_list<sycl::half> sat_ref_vals1 = {1.f, 1.f, 1.f,
                                                       0.f, 0.f, 0.25f};
    std::initializer_list<sycl::half> ref_vals2 = {
        -3.3125f, 12.65625f, -8856.5625f, 0.f, 0.078125f, 0.f};
    std::initializer_list<sycl::half> sat_ref_vals2 = {0.f, 1.f,       0.f,
                                                       0.f, 0.078125f, 0.f};
    std::initializer_list<sycl::half> ref_vals3 = {-0.5f, 1.125f,  -100.5f,
                                                   0.f,   -0.125f, -0.25f};
    test2(device_queue, input_vals1, input_vals2, ref_vals, F2(sycl_imf::hadd));
    std::cout << "hadd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, sat_ref_vals,
          F2(sycl_imf::hadd_sat));
    std::cout << "hadd_sat passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals1,
          F2(sycl_imf::hsub));
    std::cout << "hsub passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, sat_ref_vals1,
          F2(sycl_imf::hsub_sat));
    std::cout << "hsub_sat passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(sycl_imf::hmul));
    std::cout << "hmul passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, sat_ref_vals2,
          F2(sycl_imf::hmul_sat));
    std::cout << "hmul_sat passes." << std::endl;
    test(device_queue, input_vals1, ref_vals3, F(sycl_imf::hneg));
    std::cout << "hneg passes." << std::endl;
  }

  {
    std::initializer_list<sycl::half> input_vals1 = {1.5f, 10.5, 0.75f, 0.75f};
    std::initializer_list<sycl::half> input_vals2 = {2.5f, 2.75, 1.25f, -1.25f};
    std::initializer_list<sycl::half> input_vals3 = {-3.25f, -8.375, -1.5f,
                                                     1.5f};
    std::initializer_list<sycl::half> ref_vals = {0.5f, 20.5f, -0.5625f,
                                                  0.5625f};
    std::initializer_list<sycl::half> sat_ref_vals = {0.5f, 1.f, 0.f, 0.5625f};
    std::initializer_list<sycl::half> relu_ref_vals = {0.5f, 20.5f, -0.f,
                                                       0.5625f};
    test3(device_queue, input_vals1, input_vals2, input_vals3, ref_vals,
          F3(sycl_imf::hfma));
    std::cout << "hfma passes." << std::endl;
    test3(device_queue, input_vals1, input_vals2, input_vals3, sat_ref_vals,
          F3(sycl_imf::hfma_sat));
    std::cout << "hfma_sat passes." << std::endl;
    test3(device_queue, input_vals1, input_vals2, input_vals3, relu_ref_vals,
          F3(sycl_imf::hfma_relu));
    std::cout << "hfma_relu passes." << std::endl;
  }

  {
    std::initializer_list<sycl::half> input_vals1 = {
        0.5f, 99.41f, 1.0f, -0.25f, NAN, NAN, 21.9f, 99.2f};
    std::initializer_list<sycl::half> input_vals2 = {
        0.501f, 99.41f, 1.0f, -0.25f, 2.123f, NAN, -9.1f, 99.23f};
    std::initializer_list<sycl::half> input_vals3 = {12.f, -99.f, INFINITY, 0.f,
                                                     -INFINITY};
    std::initializer_list<bool> ref_vals1 = {false, true,  true,  true,
                                             false, false, false, false};
    std::initializer_list<bool> ref_vals2 = {false, true, true,  true,
                                             true,  true, false, false};
    std::initializer_list<bool> ref_vals3 = {false, false, false, false,
                                             false, false, true,  false};
    std::initializer_list<bool> ref_vals4 = {false, false, false, false,
                                             true,  true,  true,  false};
    std::initializer_list<bool> ref_vals5 = {false, true,  true, true,
                                             false, false, true, false};
    std::initializer_list<bool> ref_vals6 = {false, true, true, true,
                                             true,  true, true, false};
    std::initializer_list<bool> ref_vals7 = {true,  false, false, false,
                                             false, false, false, true};
    std::initializer_list<bool> ref_vals8 = {true,  true,  true,  true,
                                             false, false, false, true};
    std::initializer_list<bool> ref_vals9 = {true, false, false, false,
                                             true, true,  false, true};
    std::initializer_list<bool> ref_vals10 = {true, true, true,  true,
                                              true, true, false, true};
    std::initializer_list<bool> ref_vals11 = {true,  false, false, false,
                                              false, false, true,  true};
    std::initializer_list<bool> ref_vals12 = {true, false, false, false,
                                              true, true,  true,  true};
    std::initializer_list<bool> ref_vals13 = {false, false, false, false,
                                              true,  true,  false, false};
    std::initializer_list<bool> ref_vals14 = {false, false, true, false, true};
    test2(device_queue, input_vals1, input_vals2, ref_vals1, F2(sycl_imf::heq));
    std::cout << "heq passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(sycl_imf::hequ));
    std::cout << "hequ passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals3, F2(sycl_imf::hgt));
    std::cout << "hgt passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals4,
          F2(sycl_imf::hgtu));
    std::cout << "hgtu passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals5, F2(sycl_imf::hge));
    std::cout << "hge passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals6,
          F2(sycl_imf::hgeu));
    std::cout << "hgeu passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals7, F2(sycl_imf::hlt));
    std::cout << "hlt passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals8, F2(sycl_imf::hle));
    std::cout << "hle passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals9,
          F2(sycl_imf::hltu));
    std::cout << "hltu passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals10,
          F2(sycl_imf::hleu));
    std::cout << "hleu passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals11,
          F2(sycl_imf::hne));
    std::cout << "hne passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals12,
          F2(sycl_imf::hneu));
    std::cout << "hneu passes." << std::endl;
    test(device_queue, input_vals1, ref_vals13, F(sycl_imf::hisnan));
    std::cout << "hisnan passes." << std::endl;
    test(device_queue, input_vals3, ref_vals14, F(sycl_imf::hisinf));
    std::cout << "hisinf passes." << std::endl;
  }

  {
    std::initializer_list<sycl::half2> input_vals1 = {
        {1.5f, -0.5f}, {0.5f, -12.5f}, {-0.5f, 0.625f}, {2.5f, -0.75f}};
    std::initializer_list<sycl::half2> input_vals2 = {
        {0.25f, -1.375f}, {0.25f, 10.375f}, {-0.5f, 0.125f}, {3.f, 10.f}};
    std::initializer_list<sycl::half2> input_vals3 = {
        {0.375f, 1.4f}, {-5.125f, 130.25f}, {0.25f, -21.5f}, {-6.625f, 7.875f}};
    std::initializer_list<sycl::half2> ref_vals1 = {
        {1.75f, -1.875f}, {0.75f, -2.125f}, {-1.f, 0.75f}, {5.5f, 9.25f}};
    std::initializer_list<sycl::half2> ref_vals2 = {
        {1.f, 0.f}, {0.75f, 0.f}, {0.f, 0.75f}, {1.f, 1.f}};
    std::initializer_list<sycl::half2> ref_vals3 = {{0.375f, 0.6875f},
                                                    {0.125f, -129.6875f},
                                                    {0.25f, 0.078125f},
                                                    {7.5f, -7.5f}};
    std::initializer_list<sycl::half2> ref_vals4 = {
        {0.375f, 0.6875f}, {0.125f, 0.f}, {0.25f, 0.078125f}, {1.f, 0.f}};
    std::initializer_list<sycl::half2> ref_vals5 = {
        {6.f, 0.3636f}, {2.f, -1.20481f}, {1.f, 5.f}, {0.83333f, -0.075f}};
    std::initializer_list<sycl::half2> ref_vals6 = {
        {1.25f, 0.875f}, {0.25f, -22.875f}, {0.f, 0.5f}, {-0.5f, -10.75f}};
    std::initializer_list<sycl::half2> ref_vals7 = {
        {1.f, 0.875f}, {0.25f, 0.f}, {0.f, 0.5f}, {0.f, 0.f}};
    std::initializer_list<sycl::half2> ref_vals8 = {
        {-1.5f, 0.5f}, {-0.5f, 12.5f}, {0.5f, -0.625f}, {-2.5f, 0.75f}};
    std::initializer_list<sycl::half2> ref_vals9 = {{0.75f, 2.0875f},
                                                    {-5.f, 0.5625f},
                                                    {0.5f, -21.421875f},
                                                    {0.875f, 0.375f}};
    std::initializer_list<sycl::half2> ref_vals10 = {
        {0.75f, 1.f}, {0.f, 0.5625f}, {0.5f, 0.f}, {0.875f, 0.375f}};
    std::initializer_list<sycl::half2> ref_vals11 = {
        {0.75f, 2.0875f}, {0.f, 0.5625f}, {0.5f, 0.f}, {0.875f, 0.375f}};
    test2(device_queue, input_vals1, input_vals2, ref_vals1,
          F2(sycl_imf::hadd2));
    std::cout << "hadd2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(sycl_imf::hadd2_sat));
    std::cout << "hadd2_sat passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals3,
          F2(sycl_imf::hmul2));
    std::cout << "hmul2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals4,
          F2(sycl_imf::hmul2_sat));
    std::cout << "hmul2_sat passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals5,
          F2(sycl_imf::h2div));
    std::cout << "h2div passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals6,
          F2(sycl_imf::hsub2));
    std::cout << "hsub2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals7,
          F2(sycl_imf::hsub2_sat));
    std::cout << "hsub2_sat passes." << std::endl;
    test(device_queue, input_vals1, ref_vals8, F(sycl_imf::hneg2));
    std::cout << "hneg2 passes." << std::endl;
    test3(device_queue, input_vals1, input_vals2, input_vals3, ref_vals9,
          F3(sycl_imf::hfma2));
    std::cout << "hfma2 passes." << std::endl;
    test3(device_queue, input_vals1, input_vals2, input_vals3, ref_vals10,
          F3(sycl_imf::hfma2_sat));
    std::cout << "hfma2_sat passes." << std::endl;
    test3(device_queue, input_vals1, input_vals2, input_vals3, ref_vals11,
          F3(sycl_imf::hfma2_relu));
    std::cout << "hfma2_relu passes." << std::endl;
  }

  {
    std::initializer_list<sycl::half2> input_vals1 = {
        {1.5f, -0.5f}, {1.5f, -0.5f},     {-11.25f, 19.375f}, {120.375f, 10.5f},
        {NAN, 1.f},    {25.25f, -0.375f}, {NAN, 1.0f}};
    std::initializer_list<sycl::half2> input_vals2 = {
        {1.0f, -1.5f}, {1.58f, -0.4f},    {-11.25f, 18.5f}, {-230.5f, 10.5f},
        {1.f, 1.f},    {25.25f, -0.375f}, {NAN, 2.0f}};
    std::initializer_list<bool> ref_vals1 = {false, false, false, false,
                                             false, true,  false};
    std::initializer_list<bool> ref_vals2 = {false, false, false, false,
                                             true,  true,  false};
    std::initializer_list<bool> ref_vals3 = {true,  false, true, true,
                                             false, true,  false};
    std::initializer_list<bool> ref_vals4 = {true, false, true, true,
                                             true, true,  false};
    std::initializer_list<bool> ref_vals5 = {true,  false, false, false,
                                             false, false, false};
    std::initializer_list<bool> ref_vals6 = {true,  false, false, false,
                                             false, false, false};
    std::initializer_list<bool> ref_vals7 = {false, true, false, false,
                                             false, true, false};
    std::initializer_list<bool> ref_vals8 = {false, true, false, false,
                                             true,  true, true};
    std::initializer_list<bool> ref_vals9 = {false, true,  false, false,
                                             false, false, false};
    std::initializer_list<bool> ref_vals10 = {false, true,  false, false,
                                              false, false, true};
    std::initializer_list<bool> ref_vals11 = {true,  true,  false, false,
                                              false, false, false};
    std::initializer_list<bool> ref_vals12 = {true,  true,  false, false,
                                              false, false, true};

    std::initializer_list<sycl::half2> ref_vals13 = {
        {0.f, 0.f}, {0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f},
        {0.f, 1.f}, {1.f, 1.f}, {0.f, 0.f}};
    std::initializer_list<sycl::half2> ref_vals14 = {
        {0.f, 0.f}, {0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f},
        {1.f, 1.f}, {1.f, 1.f}, {1.f, 0.f}};
    std::initializer_list<sycl::half2> ref_vals15 = {
        {1.f, 1.f}, {0.f, 0.f}, {1.f, 1.f}, {1.f, 1.f},
        {0.f, 1.f}, {1.f, 1.f}, {0.f, 0.f}};
    std::initializer_list<sycl::half2> ref_vals16 = {
        {1.f, 1.f}, {0.f, 0.f}, {1.f, 1.f}, {1.f, 1.f},
        {1.f, 1.f}, {1.f, 1.f}, {1.f, 0.f}};
    std::initializer_list<sycl::half2> ref_vals17 = {
        {1.f, 1.f}, {0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f},
        {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
    std::initializer_list<sycl::half2> ref_vals18 = {
        {1.f, 1.f}, {0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f},
        {1.f, 0.f}, {0.f, 0.f}, {1.f, 0.f}};
    std::initializer_list<sycl::half2> ref_vals19 = {
        {0.f, 0.f}, {1.f, 1.f}, {1.f, 0.f}, {0.f, 1.f},
        {0.f, 1.f}, {1.f, 1.f}, {0.f, 1.f}};
    std::initializer_list<sycl::half2> ref_vals20 = {
        {0.f, 0.f}, {1.f, 1.f}, {1.f, 0.f}, {0.f, 1.f},
        {1.f, 1.f}, {1.f, 1.f}, {1.f, 1.f}};
    std::initializer_list<sycl::half2> ref_vals21 = {
        {0.f, 0.f}, {1.f, 1.f}, {0.f, 0.f}, {0.f, 0.f},
        {0.f, 0.f}, {0.f, 0.f}, {0.f, 1.f}};
    std::initializer_list<sycl::half2> ref_vals22 = {
        {0.f, 0.f}, {1.f, 1.f}, {0.f, 0.f}, {0.f, 0.f},
        {1.f, 0.f}, {0.f, 0.f}, {1.f, 1.f}};
    std::initializer_list<sycl::half2> ref_vals23 = {
        {1.f, 1.f}, {1.f, 1.f}, {0.f, 1.f}, {1.f, 0.f},
        {0.f, 0.f}, {0.f, 0.f}, {0.f, 1.f}};
    std::initializer_list<sycl::half2> ref_vals24 = {
        {1.f, 1.f}, {1.f, 1.f}, {0.f, 1.f}, {1.f, 0.f},
        {1.f, 0.f}, {0.f, 0.f}, {1.f, 1.f}};
    std::initializer_list<sycl::half2> ref_vals25 = {
        {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
        {1.f, 0.f}, {0.f, 0.f}, {1.f, 0.f}};

    test2(device_queue, input_vals1, input_vals2, ref_vals1,
          F2(sycl_imf::hbeq2));
    std::cout << "hbeq2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(sycl_imf::hbequ2));
    std::cout << "hbequ2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals3,
          F2(sycl_imf::hbge2));
    std::cout << "hbge2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals4,
          F2(sycl_imf::hbgeu2));
    std::cout << "hbgeu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals5,
          F2(sycl_imf::hbgt2));
    std::cout << "hbgt2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals6,
          F2(sycl_imf::hbgtu2));
    std::cout << "hbgtu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals7,
          F2(sycl_imf::hble2));
    std::cout << "hble2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals8,
          F2(sycl_imf::hbleu2));
    std::cout << "hbleu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals9,
          F2(sycl_imf::hblt2));
    std::cout << "hblt2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals10,
          F2(sycl_imf::hbltu2));
    std::cout << "hbltu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals11,
          F2(sycl_imf::hbne2));
    std::cout << "hbne2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals12,
          F2(sycl_imf::hbneu2));
    std::cout << "hbneu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals13,
          F2(sycl_imf::heq2));
    std::cout << "heq2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals14,
          F2(sycl_imf::hequ2));
    std::cout << "hequ2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals15,
          F2(sycl_imf::hge2));
    std::cout << "hge2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals16,
          F2(sycl_imf::hgeu2));
    std::cout << "hgeu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals17,
          F2(sycl_imf::hgt2));
    std::cout << "hgt2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals18,
          F2(sycl_imf::hgtu2));
    std::cout << "hgtu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals19,
          F2(sycl_imf::hle2));
    std::cout << "hle2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals20,
          F2(sycl_imf::hleu2));
    std::cout << "hleu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals21,
          F2(sycl_imf::hlt2));
    std::cout << "hlt2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals22,
          F2(sycl_imf::hltu2));
    std::cout << "hltu2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals23,
          F2(sycl_imf::hne2));
    std::cout << "hne2 passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals24,
          F2(sycl_imf::hneu2));
    std::cout << "hneu2 passes." << std::endl;
    test(device_queue, input_vals1, ref_vals25, F(sycl_imf::hisnan2));
    std::cout << "hisnan2 passes." << std::endl;
  }

  {
    std::initializer_list<sycl::half> input_vals1 = {1.5f, -0.5f, -100.125f,
                                                     NAN};
    std::initializer_list<sycl::half> input_vals2 = {11.25f, -1.25f, 10.5f,
                                                     0.f};
    std::initializer_list<sycl::half2> input_vals3 = {{-1.5f, 0.25f},
                                                      {2.5f, -100.375f},
                                                      {-1.5f, -22.25f},
                                                      {100.f, 20.5f},
                                                      {0.f, 0.f}};
    std::initializer_list<sycl::half> ref_vals1 = {11.25f, -0.5f, 10.5f, 0.f};
    std::initializer_list<sycl::half> ref_vals2 = {1.5f, -1.25f, -100.125f,
                                                   0.f};
    std::initializer_list<sycl::half> ref_vals3 = {11.25f, 1.25f, 10.5f, 0.f};
    std::initializer_list<sycl::half2> ref_vals4 = {{1.5f, 0.25f},
                                                    {2.5f, 100.375f},
                                                    {1.5f, 22.25f},
                                                    {100.f, 20.5f},
                                                    {0.f, 0.f}};
    test2(device_queue, input_vals1, input_vals2, ref_vals1,
          F2(sycl_imf::hmax));
    std::cout << "hmax passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals2,
          F2(sycl_imf::hmin));
    std::cout << "hmin passes." << std::endl;
    test(device_queue, input_vals2, ref_vals3, F(sycl_imf::habs));
    std::cout << "habs passes." << std::endl;
    test(device_queue, input_vals3, ref_vals4, F(sycl_imf::habs2));
    std::cout << "habs2 passes." << std::endl;
  }

  {
    std::initializer_list<sycl::half2> input_vals1 = {
        {1.0f, -2.0f}, {9.5f, -2.25f}, {10.f, 21.f}, {-12.5f, 2.25f}};
    std::initializer_list<sycl::half2> input_vals2 = {
        {9.375f, 3.0f}, {-1.f, 2.25f}, {2.5f, 1.f}, {0.f, 6.f}};
    std::initializer_list<sycl::half2> input_vals3 = {
        {8.f, 4.0f}, {1.f, -20.f}, {3.5f, -1.f}, {10.f, -2.f}};
    std::initializer_list<sycl::half2> ref_vals = {
        {23.375f, -11.75f}, {-3.4375f, 3.625f}, {7.5f, 61.5f}, {-3.5f, -77.f}};
    test3(device_queue, input_vals1, input_vals2, input_vals3, ref_vals,
          F3(sycl_imf::hcmadd));
    std::cout << "hcmadd passes." << std::endl;
  }

  return 0;
}
