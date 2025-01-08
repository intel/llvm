//===---- math_emu_simd_from_syclomatic.cpp ---------- *- C++ -* ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file is modified from the code migrated by SYCLomatic.

// REQUIRES: aspect-fp16

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <syclcompat/device.hpp>
#include <syclcompat/math.hpp>

using namespace std;

typedef pair<unsigned int, unsigned int> Uint_pair;

void checkResult(const string &FuncName, const vector<unsigned int> &Inputs,
                 const unsigned int &Expect, const unsigned int &DeviceResult) {
  cout << FuncName << "(" << Inputs[0];
  for (size_t i = 1; i < Inputs.size(); ++i) {
    cout << ", " << Inputs[i];
  }
  cout << ") = " << DeviceResult << " (expect " << Expect << ")";
  assert(DeviceResult == Expect);
}

void vabs2(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult =
      syclcompat::vectorized_unary<sycl::short2>(Input1, syclcompat::abs());
}

void testVabs2Cases(const vector<pair<unsigned int, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_ct1 = TestCase.first;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vabs2(DeviceResult, TestCase_first_ct1);
          });
    });
    q_ct1.wait();
    checkResult("__vabs2", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

void vabs4(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult =
      syclcompat::vectorized_unary<sycl::char4>(Input1, syclcompat::abs());
}

void testVabs4Cases(const vector<pair<unsigned int, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_ct1 = TestCase.first;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vabs4(DeviceResult, TestCase_first_ct1);
          });
    });
    q_ct1.wait();
    checkResult("__vabs4", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

void vabsdiffs2(unsigned int *const DeviceResult, unsigned int Input1,
                unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, syclcompat::abs_diff());
}

void testVabsdiffs2Cases(
    const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vabsdiffs2(DeviceResult, TestCase_first_first_ct1,
                       TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vabsdiffs2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vabsdiffs4(unsigned int *const DeviceResult, unsigned int Input1,
                unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, syclcompat::abs_diff());
}

void testVabsdiffs4Cases(
    const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vabsdiffs4(DeviceResult, TestCase_first_first_ct1,
                       TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vabsdiffs4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vabsdiffu2(unsigned int *const DeviceResult, unsigned int Input1,
                unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, syclcompat::abs_diff());
}

void testVabsdiffu2Cases(
    const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vabsdiffu2(DeviceResult, TestCase_first_first_ct1,
                       TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vabsdiffu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vabsdiffu4(unsigned int *const DeviceResult, unsigned int Input1,
                unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, syclcompat::abs_diff());
}

void testVabsdiffu4Cases(
    const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vabsdiffu4(DeviceResult, TestCase_first_first_ct1,
                       TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vabsdiffu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vabsss2(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, 0, syclcompat::abs_diff());
}

void testVabsss2Cases(
    const vector<pair<unsigned int, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_ct1 = TestCase.first;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vabsss2(DeviceResult, TestCase_first_ct1);
          });
    });
    q_ct1.wait();
    checkResult("__vabsss2", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

void vabsss4(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, 0, syclcompat::abs_diff());
}

void testVabsss4Cases(
    const vector<pair<unsigned int, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_ct1 = TestCase.first;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vabsss4(DeviceResult, TestCase_first_ct1);
          });
    });
    q_ct1.wait();
    checkResult("__vabsss4", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

void vadd2(unsigned int *const DeviceResult, unsigned int Input1,
           unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(Input1, Input2,
                                                               std::plus<>());
}

void testVadd2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vadd2(DeviceResult, TestCase_first_first_ct1,
                  TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vadd2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vadd4(unsigned int *const DeviceResult, unsigned int Input1,
           unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(Input1, Input2,
                                                              std::plus<>());
}

void testVadd4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vadd4(DeviceResult, TestCase_first_first_ct1,
                  TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vadd4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vaddss2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, syclcompat::add_sat());
}

void testVaddss2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vaddss2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vaddss2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vaddss4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, syclcompat::add_sat());
}

void testVaddss4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vaddss4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vaddss4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vaddus2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, syclcompat::add_sat());
}

void testVaddus2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vaddus2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vaddus2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vaddus4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, syclcompat::add_sat());
}

void testVaddus4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vaddus4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vaddus4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vavgs2(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, syclcompat::rhadd());
}

void testVavgs2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vavgs2(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vavgs2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vavgs4(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, syclcompat::rhadd());
}

void testVavgs4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vavgs4(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vavgs4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vavgu2(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, syclcompat::rhadd());
}

void testVavgu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vavgu2(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vavgu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vavgu4(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, syclcompat::rhadd());
}

void testVavgu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vavgu4(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vavgu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpeq2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::equal_to<>());
}

void testVcmpeq2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpeq2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpeq2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpeq4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::equal_to<>());
}

void testVcmpeq4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpeq4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpeq4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpges2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, std::greater_equal<>());
}

void testVcmpges2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpges2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpges2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpges4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, std::greater_equal<>());
}

void testVcmpges4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpges4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpges4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpgeu2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::greater_equal<>());
}

void testVcmpgeu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpgeu2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpgeu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpgeu4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::greater_equal<>());
}

void testVcmpgeu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpgeu4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpgeu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpgts2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(Input1, Input2,
                                                              std::greater<>());
}

void testVcmpgts2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpgts2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpgts2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpgts4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(Input1, Input2,
                                                             std::greater<>());
}

void testVcmpgts4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpgts4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpgts4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpgtu2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::greater<>());
}

void testVcmpgtu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpgtu2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpgtu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpgtu4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(Input1, Input2,
                                                              std::greater<>());
}

void testVcmpgtu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpgtu4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpgtu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmples2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, std::less_equal<>());
}

void testVcmples2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmples2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmples2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmples4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, std::less_equal<>());
}

void testVcmples4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmples4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmples4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpleu2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::less_equal<>());
}

void testVcmpleu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpleu2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpleu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpleu4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::less_equal<>());
}

void testVcmpleu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpleu4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpleu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmplts2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(Input1, Input2,
                                                              std::less<>());
}

void testVcmplts2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmplts2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmplts2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmplts4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult =
      syclcompat::vectorized_binary<sycl::char4>(Input1, Input2, std::less<>());
}

void testVcmplts4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmplts4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmplts4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpltu2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(Input1, Input2,
                                                               std::less<>());
}

void testVcmpltu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpltu2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpltu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpltu4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(Input1, Input2,
                                                              std::less<>());
}

void testVcmpltu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpltu4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpltu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpne2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::not_equal_to<>());
}

void testVcmpne2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpne2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpne2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vcmpne4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::not_equal_to<>());
}

void testVcmpne4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vcmpne4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vcmpne4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vhaddu2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, syclcompat::hadd());
}

void testVhaddu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vhaddu2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vhaddu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vhaddu4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, syclcompat::hadd());
}

void testVhaddu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vhaddu4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vhaddu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vmaxs2(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, syclcompat::maximum());
}

void testVmaxs2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vmaxs2(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vmaxs2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vmaxs4(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, syclcompat::maximum());
}

void testVmaxs4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vmaxs4(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vmaxs4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vmaxu2(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, syclcompat::maximum());
}

void testVmaxu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vmaxu2(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vmaxu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vmaxu4(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, syclcompat::maximum());
}

void testVmaxu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vmaxu4(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vmaxu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vmins2(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, syclcompat::minimum());
}

void testVmins2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vmins2(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vmins2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vmins4(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, syclcompat::minimum());
}

void testVmins4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vmins4(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vmins4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vminu2(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, syclcompat::minimum());
}

void testVminu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vminu2(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vminu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vminu4(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, syclcompat::minimum());
}

void testVminu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vminu4(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vminu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vneg2(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult =
      syclcompat::vectorized_unary<sycl::short2>(Input1, std::negate<>());
}

void testVneg2Cases(const vector<pair<unsigned int, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_ct1 = TestCase.first;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vneg2(DeviceResult, TestCase_first_ct1);
          });
    });
    q_ct1.wait();
    checkResult("__vneg2", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

void vneg4(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult =
      syclcompat::vectorized_unary<sycl::char4>(Input1, std::negate<>());
}

void testVneg4Cases(const vector<pair<unsigned int, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_ct1 = TestCase.first;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vneg4(DeviceResult, TestCase_first_ct1);
          });
    });
    q_ct1.wait();
    checkResult("__vneg4", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

void vnegss2(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      0, Input1, syclcompat::sub_sat());
}

void testVnegss2Cases(
    const vector<pair<unsigned int, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_ct1 = TestCase.first;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vnegss2(DeviceResult, TestCase_first_ct1);
          });
    });
    q_ct1.wait();
    checkResult("__vnegss2", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

void vnegss4(unsigned int *const DeviceResult, unsigned int Input1) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      0, Input1, syclcompat::sub_sat());
}

void testVnegss4Cases(
    const vector<pair<unsigned int, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_ct1 = TestCase.first;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vnegss4(DeviceResult, TestCase_first_ct1);
          });
    });
    q_ct1.wait();
    checkResult("__vnegss4", {TestCase.first}, TestCase.second, *DeviceResult);
  }
}

void vsads2(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult =
      syclcompat::vectorized_sum_abs_diff<sycl::short2>(Input1, Input2);
}

void testVsads2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsads2(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsads2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsads4(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult =
      syclcompat::vectorized_sum_abs_diff<sycl::char4>(Input1, Input2);
}

void testVsads4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsads4(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsads4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsadu2(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult =
      syclcompat::vectorized_sum_abs_diff<sycl::ushort2>(Input1, Input2);
}

void testVsadu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsadu2(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsadu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsadu4(unsigned int *const DeviceResult, unsigned int Input1,
            unsigned int Input2) {
  *DeviceResult =
      syclcompat::vectorized_sum_abs_diff<sycl::uchar4>(Input1, Input2);
}

void testVsadu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsadu4(DeviceResult, TestCase_first_first_ct1,
                   TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsadu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vseteq2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::equal_to<unsigned short>());
}

void testVseteq2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vseteq2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vseteq2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vseteq4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::equal_to<unsigned char>());
}

void testVseteq4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vseteq4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vseteq4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetges2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, std::greater_equal<short>());
}

void testVsetges2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetges2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetges2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetges4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, std::greater_equal<char>());
}

void testVsetges4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetges4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetges4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetgeu2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::greater_equal<unsigned short>());
}

void testVsetgeu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetgeu2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetgeu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetgeu4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::greater_equal<unsigned char>());
}

void testVsetgeu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetgeu4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetgeu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetgts2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, std::greater<short>());
}

void testVsetgts2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetgts2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetgts2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetgts4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, std::greater<char>());
}

void testVsetgts4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetgts4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetgts4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetgtu2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::greater<unsigned short>());
}

void testVsetgtu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetgtu2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetgtu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetgtu4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::greater<unsigned char>());
}

void testVsetgtu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetgtu4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetgtu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetles2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, std::less_equal<short>());
}

void testVsetles2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetles2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetles2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetles4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, std::less_equal<char>());
}

void testVsetles4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetles4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetles4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetleu2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::less_equal<unsigned short>());
}

void testVsetleu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetleu2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetleu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetleu4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::less_equal<unsigned char>());
}

void testVsetleu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetleu4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetleu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetlts2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, std::less<short>());
}

void testVsetlts2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetlts2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetlts2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetlts4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(Input1, Input2,
                                                             std::less<char>());
}

void testVsetlts4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetlts4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetlts4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetltu2(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::less<unsigned short>());
}

void testVsetltu2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetltu2(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetltu2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetltu4(unsigned int *const DeviceResult, unsigned int Input1,
              unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::less<unsigned char>());
}

void testVsetltu4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetltu4(DeviceResult, TestCase_first_first_ct1,
                     TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetltu4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetne2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, std::not_equal_to<unsigned short>());
}

void testVsetne2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetne2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetne2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsetne4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, std::not_equal_to<unsigned char>());
}

void testVsetne4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsetne4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsetne4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsub2(unsigned int *const DeviceResult, unsigned int Input1,
           unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(Input1, Input2,
                                                               std::minus<>());
}

void testVsub2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsub2(DeviceResult, TestCase_first_first_ct1,
                  TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsub2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsub4(unsigned int *const DeviceResult, unsigned int Input1,
           unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(Input1, Input2,
                                                              std::minus<>());
}

void testVsub4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsub4(DeviceResult, TestCase_first_first_ct1,
                  TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsub4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsubss2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::short2>(
      Input1, Input2, syclcompat::sub_sat());
}

void testVsubss2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsubss2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsubss2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsubss4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::char4>(
      Input1, Input2, syclcompat::sub_sat());
}

void testVsubss4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsubss4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsubss4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsubus2(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::ushort2>(
      Input1, Input2, syclcompat::sub_sat());
}

void testVsubus2Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsubus2(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsubus2", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

void vsubus4(unsigned int *const DeviceResult, unsigned int Input1,
             unsigned int Input2) {
  *DeviceResult = syclcompat::vectorized_binary<sycl::uchar4>(
      Input1, Input2, syclcompat::sub_sat());
}

void testVsubus4Cases(const vector<pair<Uint_pair, unsigned int>> &TestCases) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  unsigned int *DeviceResult;
  DeviceResult =
      (unsigned int *)sycl::malloc_shared(sizeof(*DeviceResult), q_ct1);
  for (const auto &TestCase : TestCases) {
    q_ct1.submit([&](sycl::handler &cgh) {
      auto TestCase_first_first_ct1 = TestCase.first.first;
      auto TestCase_first_second_ct2 = TestCase.first.second;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            vsubus4(DeviceResult, TestCase_first_first_ct1,
                    TestCase_first_second_ct2);
          });
    });
    q_ct1.wait();
    checkResult("__vsubus4", {TestCase.first.first, TestCase.first.second},
                TestCase.second, *DeviceResult);
  }
}

int main() {
  testVabs2Cases({
      {214321, 214321},
      {3, 3},
      {2147483647, 2147418113}, // 7FFF,FFFF-->7FFF,0001
      {0, 0},
      {4294967295, 65537}, // FFFF,FFFF-->0001,0001
  });
  testVabs4Cases({
      {214321, 214321},
      {3, 3},
      {2147483647, 2130772225}, // 7F,FF,FF,FF-->7F,01,01,01
      {0, 0},
      {4294967295, 16843009}, // FF,FF,FF,FF-->01,01,01,01
  });
  testVabsdiffs2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147239218},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVabsdiffs4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2130986546},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVabsdiffu2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147269326},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVabsdiffu4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147269326},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVabsss2Cases({
      {214321, 214321},
      {3, 3},
      {2147483647, 2147418113},
      {0, 0},
      {4294967295, 65537},
  });
  testVabsss4Cases({
      {214321, 214321},
      {3, 3},
      {2147483647, 2130772225},
      {0, 0},
      {4294967295, 16843009},
  });
  testVadd2Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2147632432},
      {{4294967295, 2147483647}, 2147418110},
      {{4294967295, 4294967295}, 4294901758},
      {{3, 4}, 7},
  });
  testVadd4Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2130854960},
      {{4294967295, 2147483647}, 2130640638},
      {{4294967295, 4294967295}, 4278124286},
      {{3, 4}, 7},
  });
  testVaddss2Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2147435824}, // 3,4531+7FFF,FFFF-->7FFF,4530
      {{4294967295, 2147483647}, 2147418110},
      {{4294967295, 4294967295}, 4294901758},
      {{3, 4}, 7},
  });
  testVaddss4Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2130854960},
      {{4294967295, 2147483647}, 2130640638},
      {{4294967295, 4294967295}, 4278124286},
      {{3, 4}, 7},
  });
  testVaddus2Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2147680255},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 7},
  });
  testVaddus4Cases({
      {{4, 3}, 7},
      {{214321, 2147483647}, 2147483647},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 7},
  });
  testVavgs2Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 1073816216},
      {{4294967295, 2147483647}, 1073741823},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVavgs4Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 1073816088},
      {{4294967295, 2147483647}, 1073741823},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVavgu2Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 1073848984},
      {{4294967295, 2147483647}, 3221225471},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVavgu4Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 1082237592},
      {{4294967295, 2147483647}, 3221225471},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVcmpeq2Cases({
      {{4, 3}, 4294901760},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 65535},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294901760},
  });
  testVcmpeq4Cases({
      {{4, 3}, 4294967040},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 16777215},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967040},
  });
  testVcmpges2Cases({
      {{4, 3}, 4294967295},
      {{214321, 2147483647}, 65535},
      {{4294967295, 2147483647}, 65535},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294901760},
  });
  testVcmpges4Cases({
      {{4, 3}, 4294967295},
      {{214321, 2147483647}, 16777215},
      {{4294967295, 2147483647}, 16777215},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967040},
  });
  testVcmpgeu2Cases({
      {{4, 3}, 4294967295},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294901760},
  });
  testVcmpgeu4Cases({
      {{4, 3}, 4294967295},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967040},
  });
  testVcmpgts2Cases({
      {{4, 3}, 65535},
      {{214321, 2147483647}, 65535},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVcmpgts4Cases({
      {{4, 3}, 255},
      {{214321, 2147483647}, 16777215},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVcmpgtu2Cases({
      {{4, 3}, 65535},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 4294901760},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVcmpgtu4Cases({
      {{4, 3}, 255},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 4278190080},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVcmples2Cases({
      {{4, 3}, 4294901760},
      {{214321, 2147483647}, 4294901760},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967295},
  });
  testVcmples4Cases({
      {{4, 3}, 4294967040},
      {{214321, 2147483647}, 4278190080},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967295},
  });
  testVcmpleu2Cases({
      {{4, 3}, 4294901760},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 65535},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967295},
  });
  testVcmpleu4Cases({
      {{4, 3}, 4294967040},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 16777215},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4294967295},
  });
  testVcmplts2Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 4294901760},
      {{4294967295, 2147483647}, 4294901760},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVcmplts4Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 4278190080},
      {{4294967295, 2147483647}, 4278190080},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVcmpltu2Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVcmpltu4Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVcmpne2Cases({
      {{4, 3}, 65535},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 4294901760},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVcmpne4Cases({
      {{4, 3}, 255},
      {{214321, 2147483647}, 4294967295},
      {{4294967295, 2147483647}, 4278190080},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVhaddu2Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 1073848984},
      {{4294967295, 2147483647}, 3221225471},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVhaddu4Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 1065460376},
      {{4294967295, 2147483647}, 3221225471},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVmaxs2Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2147435825},
      {{4294967295, 2147483647}, 2147483647},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVmaxs4Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2130920753},
      {{4294967295, 2147483647}, 2147483647},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVmaxu2Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2147483647},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVmaxu4Cases({
      {{4, 3}, 4},
      {{214321, 2147483647}, 2147483647},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 4},
  });
  testVmins2Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 262143},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVmins4Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 16777215},
      {{4294967295, 2147483647}, 4294967295},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVminu2Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 214321},
      {{4294967295, 2147483647}, 2147483647},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVminu4Cases({
      {{4, 3}, 3},
      {{214321, 2147483647}, 214321},
      {{4294967295, 2147483647}, 2147483647},
      {{4294967295, 4294967295}, 4294967295},
      {{3, 4}, 3},
  });
  testVneg2Cases({
      {214321, 4294818511},
      {3, 65533},
      {2147483647, 2147549185},
      {0, 0},
      {4294967295, 65537},
  });
  testVneg4Cases({
      {214321, 16628687},
      {3, 253},
      {2147483647, 2164326657},
      {0, 0},
      {4294967295, 16843009},
  });
  testVnegss2Cases({
      {214321, 4294818511},
      {3, 65533},
      {2147483647, 2147549185},
      {0, 0},
      {4294967295, 65537},
  });
  testVnegss4Cases({
      {214321, 16628687},
      {3, 253},
      {2147483647, 2164326657},
      {0, 0},
      {4294967295, 16843009},
  });
  testVsads2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 50478},
      {{4294967295, 2147483647}, 32768},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsads4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 251},
      {{4294967295, 2147483647}, 128},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsadu2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 80586},
      {{4294967295, 2147483647}, 32768},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsadu4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 771},
      {{4294967295, 2147483647}, 128},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVseteq2Cases({
      {{4, 3}, 65536},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 1},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65536},
  });
  testVseteq4Cases({
      {{4, 3}, 16843008},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 65793},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843008},
  });
  testVsetges2Cases({
      {{4, 3}, 65537},
      {{214321, 2147483647}, 1},
      {{4294967295, 2147483647}, 1},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65536},
  });
  testVsetges4Cases({
      {{4, 3}, 16843009},
      {{214321, 2147483647}, 65793},
      {{4294967295, 2147483647}, 65793},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843008},
  });
  testVsetgeu2Cases({
      {{4, 3}, 65537},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 65537},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65536},
  });
  testVsetgeu4Cases({
      {{4, 3}, 16843009},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 16843009},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843008},
  });
  testVsetgts2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 1},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsetgts4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 65793},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsetgtu2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 65536},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsetgtu4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 16777216},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsetles2Cases({
      {{4, 3}, 65536},
      {{214321, 2147483647}, 65536},
      {{4294967295, 2147483647}, 65537},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65537},
  });
  testVsetles4Cases({
      {{4, 3}, 16843008},
      {{214321, 2147483647}, 16777216},
      {{4294967295, 2147483647}, 16843009},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843009},
  });
  testVsetleu2Cases({
      {{4, 3}, 65536},
      {{214321, 2147483647}, 65537},
      {{4294967295, 2147483647}, 1},
      {{4294967295, 4294967295}, 65537},
      {{3, 4}, 65537},
  });
  testVsetleu4Cases({
      {{4, 3}, 16843008},
      {{214321, 2147483647}, 16843009},
      {{4294967295, 2147483647}, 65793},
      {{4294967295, 4294967295}, 16843009},
      {{3, 4}, 16843009},
  });
  testVsetlts2Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 65536},
      {{4294967295, 2147483647}, 65536},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetlts4Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 16777216},
      {{4294967295, 2147483647}, 16777216},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetltu2Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 65537},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetltu4Cases({
      {{4, 3}, 0},
      {{214321, 2147483647}, 16843009},
      {{4294967295, 2147483647}, 0},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetne2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 65537},
      {{4294967295, 2147483647}, 65536},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsetne4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 16843009},
      {{4294967295, 2147483647}, 16777216},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 1},
  });
  testVsub2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147763506},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVsub4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2164540978},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVsubss2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2147763506},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 65535},
  });
  testVsubss4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 2164540978},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 255},
  });
  testVsubus2Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  testVsubus4Cases({
      {{4, 3}, 1},
      {{214321, 2147483647}, 0},
      {{4294967295, 2147483647}, 2147483648},
      {{4294967295, 4294967295}, 0},
      {{3, 4}, 0},
  });
  return 0;
}
