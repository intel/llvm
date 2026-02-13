//==------- saturation_smoke.cpp  - DPC++ ESIMD on-device test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out
//
// The test checks main functionality of esimd::saturate function.

#include "../esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <class T> struct char_to_int {
  using type = typename std::conditional<
      sizeof(T) == 1,
      typename std::conditional<std::is_signed<T>::value, int, unsigned>::type,
      T>::type;
};

template <class T> bool verify(T *data_arr, T *gold_arr, int N) {
  int err_cnt = 0;

  for (unsigned i = 0; i < N; ++i) {
    T val = data_arr[i];
    T gold = gold_arr[i];

    if (val != gold) {
      if (++err_cnt < 10) {
        using T1 = typename char_to_int<T>::type;
        std::cout << "  failed at index " << i << ": " << (T1)val
                  << " != " << (T1)gold << " (gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: " << ((float)(N - err_cnt) / (float)N) * 100.0f
              << "% (" << (N - err_cnt) << "/" << N << ")\n";
  }
  return err_cnt == 0;
}

template <class From, class To, int Nx> struct DataMgr {
  From *src;
  To *dst;
  To *gold;
  static inline constexpr int N = Nx;

  DataMgr(From (&&src_data)[N], To (&&gold_data)[N]) {
    src = new From[N];
    dst = new To[N];
    gold = new To[N];

    for (int i = 0; i < N; i++) {
      src[i] = src_data[i];
      dst[i] = (To)2; // 0, 1 can be results of saturation, so use 2
      gold[i] = gold_data[i];
    }
  }

  ~DataMgr() {
    delete[] src;
    delete[] dst;
    delete[] gold;
  }
};

template <class From, class To, template <class, class> class Mgr>
bool test(queue q) {
  std::cout << "Testing " << typeid(From).name() << " -> " << typeid(To).name()
            << "\n";

  Mgr<From, To> dm;
  constexpr int N = Mgr<From, To>::N;

  try {
    sycl::buffer<From, 1> src_buf(dm.src, N);
    sycl::buffer<To, 1> dst_buf(dm.dst, N);

    auto e = q.submit([&](handler &cgh) {
      auto src_acc = src_buf.template get_access<access::mode::read>(cgh);
      auto dst_acc = dst_buf.template get_access<access::mode::write>(cgh);

      cgh.single_task([=]() SYCL_ESIMD_KERNEL {
        simd<From, N> x(src_acc, 0);
        simd<To, N> y = saturate<To>(x);
        y.copy_to(dst_acc, 0);
      });
    });
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false; // not success
  }
  return verify<To>(dm.dst, dm.gold, N);
}

// clang-format off
template <class From, class To> struct FpToInt : public DataMgr<From, To, 2> {
  static_assert(
    (std::is_floating_point_v<From> || std::is_same_v<From, half>) &&
     std::is_integral_v<To>);
  static inline constexpr int N = 2;

  FpToInt() : DataMgr<From, To, N>(
    // need this trick with -127 + 130 because INT_MAX is not accurately
    // representable with float, and compiler warns:
    //   implicit conversion from 'int' to 'const float' changes value from
    //   2147483647 to 2147483648
    // INT_MAX-127 is accurately representable with float. Use +130 to exceed
    // representable range to actually test saturation.
    // Test data:
    { (From)std::numeric_limits<To>::min() - 10,
      (From)(std::numeric_limits<To>::max()-127) + 130 },
    // Gold data (saturated test data):
    { std::numeric_limits<To>::min(),
      std::numeric_limits<To>::max() })
  {}
};

template <class From, class To>
struct UIntToSameOrNarrowAnyInt : public DataMgr<From, To, 1> {
  static_assert(std::is_integral_v<To> && std::is_integral_v<From> &&
                !std::is_signed_v<From> && (sizeof(From) >= sizeof(To)));
  static inline constexpr int N = 1;

  UIntToSameOrNarrowAnyInt() : DataMgr<From, To, N>(
    { (From)((From)std::numeric_limits<To>::max() + (From)10) },
    { (To)std::numeric_limits<To>::max() })
  {}
};

template <class From, class To>
struct IntToWiderUInt : public DataMgr<From, To, 1> {
  static_assert(std::is_signed_v<From> && !std::is_signed_v<To> &&
                (sizeof(From) < sizeof(To)));
  static inline constexpr int N = 1;

  IntToWiderUInt() : DataMgr<From, To, N>(
    { (From)-1 },
    { (To)0 })
  {}
};

template <class From, class To>
struct SIntToNarrowAnyInt : public DataMgr<From, To, 2> {
  static_assert(std::is_integral_v<From> && std::is_signed_v<From> &&
                std::is_integral_v<To> && (sizeof(From) > sizeof(To)));
  static inline constexpr int N = 2;

  SIntToNarrowAnyInt() : DataMgr<From, To, N>(
    { (From)std::numeric_limits<To>::max() + 10,
      (From)std::numeric_limits<To>::min() - 10 },
    { (To)std::numeric_limits<To>::max(),
      (To)std::numeric_limits<To>::min() })
  {}
};

template <class From, class To> struct FpToFp : public DataMgr<From, To, 5> {
  static_assert((std::is_floating_point_v<To> || std::is_same_v<To, half>));
  static inline constexpr int N = 5;

  FpToFp() : DataMgr<From, To, N>(
    { (From)-10, (From)0, (From)0.5,       (From)1, (From)10 },
    { (To)0,     (To)0,   (To)((From)0.5), (To)1,   (To)1 })
  {}
};

// clang-format on

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";
  const bool doublesSupported = dev.has(sycl::aspect::fp64);
  const bool halfsSupported = dev.has(sycl::aspect::fp16);

  bool passed = true;
  if (halfsSupported)
    passed &= test<half, int, FpToInt>(q);
  if (halfsSupported)
    passed &= test<half, unsigned char, FpToInt>(q);
  passed &= test<float, int, FpToInt>(q);
  if (doublesSupported)
    passed &= test<double, short, FpToInt>(q);

  passed &= test<unsigned char, char, UIntToSameOrNarrowAnyInt>(q);
  passed &= test<unsigned short, short, UIntToSameOrNarrowAnyInt>(q);
  passed &= test<unsigned int, int, UIntToSameOrNarrowAnyInt>(q);
  passed &= test<unsigned int, char, UIntToSameOrNarrowAnyInt>(q);
  passed &= test<unsigned short, unsigned char, UIntToSameOrNarrowAnyInt>(q);

  passed &= test<char, unsigned int, IntToWiderUInt>(q);
  passed &= test<char, unsigned short, IntToWiderUInt>(q);
  passed &= test<short, unsigned int, IntToWiderUInt>(q);

  passed &= test<short, char, SIntToNarrowAnyInt>(q);
  passed &= test<int, unsigned char, SIntToNarrowAnyInt>(q);

  passed &= test<float, float, FpToFp>(q);
  if (halfsSupported)
    passed &= test<half, half, FpToFp>(q);
  if (doublesSupported)
    passed &= test<double, double, FpToFp>(q);

  std::cout << (passed ? "Test passed\n" : "Test FAILED\n");
  return passed ? 0 : 1;
}
