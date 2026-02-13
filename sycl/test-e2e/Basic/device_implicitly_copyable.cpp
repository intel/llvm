// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==--device_implicitly_copyable.cpp - SYCL implicit device copyable test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/sycl_span.hpp>

struct ACopyable {
  int i;
  ACopyable() = default;
  ACopyable(int _i) : i(_i) {}
  ACopyable(const ACopyable &x) : i(x.i) {}
};

template <> struct sycl::is_device_copyable<ACopyable> : std::true_type {};

template <typename DataT, size_t ArrSize>
void CaptureAndCopy(const DataT *data_arr, const DataT &data_scalar,
                    DataT *result_arr, DataT *result_scalar, sycl::queue &q) {
  // We need to copy data_arr, otherwise when using a device it tries to use the
  // host memory.
  DataT cpy_data_arr[ArrSize];
  std::memcpy(cpy_data_arr, data_arr, sizeof(cpy_data_arr));
  sycl::buffer<DataT, 1> buf_arr{result_arr, sycl::range<1>(ArrSize)};
  sycl::buffer<DataT, 1> buf_scalar{result_scalar, sycl::range<1>(1)};

  q.submit([&](sycl::handler &cgh) {
    auto acc_arr = sycl::accessor{buf_arr, cgh, sycl::read_write};
    auto acc_scalar = sycl::accessor{buf_scalar, cgh, sycl::read_write};
    cgh.single_task([=]() {
      for (auto i = 0; i < ArrSize; i++) {
        acc_arr[i] = cpy_data_arr[i];
      }
      acc_scalar[0] = data_scalar;
    });
  });
}

int main() {
  constexpr size_t arr_size = 5;
  constexpr int ref_val = 14;
  sycl::queue q;
  {
    using DataT = std::pair<int, float>;
    DataT pair_arr[arr_size];
    DataT pair{ref_val, ref_val};
    DataT result_pair_arr[arr_size];
    DataT result_pair;

    for (auto i = 0; i < arr_size; i++) {
      pair_arr[i].first = i;
      pair_arr[i].second = i;
    }

    CaptureAndCopy<DataT, arr_size>(pair_arr, pair, result_pair_arr,
                                    &result_pair, q);

    for (auto i = 0; i < arr_size; i++) {
      assert(result_pair_arr[i].first == i);
      assert(result_pair_arr[i].second == i);
    }
    assert(result_pair.first == ref_val && result_pair.second == ref_val);
  }

  {
    using DataT = std::pair<ACopyable, float>;
    DataT pair_arr[arr_size];
    DataT pair{ACopyable(ref_val), ref_val};
    DataT result_pair_arr[arr_size];
    DataT result_pair;

    for (auto i = 0; i < arr_size; i++) {
      pair_arr[i].first = ACopyable(i);
      pair_arr[i].second = i;
    }

    CaptureAndCopy<DataT, arr_size>(pair_arr, pair, result_pair_arr,
                                    &result_pair, q);

    for (auto i = 0; i < arr_size; i++) {
      assert(result_pair_arr[i].first.i == i);
      assert(result_pair_arr[i].second == i);
    }
    assert(result_pair.first.i == ref_val && result_pair.second == ref_val);
  }

  {
    using DataT = std::tuple<int, float, bool>;
    DataT tuple_arr[arr_size];
    DataT tuple{ref_val, ref_val, true};
    DataT result_tuple_arr[arr_size];
    DataT result_tuple;

    for (auto i = 0; i < arr_size; i++) {
      auto &t = tuple_arr[i];
      std::get<0>(t) = i;
      std::get<1>(t) = i;
      std::get<2>(t) = true;
    }

    CaptureAndCopy<DataT, arr_size>(tuple_arr, tuple, result_tuple_arr,
                                    &result_tuple, q);

    for (auto i = 0; i < arr_size; i++) {
      auto t = result_tuple_arr[i];
      assert(std::get<0>(t) == i);
      assert(std::get<1>(t) == i);
      assert(std::get<2>(t) == true);
    }
    assert(std::get<0>(result_tuple) == ref_val);
    assert(std::get<1>(result_tuple) == ref_val);
    assert(std::get<2>(result_tuple) == true);
  }

  {
    using DataT = std::tuple<ACopyable, float, bool>;
    DataT tuple_arr[arr_size];
    DataT tuple{ACopyable(ref_val), ref_val, true};
    DataT result_tuple_arr[arr_size];
    DataT result_tuple;

    for (auto i = 0; i < arr_size; i++) {
      auto &t = tuple_arr[i];
      std::get<0>(t) = ACopyable(i);
      std::get<1>(t) = i;
      std::get<2>(t) = true;
    }

    CaptureAndCopy<DataT, arr_size>(tuple_arr, tuple, result_tuple_arr,
                                    &result_tuple, q);

    for (auto i = 0; i < arr_size; i++) {
      auto t = result_tuple_arr[i];
      assert(std::get<0>(t).i == i);
      assert(std::get<1>(t) == i);
      assert(std::get<2>(t) == true);
    }
    assert(std::get<0>(result_tuple).i == ref_val);
    assert(std::get<1>(result_tuple) == ref_val);
    assert(std::get<2>(result_tuple) == true);
  }

  {
    using DataT = std::variant<int, float, bool>;
    DataT variant_arr[arr_size];
    DataT variant{14};
    DataT result_variant_arr[arr_size];
    DataT result_variant;

    constexpr int variant_size = 3;
    for (auto i = 0; i < arr_size; i++) {
      auto &v = variant_arr[i];
      auto index = i % variant_size;
      if (index == 0) {
        v = i;
      } else if (index == 1) {
        v = (float)i;
      } else {
        v = true;
      }
    }

    CaptureAndCopy<DataT, arr_size>(variant_arr, variant, result_variant_arr,
                                    &result_variant, q);

    for (auto i = 0; i < arr_size; i++) {
      auto v = result_variant_arr[i];
      auto index = i % variant_size;
      if (index == 0) {
        assert(std::get<0>(v) == i);
      } else if (index == 1) {
        assert(std::get<1>(v) == i);
      } else {
        assert(std::get<2>(v) == true);
      }
    }
    assert(std::get<0>(result_variant) == ref_val);
  }

  {
    using DataT = std::variant<ACopyable>;
    DataT variant_arr[arr_size];
    DataT variant;
    DataT result_variant_arr[arr_size];
    DataT result_variant;
    q.submit([&](sycl::handler &cgh) {
       cgh.single_task([=]() {
         // Some implementations of std::variant with complex types relies on
         // virtual functions, so they cannot be used within sycl kernels
         auto size = sizeof(variant_arr[0]);
         size = sizeof(variant);
       });
     }).wait_and_throw();
  }

  {
    using DataT = std::array<int, arr_size>;
    DataT arr_arr[arr_size];
    DataT arr;
    DataT result_arr_arr[arr_size];
    DataT result_arr;

    for (auto i = 0; i < arr_size; i++) {
      auto &a = arr_arr[i];
      for (auto j = 0; j < arr_size; j++) {
        a[j] = j;
      }
      arr[i] = i;
    }

    CaptureAndCopy<DataT, arr_size>(arr_arr, arr, result_arr_arr, &result_arr,
                                    q);

    for (auto i = 0; i < arr_size; i++) {
      auto a = result_arr_arr[i];
      for (auto j = 0; j < arr_size; j++) {
        assert(a[j] == j);
      }
      assert(result_arr[i] == i);
    }
  }

  {
    using DataT = std::array<ACopyable, arr_size>;
    DataT arr_arr[arr_size];
    DataT arr;
    DataT result_arr_arr[arr_size];
    DataT result_arr;

    for (auto i = 0; i < arr_size; i++) {
      auto &a = arr_arr[i];
      for (auto j = 0; j < arr_size; j++) {
        a[j] = ACopyable(j);
      }
      arr[i] = ACopyable(i);
    }

    CaptureAndCopy<DataT, arr_size>(arr_arr, arr, result_arr_arr, &result_arr,
                                    q);

    for (auto i = 0; i < arr_size; i++) {
      auto a = result_arr_arr[i];
      for (auto j = 0; j < arr_size; j++) {
        assert(a[j].i == j);
      }
      assert(result_arr[i].i == i);
    }
  }

  {
    using DataT = std::optional<int>;
    DataT opt_arr[arr_size];
    DataT opt;
    DataT result_opt_arr[arr_size];
    DataT result_opt;

    for (auto i = 0; i < arr_size; i++) {
      opt_arr[i] = i;
    }
    opt = ref_val;

    CaptureAndCopy<DataT, arr_size>(opt_arr, opt, result_opt_arr, &result_opt,
                                    q);

    for (auto i = 0; i < arr_size; i++) {
      assert(result_opt_arr[i] == i);
    }

    assert(result_opt == ref_val);
  }

  {
    using DataT = std::optional<ACopyable>;
    DataT opt_arr[arr_size];
    DataT opt;
    DataT result_opt_arr[arr_size];
    DataT result_opt;

    for (auto i = 0; i < arr_size; i++) {
      opt_arr[i] = ACopyable(i);
    }
    opt = ACopyable(ref_val);

    CaptureAndCopy<DataT, arr_size>(opt_arr, opt, result_opt_arr, &result_opt,
                                    q);

    for (auto i = 0; i < arr_size; i++) {
      assert(result_opt_arr[i]->i == i);
    }

    assert(result_opt->i == ref_val);
  }

  {
    using DataT = std ::string_view;
    std::string strv_arr_val[arr_size];
    std::string strv_val{std::to_string(ref_val)};
    DataT strv_arr[arr_size];
    DataT strv{strv_val};
    DataT result_strv_arr[arr_size];
    DataT result_strv;

    for (auto i = 0; i < arr_size; i++) {
      strv_arr_val[i] = std::to_string(i);
      strv_arr[i] = std::string_view{strv_arr_val[i]};
    }

    CaptureAndCopy<DataT, arr_size>(strv_arr, strv, result_strv_arr,
                                    &result_strv, q);

    for (auto i = 0; i < arr_size; i++) {
      assert(result_strv_arr[i] == std::to_string(i));
    }

    assert(result_strv == std::to_string(ref_val));
  }

#if __cpp_lib_span >= 202002
  {
    using DataT = std::span<int>;
    std::vector<int> v(arr_size);
    DataT s_arr[arr_size];
    DataT s{v.data(), arr_size};
    DataT result_s_arr[arr_size];
    DataT result_s{v.data(), arr_size};

    for (auto i = 0; i < arr_size; i++) {
      s[i] = i;
      s_arr[i] = s;
    }

    CaptureAndCopy<DataT, arr_size>(s_arr, s, result_s_arr, &result_s, q);

    for (auto i = 0; i < arr_size; i++) {
      assert(result_s[i] == i);
      for (auto j = 0; j < arr_size; j++) {
        assert(result_s_arr[i][j] == j);
      }
    }
  }
#endif

  {
    using DataT = sycl::span<int>;
    std::vector<int> v(arr_size);
    DataT s_arr[arr_size];
    DataT s{v.data(), arr_size};
    DataT result_s_arr[arr_size];
    DataT result_s{v.data(), arr_size};

    for (auto i = 0; i < arr_size; i++) {
      s[i] = i;
      s_arr[i] = s;
    }

    CaptureAndCopy<DataT, arr_size>(s_arr, s, result_s_arr, &result_s, q);

    for (auto i = 0; i < arr_size; i++) {
      assert(result_s[i] == i);
      for (auto j = 0; j < arr_size; j++) {
        assert(result_s_arr[i][j] == j);
      }
    }
  }

  std::cout << "Test passed" << std::endl;
}
