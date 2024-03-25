// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#include "helpers.hpp"

int main() {
  using namespace sycl;

  marray<half, 2> ma1_f16{1.f, 2.f};
  marray<half, 2> ma2_f16{1.f, 2.f};
  marray<half, 2> ma3_f16{2.f, 1.f};
  marray<half, 2> ma4_f16{2.f, 2.f};
  marray<half, 3> ma5_f16{2.f, 2.f, 1.f};
  marray<half, 3> ma6_f16{1.f, 5.f, 8.f};
  marray<half, 2> ma8_f16{1.f, 1.f};
  marray<half, 2> ma9_f16{0.5f, 0.5f};
  marray<half, 2> ma10_f16{2.f, 2.f};

  marray<float, 2> ma1{1.f, 2.f};
  marray<float, 2> ma2{1.f, 2.f};
  marray<float, 2> ma3{2.f, 1.f};
  marray<float, 2> ma4{2.f, 2.f};
  marray<float, 3> ma5{2.f, 2.f, 1.f};
  marray<float, 3> ma6{1.f, 5.f, 8.f};
  marray<int, 3> ma7{50, 2, 31};
  marray<float, 2> ma8{1.f, 1.f};
  marray<float, 2> ma9{0.5f, 0.5f};
  marray<float, 2> ma10{2.f, 2.f};
  marray<bool, 3> c(1, 0, 1);

  bool has_fp16 = queue{}.get_device().has(sycl::aspect::fp16);
  // clang-format off
  test(has_fp16, F(isequal),        marray<bool, 2>(true,  true),  ma1_f16, ma2_f16);
  test(has_fp16, F(isnotequal),     marray<bool, 2>{false, false}, ma1_f16, ma2_f16);
  test(has_fp16, F(isgreater),      marray<bool, 2>{false, true},  ma1_f16, ma3_f16);
  test(has_fp16, F(isgreaterequal), marray<bool, 2>{false, true},  ma1_f16, ma4_f16);
  test(has_fp16, F(isless),         marray<bool, 2>{false, true},  ma3_f16, ma1_f16);
  test(has_fp16, F(islessequal),    marray<bool, 2>{false, true},  ma4_f16, ma1_f16);
  test(has_fp16, F(islessgreater),  marray<bool, 2>{false, false}, ma1_f16, ma2_f16);
  test(has_fp16, F(isfinite),       marray<bool, 2>{true,  true},  ma1_f16);
  test(has_fp16, F(isinf),          marray<bool, 2>{false, false}, ma1_f16);
  test(has_fp16, F(isnan),          marray<bool, 2>{false, false}, ma1_f16);
  test(has_fp16, F(isnormal),       marray<bool, 2>{true,  true},  ma1_f16);
  test(has_fp16, F(isordered),      marray<bool, 2>{true,  true},  ma1_f16, ma2_f16);
  test(has_fp16, F(isunordered),    marray<bool, 2>{false, false}, ma1_f16, ma2_f16);
  test(has_fp16, F(signbit),        marray<bool, 2>{false, false}, ma1_f16);

  test(has_fp16, F(bitselect), marray<half, 2>{1.0, 1.0},      ma8_f16, ma9_f16, ma10_f16);
  test(has_fp16, F(select),    marray<half, 3>{1.0, 2.0, 8.0}, ma5_f16, ma6_f16, c);

  test(F(isequal),        marray<bool, 2>{true,  true},  ma1, ma2);
  test(F(isnotequal),     marray<bool, 2>{false, false}, ma1, ma2);
  test(F(isgreater),      marray<bool, 2>{false, true},  ma1, ma3);
  test(F(isgreaterequal), marray<bool, 2>{false, true},  ma1, ma4);
  test(F(isless),         marray<bool, 2>{false, true},  ma3, ma1);
  test(F(islessequal),    marray<bool, 2>{false, true},  ma4, ma1);
  test(F(islessgreater),  marray<bool, 2>{false, false}, ma1, ma2);
  test(F(isfinite),       marray<bool, 2>{true,  true},  ma1);
  test(F(isinf),          marray<bool, 2>{false, false}, ma1);
  test(F(isnan),          marray<bool, 2>{false, false}, ma1);
  test(F(isnormal),       marray<bool, 2>{true,  true},  ma1);
  test(F(isordered),      marray<bool, 2>{true,  true},  ma1, ma2);
  test(F(isunordered),    marray<bool, 2>{false, false}, ma1, ma2);
  test(F(signbit),        marray<bool, 2>{false, false}, ma1);

  test(F(bitselect), marray<float, 2>{1.0, 1.0},      ma8, ma9, ma10);
  test(F(select),    marray<float, 3>{1.0, 2.0, 8.0}, ma5, ma6, c);
  // clang-format on

  test(F(all), bool{false}, ma7);
  test(F(any), bool{false}, ma7);

  {
    // Extra tests for select/bitselect due to special handling required for
    // integer return types.

    marray<char, 2> a{0b1100, 0b0011};
    marray<char, 2> b{0b0011, 0b1100};
    marray<char, 2> c{0b1010, 0b1010};
    marray<char, 2> r{0b0110, 0b1001};

    auto BitSelect = F(bitselect);
    test(BitSelect, r, a, b, c);
    // Input values/results above are positive, so use the same values for
    // signed/unsigned char tests.
    [&](auto... xs) { test(BitSelect, marray<signed char, 2>{xs}...); }(r, a, b,
                                                                        c);
    [&](auto... xs) { test(BitSelect, marray<unsigned char, 2>{xs}...); }(r, a,
                                                                          b, c);

    auto Select = F(select);
    marray<bool, 2> c2{false, true};
    marray<char, 2> r2{a[0], b[1]};
    test(Select, r2, a, b, c2);
    [&](auto... xs) { test(Select, marray<signed char, 2>{xs}..., c2); }(r2, a,
                                                                         b);
    [&](auto... xs) {
      test(Select, marray<unsigned char, 2>{xs}..., c2);
    }(r2, a, b);
  }

  return 0;
}
