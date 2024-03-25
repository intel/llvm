// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#include "helpers.hpp"

template <typename... Ts, typename FuncTy> void TestTypes(FuncTy Fn) {
  (Fn(Ts{}), ...);
}

int main() {
  auto TestBitSelect = [&](auto type_val) {
    using T = decltype(type_val);
    static_assert(std::is_integral_v<T>,
                  "Only integer test is implemented here!");
    test(F(bitselect), T{0b0110}, T{0b1100}, T{0b0011}, T{0b1010});
  };

  TestTypes<signed char, unsigned char, char, long, long long, unsigned long,
            unsigned long long>(TestBitSelect);

  auto TestSelect = [&](auto type_val) {
    using T = decltype(type_val);
    auto Select = F(select);

    test(Select, T{0}, T{1}, T{0}, true);
    test(Select, T{1}, T{1}, T{0}, false);
  };

  TestTypes<signed char, unsigned char, char>(TestSelect);

  return 0;
}
