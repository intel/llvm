// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// Test initialization works

#include "sycl/sycl.hpp"

using namespace sycl;
namespace intel = sycl::ext::intel::experimental; // for fpga_datapath

struct foo {
  int i;
  float f;
  char c;
  bool b;
};

int main() {
  // Scalars
  constexpr intel::fpga_datapath<int> scalar_int{7};
  static_assert(scalar_int.get() == 7);

  constexpr intel::fpga_datapath<float> scalar_float{53.6f};
  static_assert(scalar_float.get() == 53.6f);

  constexpr intel::fpga_datapath<char> scalar_char{'!'};
  static_assert(scalar_char.get() == '!');

  constexpr intel::fpga_datapath<bool> scalar_bool{true};
  static_assert(scalar_bool.get());

  constexpr intel::fpga_datapath<foo> scalar_struct{8, 9.11f, '$', false};
  static_assert(scalar_struct.get().i == 8);

  // Aggregates
  constexpr intel::fpga_datapath<int[3]> aggr_int{7, 5, -8};
  static_assert(aggr_int[1] == 5);

  constexpr intel::fpga_datapath<float[3]> aggr_float{53.6f, -2.0f, 0.0f};
  static_assert(aggr_float[2] == 0.0f);

  constexpr intel::fpga_datapath<char[3]> aggr_char{'b', 'y', 'e'};
  static_assert(aggr_char[0] == 'b');

  constexpr intel::fpga_datapath<bool[1]> aggr_bool{false};
  static_assert(aggr_bool[0] == false);

  // a bit weird that there is no seperation between each individual structs
  constexpr intel::fpga_datapath<foo[2]> aggr_struct{8, 9.11f, '$', false,
                                                     6, 6.66f, '^', true};
  static_assert(aggr_struct[1].c == '^');

  // multi-dimensional array
  constexpr intel::fpga_datapath<int[2][3]> aggr_int_2d{
      7, 5, -8, 6, 7, 8,
  };
  static_assert(aggr_int_2d[1][2] == 8);

  // default initialization
  constexpr intel::fpga_datapath<int> default_int{};
  static_assert(default_int.get() == 0);

  constexpr intel::fpga_datapath<float> default_float{};
  static_assert(default_float.get() == 0.0f);

  constexpr intel::fpga_datapath<char> default_char{};
  static_assert(default_char.get() == 0);

  constexpr intel::fpga_datapath<bool> default_bool{};
  static_assert(default_bool.get() == false);

  constexpr intel::fpga_datapath<foo> default_struct{};
  static_assert(default_struct.get().i == 0);

  constexpr intel::fpga_datapath<int[3]> default_aggr_int{};
  static_assert(default_aggr_int[1] == 0);

  return 0;
}
