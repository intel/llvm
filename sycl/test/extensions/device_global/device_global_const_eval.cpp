// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Tests that device_globals with device_image_scope property can be compile
// time constant initialized.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

// Constant int and array of ints device_globals
constexpr device_global<int, decltype(properties(device_image_scope))> 
   dg_int{5};
constexpr device_global<int[3], decltype(properties(device_image_scope))>
   dg_int_arr{5, 2, 3};
static_assert(dg_int == 5);
static_assert(dg_int_arr[0] == 5);
static_assert(dg_int_arr[1] == 2);
static_assert(dg_int_arr[2] == 3);

// Constant char and array of char device_globals
constexpr device_global<char, decltype(properties(device_image_scope))>
    dg_char{'f'};
constexpr device_global<char[3], decltype(properties(device_image_scope))>
    dg_char_arr{'d', '4', 'S'};
static_assert(dg_char == 'f');
static_assert(dg_char_arr[0] == 'd');
static_assert(dg_char_arr[1] == '4');
static_assert(dg_char_arr[2] == 'S');

// Multidimensional array of integers
constexpr device_global<int[3][2], decltype(properties(device_image_scope))>
    dg_multi_dim_arr{3, 4, 5, 6, 7, 8};
static_assert(dg_multi_dim_arr[0][0] == 3);
static_assert(dg_multi_dim_arr[0][1] == 4);
static_assert(dg_multi_dim_arr[1][0] == 5);
static_assert(dg_multi_dim_arr[1][1] == 6);
static_assert(dg_multi_dim_arr[2][0] == 7);
static_assert(dg_multi_dim_arr[2][1] == 8);

// Constant float and array of float device_globals
constexpr device_global<float, decltype(properties(device_image_scope))>
   dg_float{4.5};
constexpr device_global<float[6], decltype(properties(device_image_scope))>
   dg_float_arr{4.5, 2.1, 3.5, 9.33, 2.33, 2.1};
static_assert(dg_float == 4.5);
static_assert(dg_float_arr[0] == float(4.5));
static_assert(dg_float_arr[1] == float(2.1));
static_assert(dg_float_arr[2] == float(3.5));
static_assert(dg_float_arr[3] == float(9.33));
static_assert(dg_float_arr[4] == float(2.33));
static_assert(dg_float_arr[5] == float(2.1));
   
// Constant double and array of double device_globals
constexpr device_global<double, decltype(properties(device_image_scope))>
   dg_double{3.56543};
constexpr device_global<double[3], decltype(properties(device_image_scope))>
   dg_double_arr{2.2341234, 233.23423, 236.52321};
static_assert(dg_double == double(3.56543));
static_assert(dg_double_arr[0] == double(2.2341234));
static_assert(dg_double_arr[1] == double(233.23423));
static_assert(dg_double_arr[2] == double(236.52321));

// Constant bool and array of bool device_globals
constexpr device_global<bool, decltype(properties(device_image_scope))>
   dg_bool{true};
constexpr device_global<bool[3], decltype(properties(device_image_scope))>
   dg_bool_arr{true, false, true};
static_assert(dg_bool == true);
static_assert(dg_bool_arr[0] == true);
static_assert(dg_bool_arr[1] == false);
static_assert(dg_bool_arr[2] == true);

// Constant struct and array of struct device_globals
struct TestStruct {
  int field1;
  bool field2;
  float field3;
  int field4[4];
};
constexpr TestStruct TS1(5, true, 2.1, {1, 2, 3, 4});
constexpr TestStruct TS2(7, false, 2.4, {1, 2, 3, 4});
constexpr device_global<TestStruct, decltype(properties(device_image_scope))>
   dg_struct{TS1};
constexpr device_global<TestStruct[2], decltype(properties(device_image_scope))>
   dg_struct_arr{TS1, TS2};
static_assert(dg_struct.get().field1 == 5);
static_assert(dg_struct.get().field2 == true);
static_assert(dg_struct.get().field3 == float(2.1));
static_assert(dg_struct.get().field4[0] == 1);
static_assert(dg_struct.get().field4[1] == 2);
static_assert(dg_struct.get().field4[2] == 3);
static_assert(dg_struct.get().field4[3] == 4);
static_assert(dg_struct_arr[1].field1 == 7);
static_assert(dg_struct_arr[1].field2 == false);
static_assert(dg_struct_arr[1].field3 == float(2.4));
static_assert(dg_struct_arr[1].field4[0] == 1);
static_assert(dg_struct_arr[1].field4[1] == 2);
static_assert(dg_struct_arr[1].field4[2] == 3);
static_assert(dg_struct_arr[1].field4[3] == 4);

// Test struct with constexpr constructor
struct TestStruct2 {
   int value;
   constexpr TestStruct2(int val) : value(val) {}; 
};
constexpr TestStruct2 TS3{4};
constexpr device_global<TestStruct2, decltype(properties(device_image_scope))>
    dg_constexpr_constructor_struct{TS3};
static_assert(dg_constexpr_constructor_struct.get().value == 4);