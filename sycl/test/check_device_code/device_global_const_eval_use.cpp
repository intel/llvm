// RUN: %clangxx -std=c++20 -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Tests that device_globals with device_image_scope property can be compile
// time constant initialized and that the constant value appears in the IR

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

// Constant int and array of ints device_globals
constexpr device_global<int, decltype(properties(device_image_scope))> 
   dg_int{5};
constexpr device_global<int[3], decltype(properties(device_image_scope))>
   dg_int_arr{5, 2, 3};

// Constant char and array of char device_globals
constexpr device_global<char, decltype(properties(device_image_scope))>
    dg_char{'f'};
constexpr device_global<char[3], decltype(properties(device_image_scope))>
    dg_char_arr{'d', '4', 'S'};

// Multidimensional array of integers
constexpr device_global<int[3][2], decltype(properties(device_image_scope))>
    dg_multi_dim_arr{3, 4, 5, 6, 7, 8};

// Constant float and array of float device_globals
constexpr device_global<float, decltype(properties(device_image_scope))>
   dg_float{4.5};
constexpr device_global<float[6], decltype(properties(device_image_scope))>
   dg_float_arr{4.5, 2.1, 3.5, 9.33, 2.33, 2.1};
   
// Constant double and array of double device_globals
constexpr device_global<double, decltype(properties(device_image_scope))>
   dg_double{3.56543};
constexpr device_global<double[3], decltype(properties(device_image_scope))>
   dg_double_arr{2.2341234, 233.23423, 236.52321};

// Constant bool and array of bool device_globals
constexpr device_global<bool, decltype(properties(device_image_scope))>
   dg_bool{true};
constexpr device_global<bool[3], decltype(properties(device_image_scope))>
   dg_bool_arr{true, false, true};

// Constant struct and array of struct device_globals
struct TestStruct {
  int field1;
  bool field2;
  float field3;
  int field4[4];
};
constexpr TestStruct TS1(5, true, 2.1, {1, 2, 3, 4});
constexpr TestStruct TS2(7, false, 2.4, {1, 2, 3, 4});
constexpr TestStruct TS3(6, false, 4.34534, {5, 6, 7, 8});
constexpr device_global<TestStruct, decltype(properties(device_image_scope))>
   dg_struct{TS3};
constexpr device_global<TestStruct[2], decltype(properties(device_image_scope))>
   dg_struct_arr{TS1, TS2};

// Struct with constexpr constructor
struct TestStruct2 {
   int value;
   constexpr TestStruct2(int val) : value(val) {}; 
};
constexpr TestStruct2 TS4{4};
constexpr device_global<TestStruct2, decltype(properties(device_image_scope))>
    dg_constexpr_constructor_struct{TS4};

int main () {
  sycl::queue Q;
  Q.submit([&](sycl::handler& h) {
    // Simple kernel that just copies over the values from the device_globals so
    // that we can observe the GlobalVariables that are created to represent
    // them in the IR
    h.single_task([=] {
      // Int and array of ints
      int test_int = dg_int;
      int test_int_arr[3];
      for (int i = 0; i < 3; i++) {
        test_int_arr[i] = dg_int_arr[i];
      }
      // CHECK: @{{[A-Za-z0-9_]*}}dg_int = internal addrspace(1) constant { i32 } { i32 5 }, align 4, !spirv.Decorations
      // CHECK: @{{[A-Za-z0-9_]*}}dg_int_arr = internal addrspace(1) constant { [3 x i32] } { [3 x i32] [i32 5, i32 2, i32 3] }, align 4, !spirv.Decorations

      // Char and array of chars
      char test_char = dg_char;
      int test_char_arr[3];
      for (int i = 0; i < 3; i++) {
         test_char_arr[i] = dg_char_arr[i];
      }
      // CHECK: @{{[A-Za-z0-9_]*}}dg_char = internal addrspace(1) constant { i8 } { i8 102 }, align 1, !spirv.Decorations
      // CHECK: @{{[A-Za-z0-9_]*}}dg_char_arr = internal addrspace(1) constant { [3 x i8] } { [3 x i8] c"d4S" }, align 1, !spirv.Decorations

      // Multidimensional array of integers
      int test_multidimensional_arr[3][2];
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 2; j++) {
            test_multidimensional_arr[i][j] = dg_multi_dim_arr[i][j];
         }
      }
      // CHECK: @{{[A-Za-z0-9_]*}}dg_multi_dim_arr = internal addrspace(1) constant { [3 x [2 x i32]] } { [3 x [2 x i32]] [[2 x i32] [i32 3, i32 4], [2 x i32] [i32 5, i32 6], [2 x i32] [i32 7, i32 8]] }, align 4, !spirv.Decorations

      // Float and array of floats
      float test_float = dg_float;
      float test_float_arr[6];
      for (int i = 0; i < 6; i++) {
        test_float_arr[i] = dg_float_arr[i];
      }
      // CHECK: @{{[A-Za-z0-9_]*}}g_float = internal addrspace(1) constant { float } { float 4.500000e+00 }, align 4, !spirv.Decorations
      // CHECK: @{{[A-Za-z0-9_]*}}dg_float_arr = internal addrspace(1) constant { [6 x float] } { [6 x float] [float 4.500000e+00, float 0x4000CCCCC0000000, float 3.500000e+00, float 0x4022A8F5C0000000, float 0x4002A3D700000000, float 0x4000CCCCC0000000] }, align 4, !spirv.Decorations

      // Double and array of doubles
      double test_double = dg_double;
      double test_double_arr[3];
      for (int i = 0; i < 3; i++) {
        test_double_arr[i] = dg_double_arr[i];
      }
      // CHECK: @{{[A-Za-z0-9_]*}}dg_double = internal addrspace(1) constant { double } { double 3.565430e+00 }, align 8, !spirv.Decorations
      // CHECK: @{{[A-Za-z0-9_]*}}dg_double_arr = internal addrspace(1) constant { [3 x double] } { [3 x double] [double 0x4001DF7C16D1D39D, double 0x406D277ECFE9B7BF, double 0x406D90BE22E5DE16] }, align 8, !spirv.Decorations

      // Bool and array of bools
      bool test_bool = dg_bool;
      bool test_bool_arr[3];
      for (int i = 0; i < 3; i++) {
        test_bool_arr[i] = dg_bool_arr[i];
      }
      // CHECK: @{{[A-Za-z0-9_]*}}dg_bool = internal addrspace(1) constant { i8 } { i8 1 }, align 1, !spirv.Decorations
      // CHECK: @{{[A-Za-z0-9_]*}}dg_bool_arr = internal addrspace(1) constant { [3 x i8] } { [3 x i8] c"\01\00\01" }, align 1, !spirv.Decorations

      // Struct and array of structs
      int test_struct_int = dg_struct.get().field1;
      TestStruct test_struct_arr[3];
      for (int i = 0; i < 2; i++) {
        test_struct_arr[i] = dg_struct_arr[i];
      }
      // CHECK: @{{[A-Za-z0-9_]*}}dg_struct = internal addrspace(1) constant { %struct.TestStruct } { %struct.TestStruct { i32 6, i8 0, float 0x401161A0C0000000, [4 x i32] [i32 5, i32 6, i32 7, i32 8] } }, align 4, !spirv.Decorations
      // CHECK: @{{[A-Za-z0-9_]*}}dg_struct_arr = internal addrspace(1) constant { [2 x %struct.TestStruct] } { [2 x %struct.TestStruct] [%struct.TestStruct { i32 5, i8 1, float 0x4000CCCCC0000000, [4 x i32] [i32 1, i32 2, i32 3, i32 4] }, %struct.TestStruct { i32 7, i8 0, float 0x4003333340000000, [4 x i32] [i32 1, i32 2, i32 3, i32 4] }] }, align 4, !spirv.Decorations
    
      // Struct with constexpr constructor
      int test_constexpr_struct = dg_constexpr_constructor_struct.get().value;
      // CHECK: @{{[A-Za-z0-9_]*}}dg_constexpr_constructor_struct = internal addrspace(1) constant { %struct.TestStruct2 } { %struct.TestStruct2 { i32 4 } }, align 4, !spirv.Decorations
    });
  });
}
