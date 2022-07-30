// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// Specialization constants are not supported on FPGA h/w and emulator.
// UNSUPPORTED: cuda || hip
//
//==----------- specialization_constants.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Basic checks for some primitive types

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

#define HALF 0 // FIXME Spec constants do not support half type yet

class SpecializedKernel;

class MyBoolConst;
class MyInt8Const;
class MyUInt8Const;
class MyInt16Const;
class MyUInt16Const;
class MyInt32Const;
class MyUInt32Const;
class MyInt64Const;
class MyUInt64Const;
class MyHalfConst;
class MyFloatConst;
class MyDoubleConst;

using namespace sycl;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937_64 rnd(seed);

bool bool_ref = true;
// Fetch a value at runtime.
int8_t int8_ref = rnd() % std::numeric_limits<int8_t>::max();
uint8_t uint8_ref = rnd() % std::numeric_limits<uint8_t>::max();
int16_t int16_ref = rnd() % std::numeric_limits<int16_t>::max();
uint16_t uint16_ref = rnd() % std::numeric_limits<uint16_t>::max();
int32_t int32_ref = rnd() % std::numeric_limits<int32_t>::max();
uint32_t uint32_ref = rnd() % std::numeric_limits<uint32_t>::max();
int64_t int64_ref = rnd() % std::numeric_limits<int64_t>::max();
uint64_t uint64_ref = rnd() % std::numeric_limits<uint64_t>::max();
half half_ref = rnd() % std::numeric_limits<uint16_t>::max();
float float_ref = rnd() % std::numeric_limits<uint32_t>::max();
double double_ref = rnd() % std::numeric_limits<uint64_t>::max();

template <typename T1, typename T2>
bool check(const T1 &test, const T2 &ref, std::string type) {

  if (test != ref) {
    std::cout << "Test != Reference: " << std::to_string(test)
              << " != " << std::to_string(ref) << " for type: " << type << "\n";
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  std::cout << "check specialization constants API. (seed =" << seed << "\n";

  auto exception_handler = [&](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "an async SYCL exception was caught: "
                  << std::string(e.what());
      }
    }
  };
  try {
    auto q = queue(exception_handler);
    program prog(q.get_context());

    // Create specialization constants.
    ext::oneapi::experimental::spec_constant<bool, MyBoolConst> i1 =
        prog.set_spec_constant<MyBoolConst>(bool_ref);
    ext::oneapi::experimental::spec_constant<int8_t, MyInt8Const> i8 =
        prog.set_spec_constant<MyInt8Const>(int8_ref);
    ext::oneapi::experimental::spec_constant<uint8_t, MyUInt8Const> ui8 =
        prog.set_spec_constant<MyUInt8Const>(uint8_ref);
    ext::oneapi::experimental::spec_constant<int16_t, MyInt16Const> i16 =
        prog.set_spec_constant<MyInt16Const>(int16_ref);
    ext::oneapi::experimental::spec_constant<uint16_t, MyUInt16Const> ui16 =
        prog.set_spec_constant<MyUInt16Const>(uint16_ref);
    ext::oneapi::experimental::spec_constant<int32_t, MyInt32Const> i32 =
        prog.set_spec_constant<MyInt32Const>(int32_ref);
    ext::oneapi::experimental::spec_constant<uint32_t, MyUInt32Const> ui32 =
        prog.set_spec_constant<MyUInt32Const>(uint32_ref);
    ext::oneapi::experimental::spec_constant<int64_t, MyInt64Const> i64 =
        prog.set_spec_constant<MyInt64Const>(int64_ref);
    ext::oneapi::experimental::spec_constant<uint64_t, MyUInt64Const> ui64 =
        prog.set_spec_constant<MyUInt64Const>(uint64_ref);
#if HALF
    ext::oneapi::experimental::spec_constant<sycl::half, MyHalfConst> f16 =
        prog.set_spec_constant<MyHalfConst>(half_ref);
#endif
    ext::oneapi::experimental::spec_constant<float, MyFloatConst> f32 =
        prog.set_spec_constant<MyFloatConst>(float_ref);

    ext::oneapi::experimental::spec_constant<double, MyDoubleConst> f64 =
        prog.set_spec_constant<MyDoubleConst>(double_ref);

    prog.build_with_kernel_type<SpecializedKernel>();

    bool bool_test = 0;
    int8_t int8_test = 0;
    uint8_t uint8_test = 0;
    int16_t int16_test = 0;
    uint16_t uint16_test = 0;
    int32_t int32_test = 0;
    uint32_t uint32_test = 0;
    int64_t int64_test = 0;
    uint64_t uint64_test = 0;
    half half_test = 0;
    float float_test = 0;
    double double_test = 0;

    {
      buffer<bool> bool_buf(&bool_test, 1);
      buffer<int8_t> int8_buf(&int8_test, 1);
      buffer<uint8_t> uint8_buf(&uint8_test, 1);
      buffer<int16_t> int16_buf(&int16_test, 1);
      buffer<uint16_t> uint16_buf(&uint16_test, 1);
      buffer<int32_t> int32_buf(&int32_test, 1);
      buffer<uint32_t> uint32_buf(&uint32_test, 1);
      buffer<int64_t> int64_buf(&int64_test, 1);
      buffer<uint64_t> uint64_buf(&uint64_test, 1);
      buffer<half> half_buf(&half_test, 1);
      buffer<float> float_buf(&float_test, 1);
      buffer<double> double_buf(&double_test, 1);

      q.submit([&](handler &cgh) {
        auto bool_acc = bool_buf.get_access<access::mode::write>(cgh);
        auto int8_acc = int8_buf.get_access<access::mode::write>(cgh);
        auto uint8_acc = uint8_buf.get_access<access::mode::write>(cgh);
        auto int16_acc = int16_buf.get_access<access::mode::write>(cgh);
        auto uint16_acc = uint16_buf.get_access<access::mode::write>(cgh);
        auto int32_acc = int32_buf.get_access<access::mode::write>(cgh);
        auto uint32_acc = uint32_buf.get_access<access::mode::write>(cgh);
        auto int64_acc = int64_buf.get_access<access::mode::write>(cgh);
        auto uint64_acc = uint64_buf.get_access<access::mode::write>(cgh);
        auto half_acc = half_buf.get_access<access::mode::write>(cgh);
        auto float_acc = float_buf.get_access<access::mode::write>(cgh);
        auto double_acc = double_buf.get_access<access::mode::write>(cgh);
        cgh.single_task<SpecializedKernel>(prog.get_kernel<SpecializedKernel>(),
                                           [=]() {
                                             bool_acc[0] = i1.get();
                                             int8_acc[0] = i8.get();
                                             uint8_acc[0] = ui8.get();
                                             int16_acc[0] = i16.get();
                                             uint16_acc[0] = ui16.get();
                                             int32_acc[0] = i32.get();
                                             uint32_acc[0] = ui32.get();
                                             int64_acc[0] = i64.get();
                                             uint64_acc[0] = ui64.get();
#if HALF
                                             half_acc[0] = f16.get();
#endif
                                             float_acc[0] = f32.get();
                                             double_acc[0] = f64.get();
                                           });
      });
    }
    if (!check(bool_test, bool_ref, "bool"))
      return 1;
    if (!check(int8_test, int8_ref, "int8"))
      return 1;
    if (!check(uint8_test, uint8_ref, "uint8"))
      return 1;
    if (!check(int16_test, int16_ref, "int16"))
      return 1;
    if (!check(uint16_test, uint16_ref, "uint16"))
      return 1;
    if (!check(int32_test, int32_ref, "int32"))
      return 1;
    if (!check(uint32_test, uint32_ref, "uint32"))
      return 1;
    if (!check(int64_test, int64_ref, "int64"))
      return 1;
    if (!check(uint64_test, uint64_ref, "uint64"))
      return 1;
#if HALF
    if (!check(half_test, half_ref, "half"))
      return 1;
#endif
    if (!check(float_test, float_ref, "float"))
      return 1;
    if (!check(double_test, double_ref, "double"))
      return 1;
  } catch (const exception &e) {
    std::cout << "an async SYCL exception was caught: "
              << std::string(e.what());
    return 1;
  }
  return 0;
}
