// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// Specialization constants are not supported on FPGA h/w and emulator.
// UNSUPPORTED: cuda || hip
//
//==----------- specialization_constants_override.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Checks that set_spec_constant can be used twice on the same program

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

class SpecializedKernelOverride;

class MyBoolConstOverride;
class MyUInt32ConstOverride;
class MyDoubleConstOverride;

using namespace sycl;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937_64 rnd(seed);

bool bool_ref = true;
bool bool_ref_override = false;
// Fetch a value at runtime.
uint32_t uint32_ref = rnd() % std::numeric_limits<uint32_t>::max();
double double_ref = rnd() % std::numeric_limits<uint64_t>::max();

// Values which override the previous ones
uint32_t uint32_ref_override = rnd() % std::numeric_limits<uint32_t>::max();
double double_ref_override = rnd() % std::numeric_limits<uint64_t>::max();

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
  std::cout << "check specialization constants overriding. (seed =" << seed
            << "\n";

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
    ext::oneapi::experimental::spec_constant<bool, MyBoolConstOverride> i1 =
        prog.set_spec_constant<MyBoolConstOverride>(bool_ref);
    ext::oneapi::experimental::spec_constant<uint32_t, MyUInt32ConstOverride>
        ui32 = prog.set_spec_constant<MyUInt32ConstOverride>(uint32_ref);
    ext::oneapi::experimental::spec_constant<double, MyDoubleConstOverride>
        f64 = prog.set_spec_constant<MyDoubleConstOverride>(double_ref);

    // Override specialization constants.
    i1 = prog.set_spec_constant<MyBoolConstOverride>(bool_ref_override);
    ui32 = prog.set_spec_constant<MyUInt32ConstOverride>(uint32_ref_override);
    f64 = prog.set_spec_constant<MyDoubleConstOverride>(double_ref_override);

    prog.build_with_kernel_type<SpecializedKernelOverride>();

    bool bool_test = true;
    uint32_t uint32_test = 0;
    double double_test = 0;

    {
      buffer<bool> bool_buf(&bool_test, 1);
      buffer<uint32_t> uint32_buf(&uint32_test, 1);
      buffer<double> double_buf(&double_test, 1);

      q.submit([&](handler &cgh) {
        auto bool_acc = bool_buf.get_access<access::mode::write>(cgh);
        auto uint32_acc = uint32_buf.get_access<access::mode::write>(cgh);
        auto double_acc = double_buf.get_access<access::mode::write>(cgh);
        cgh.single_task<SpecializedKernelOverride>(
            prog.get_kernel<SpecializedKernelOverride>(), [=]() {
              bool_acc[0] = i1.get();
              uint32_acc[0] = ui32.get();
              double_acc[0] = f64.get();
            });
      });
    }
    check(bool_test, bool_ref_override, "bool");
    check(uint32_test, uint32_ref_override, "uint32");
    check(double_test, double_ref_override, "double");

  } catch (const exception &e) {
    std::cout << "an async SYCL exception was caught: "
              << std::string(e.what());
    return 1;
  }
  return 0;
}
