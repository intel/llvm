// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// Specialization constants are not supported on FPGA h/w and emulator.
// UNSUPPORTED: cuda || hip
//
//==----------- specialization_constants_negative.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Checks for negative cases

#include <chrono>
#include <cstdint>
#include <random>
#include <sycl/sycl.hpp>

class SpecializedKernelNegative;

class MyUInt32ConstNegative;
class MyDoubleConstNegative;

using namespace sycl;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937_64 rnd(seed);

// Fetch a value at runtime.
uint32_t uint32_ref = rnd() % std::numeric_limits<uint32_t>::max();
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
  std::cout << "check specialization constants exceptions. (seed =" << seed
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
    ext::oneapi::experimental::spec_constant<uint32_t, MyUInt32ConstNegative>
        ui32 = prog.set_spec_constant<MyUInt32ConstNegative>(uint32_ref);

    prog.build_with_kernel_type<SpecializedKernelNegative>();

    // Excerpt from
    // https://github.com/codeplaysoftware/standards-proposals/pull/121:
    // Once the program is in a build state, the specialization constant
    // can no longer be changed for the program and call to
    // set_specialization_constant will throw a spec_const_error
    // exception.
    bool exception_was_thrown = false;
    try {
      ui32 = prog.set_spec_constant<MyUInt32ConstNegative>(uint32_ref + 1);
    } catch (const ext::oneapi::experimental::spec_const_error &e) {
      exception_was_thrown = true;
    }
    if (!exception_was_thrown) {
      std::cout << "Exception wasn't thrown\n";
      return 1;
    }

    uint32_t uint32_test = 0;
    {
      buffer<uint32_t> uint32_buf(&uint32_test, 1);

      q.submit([&](handler &cgh) {
        auto uint32_acc = uint32_buf.get_access<access::mode::write>(cgh);
        cgh.single_task<SpecializedKernelNegative>(
            prog.get_kernel<SpecializedKernelNegative>(),
            [=]() { uint32_acc[0] = ui32.get(); });
      });
    }
    check(uint32_test, uint32_ref, "uint32");

  } catch (const exception &e) {
    std::cout << "an async SYCL exception was caught: "
              << std::string(e.what());
    return 1;
  }
  return 0;
}
