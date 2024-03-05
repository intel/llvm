//==------------ bfloat16_vec.cpp - test sycl::vec<bfloat16>----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO currently the feature isn't supported on FPGA.
// TODO enable opaque pointers support on CPU.
// UNSUPPORTED: cpu || accelerator

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{  %{build} -fpreview-breaking-changes -o %t2.out   %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out  %}

#include <sycl/sycl.hpp>

constexpr unsigned N =
    10; // init plus arithmetic + - * /   for vec<1> and vec<2>

int main() {

  // clang-format off
    using T = sycl::ext::oneapi::bfloat16;

    sycl::queue q;

    T a0{ 2.0f };
    T b0 { 6.0f };

    T addition_ref0 = a0 + b0;
    T subtraction_ref0 = a0 - b0;
    T multiplication_ref0 = a0 * b0;
    T division_ref0 = a0 / b0;

    std::cout << " ===  vec<bfloat16, 1> === " << std::endl;
    std::cout << " ---  ON HOST --- " << std::endl;
    sycl::vec<T, 1>  init_float{ 2.0f };
    sycl::vec<T, 1>  oneA{ a0 }, oneB{ b0 };
    sycl::vec<T, 1>  simple_addition = oneA + oneB;
    sycl::vec<T, 1>  simple_subtraction = oneA - oneB;
    sycl::vec<T, 1>  simple_multiplication = oneA * oneB;
    sycl::vec<T, 1>  simple_division = oneA / oneB;

    std::cout << "iniitialization     : " << oneA[0]             << " float: " << init_float[0] << std::endl;
    std::cout << "addition.        ref: " << addition_ref0       << " vec: " << simple_addition[0] << std::endl;
    std::cout << "subtraction.     ref: " << subtraction_ref0    << " vec: " << simple_subtraction[0] << std::endl;
    std::cout << "multiplication.  ref: " << multiplication_ref0 << " vec: " << simple_multiplication[0] << std::endl;
    std::cout << "division.        ref: " << division_ref0       << " vec: " << simple_division[0] << std::endl;

    assert(oneA[0] == init_float[0]);
    assert(addition_ref0 == simple_addition[0]);
    assert(subtraction_ref0 == simple_subtraction[0]);
    assert(multiplication_ref0 == simple_multiplication[0]);
    assert(division_ref0 == simple_division[0]);

    std::cout << " ---  ON DEVICE --- " << std::endl;
    sycl::range<1> r(N);
    sycl::buffer<bool, 1> buf(r);

    q.submit([&](sycl::handler &cgh) {
        sycl::stream out(2024, 400, cgh);
        sycl::accessor acc(buf, cgh, sycl::write_only );
        cgh.single_task([=](){
            sycl::vec<T, 1>  dev_float{ 2.0f };
            sycl::vec<T, 1>  device_addition = oneA + oneB;
            sycl::vec<T, 1>  device_subtraction = oneA - oneB;
            sycl::vec<T, 1>  device_multiplication = oneA * oneB;
            sycl::vec<T, 1>  device_division = oneA / oneB;

            out << "iniitialization     : " << oneA[0]             << " float: " << dev_float[0] << sycl::endl;
            out << "addition.        ref: " << addition_ref0       << " vec: " << device_addition[0] << sycl::endl;
            out << "subtraction.     ref: " << subtraction_ref0    << " vec: " << device_subtraction[0] << sycl::endl;
            out << "multiplication.  ref: " << multiplication_ref0 << " vec: " << device_multiplication[0] << sycl::endl;
            out << "division.        ref: " << division_ref0       << " vec: " << device_division[0] << sycl::endl;

            acc[0] = (oneA[0] == dev_float[0]);
            acc[1] = (addition_ref0 == device_addition[0]);
            acc[2] = (subtraction_ref0 == device_subtraction[0]);
            acc[3] = (multiplication_ref0 == device_multiplication[0]);
            acc[4] = (division_ref0 == device_division[0]);
            
        }); 
    }).wait();


    // second value
    T a1 { 3.33333f };
    T b1 { 6.66666f };
    T addition_ref1 = a1 + b1;
    T subtraction_ref1 = a1 - b1;
    T multiplication_ref1 = a1 * b1;
    T division_ref1 = a1 / b1;

    std::cout << "\n ===  vec<bfloat16, 2> === " << std::endl;
    std::cout << " ---  ON HOST --- " << std::endl;
    sycl::vec<T, 2> twoA{ a0, a1 }, twoB{ b0, b1 };
    sycl::vec<T, 2> double_float { 2.0f, 3.33333f };
    sycl::vec<T, 2> double_addition = twoA + twoB;
    sycl::vec<T, 2> double_subtraction = twoA - twoB;
    sycl::vec<T, 2> double_multiplication = twoA * twoB;
    sycl::vec<T, 2> double_division = twoA / twoB;

    std::cout << "init ref: " << twoA[0]                << "    ref1: " << twoA[1] << std::endl;
    std::cout << "  float0: " << double_float[0]        << "  float1: " << double_float[1] << std::endl;
    std::cout << "+ ref0: " << addition_ref0            << "    ref1: " << addition_ref1 << std::endl;
    std::cout << "add[0]: " << double_addition[0]       << "  add[1]: " << double_addition[1] << std::endl;
    std::cout << "- ref0: " << subtraction_ref0         << "    ref1: " << subtraction_ref1 << std::endl;
    std::cout << "sub[0]: " << double_subtraction[0]    << "  sub[1]: " << double_subtraction[1] << std::endl;
    std::cout << "* ref0: " << multiplication_ref0      << "    ref1: " << multiplication_ref1 << std::endl;
    std::cout << "mul[0]: " << double_multiplication[0] << "  mul[1]: " << double_multiplication[1] << std::endl;
    std::cout << "/ ref0: " << division_ref0            << "    ref1: " << division_ref1 << std::endl;
    std::cout << "div[0]: " << double_division[0]       << "  div[1]: " << double_division[1] << std::endl;
    
    assert(twoA[0] == double_float[0]);                      assert(twoA[1] == double_float[1]);
    assert(addition_ref0 == double_addition[0]);             assert(addition_ref1 == double_addition[1]);
    assert(subtraction_ref0 == double_subtraction[0]);       assert(subtraction_ref1 == double_subtraction[1]);
    assert(multiplication_ref0 == double_multiplication[0]); assert(multiplication_ref1 == double_multiplication[1]);
    assert(division_ref0 == double_division[0]);             assert(division_ref1 == double_division[1]);
    
    std::cout << " ---  ON DEVICE --- " << std::endl;
    q.submit([&](sycl::handler &cgh) {
        sycl::stream out(2024, 400, cgh);
        sycl::accessor acc(buf, cgh, sycl::write_only );
        cgh.single_task([=](){
            sycl::vec<T, 2> device_float { 2.0f, 3.33333f };
            sycl::vec<T, 2> device_addition = twoA + twoB;
            sycl::vec<T, 2> device_subtraction = twoA - twoB;
            sycl::vec<T, 2> device_multiplication = twoA * twoB;
            sycl::vec<T, 2> device_division = twoA / twoB;

            out << "init ref: " << twoA[0]                << "    ref1: " << twoA[1] << sycl::endl;
            out << "  float0: " << device_float[0]        << "  float1: " << device_float[1] << sycl::endl;
            out << "+ ref0: " << addition_ref0            << "    ref1: " << addition_ref1 << sycl::endl;
            out << "add[0]: " << device_addition[0]       << "  add[1]: " << device_addition[1] << sycl::endl;
            out << "- ref0: " << subtraction_ref0         << "    ref1: " << subtraction_ref1 << sycl::endl;
            out << "sub[0]: " << device_subtraction[0]    << "  sub[1]: " << device_subtraction[1] << sycl::endl;
            out << "* ref0: " << multiplication_ref0      << "    ref1: " << multiplication_ref1 << sycl::endl;
            out << "mul[0]: " << device_multiplication[0] << "  mul[1]: " << device_multiplication[1] << sycl::endl;
            out << "/ ref0: " << division_ref0            << "    ref1: " << division_ref1 << sycl::endl;
            out << "div[0]: " << device_division[0]       << "  div[1]: " << device_division[1] << sycl::endl;

            acc[5] = (twoA[0] == device_float[0]) && (twoA[1] == device_float[1]);
            acc[6] = (addition_ref0 == device_addition[0]) && (addition_ref1 == device_addition[1]);
            acc[7] = (subtraction_ref0 == device_subtraction[0]) && (subtraction_ref1 == device_subtraction[1]);
            acc[8] = (multiplication_ref0 == device_multiplication[0]) && (multiplication_ref1 == device_multiplication[1]);
            acc[9] = (division_ref0 == device_division[0]) && (division_ref1 == device_division[1]);

        }); 
    }).wait();

    sycl::host_accessor h_acc(buf, sycl::read_only);
    for(unsigned i = 0; i < N; i++){
        assert(h_acc[i]);
    }

  // clang-format on
  return 0;
}
