// RUN: %clangxx -std=c++20 -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Tests that device_globals can be templated on arbitrary types but still be
// able to use the operators of said type

#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;

using dg_no_host = decltype(properties(device_image_scope, host_access_none));
device_global<fpga_mem<int>, dg_no_host> dg_int_simple;
static device_global<fpga_mem<bool>, dg_no_host> dg_bool;

int main() {
  sycl::queue Q;
  Q.single_task([]() {
    // Tests for device_global templated on complex types
    dg_int_simple = 4;
    dg_int_simple++;
    ++dg_int_simple;
    dg_int_simple--;
    --dg_int_simple;
    3 + dg_int_simple;
    dg_int_simple + 3;
    5 - dg_int_simple;
    dg_int_simple- 5;
    dg_int_simple * 5;
    5 * dg_int_simple;
    5 / dg_int_simple;
    dg_int_simple / 5;
    5 % dg_int_simple;
    5 << dg_int_simple;
    dg_int_simple << 5;
    5 >> dg_int_simple;
    dg_int_simple >> 5;
    int result10 = dg_int_simple % 5;
    bool result11 = result10 == dg_int_simple;
    bool result12 = dg_int_simple == result10;
    bool result13 = result10 < dg_int_simple;
    bool result14 = dg_int_simple < result10;
    bool result15 = result10 <= dg_int_simple;
    bool result16 = dg_int_simple <= result10;
    bool result17 = result10 > dg_int_simple;
    bool result18 = dg_int_simple > result10;
    bool result19 = result10 >= dg_int_simple;
    bool result20 = dg_int_simple >= result10;
    bool result21 = dg_int_simple != result10;
    bool result22 = result10 != dg_int_simple;
    dg_int_simple & result10;
    result10 & dg_int_simple;
    dg_int_simple | result10;
    result10 | dg_int_simple;
    result10^dg_int_simple;
    dg_int_simple^result10;
    ~dg_int_simple; 
    true && dg_bool;
    dg_bool && true;
    dg_bool || false;
    false || dg_bool;
    !dg_bool;

    // The following is one place where we will still need to call .get()
    int test2 = dg_int_simple.get();
  });
}