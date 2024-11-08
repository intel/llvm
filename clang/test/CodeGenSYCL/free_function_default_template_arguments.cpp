// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// This test checks integration header contents free functions kernels in
// presense of template arguments.

#include "mock_properties.hpp"
#include "sycl.hpp"

namespace ns {

struct notatuple {
  int a;
};

template <typename T, typename = int, int a = 12, typename = notatuple, typename ...TS> struct Arg {
  T val;
};

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel",
                                              2)]] void
simple(Arg<char>){
}

}


template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]] void
templated(ns::Arg<T, float, 3>, T end) {
}

template void templated(ns::Arg<int, float, 3>, int);

using namespace ns;
