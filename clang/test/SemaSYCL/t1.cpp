// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020  -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;

h.single_task<class kernel_name_3>(
        []() [[intel::initiation_interval(4)]]{}); 
you need to use max_concurrency
int main() {
  deviceQueue.submit([&](sycl::handler &h) { 
h.single_task<class kernel_name_3>(
        []() [[intel::initiation_interval(4)]]{}); // expected-error{{'max_concurrency' attribute cannot be applied to types}}
  });
  return 0;
}
