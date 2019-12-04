// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out cpu
// RUN: %t.out gpu
// RUN: %t.out acc
// RUN: %t.out host
//==--- program-link.cpp - SYCL program link test --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>
using namespace cl::sycl;

template <class kernel_name = void> struct kernel_functor {
  void operator()() {}
};

template <typename kernel_name>
program get_compiled_program(const context &cxt, const device &dev) {
  program prg{cxt};
  queue q{dev};
  q.submit(
      [](handler &cgh) { cgh.single_task(kernel_functor<kernel_name>()); });
  q.wait_and_throw();
  prg.compile_with_kernel_type<kernel_functor<kernel_name>>("");
  return prg;
}

int test_case(const device &dev) {
  // Check program constructor with program list
  {
    context cxt{dev};
    vector_class<program> prgList;
    prgList.push_back(get_compiled_program<struct kernel1>(cxt, dev));
    prgList.push_back(get_compiled_program<struct kernel2>(cxt, dev));
    program res{prgList};
  }

  {
    try {
      vector_class<program> prgList;
      program res{prgList};
      return 1;
    } catch (std::out_of_range e) {
      // Expected exception caught
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {

  std::string type(argv[1]);
  try {
    if (type == "cpu")
      return test_case(cpu_selector().select_device());
    if (type == "gpu")
      return test_case(gpu_selector().select_device());
    if (type == "acc")
      return test_case(accelerator_selector().select_device());
    if (type == "host")
      return test_case(host_selector().select_device());
    throw std::string("Bad type: ") + type;
  } catch (runtime_error e) {
    std::cout << "No device of " << type << ". Skipping..." << std::endl;
  }
  return 0;
}
