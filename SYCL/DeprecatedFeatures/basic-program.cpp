// RUN: %clangxx -D__SYCL_INTERNAL_API -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// XFAIL: hip

//==--- basic-program.cpp - Basic test of program and kernel APIs ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <iostream>

int main() {

  // Test program and kernel APIs when building a kernel.
  {
    cl::sycl::queue q;
    int data = 0;
    {
      cl::sycl::buffer<int, 1> buf(&data, cl::sycl::range<1>(1));
      cl::sycl::program prg(q.get_context());
      assert(prg.get_state() == cl::sycl::program_state::none);
      prg.build_with_kernel_type<class BuiltKernel>();
      assert(prg.get_state() == cl::sycl::program_state::linked);
      std::vector<std::vector<char>> binaries = prg.get_binaries();
      assert(prg.has_kernel<class BuiltKernel>());
      cl::sycl::kernel krn = prg.get_kernel<class BuiltKernel>();
      std::string name = krn.get_info<cl::sycl::info::kernel::function_name>();
      assert(prg.has_kernel(name));

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<class BuiltKernel>(krn, [=]() { acc[0] = acc[0] + 1; });
      });
    }
    assert(data == 1);
  }

  // Test program and kernel APIs when compiling / linking a kernel.
  {
    cl::sycl::queue q;
    int data = 0;
    {
      cl::sycl::buffer<int, 1> buf(&data, cl::sycl::range<1>(1));
      cl::sycl::program prg(q.get_context());
      assert(prg.get_state() == cl::sycl::program_state::none);
      prg.compile_with_kernel_type<class CompiledKernel>();
      assert(prg.get_state() == cl::sycl::program_state::compiled);
      prg.link();
      assert(prg.get_state() == cl::sycl::program_state::linked);
      std::vector<std::vector<char>> binaries = prg.get_binaries();
      assert(prg.has_kernel<class CompiledKernel>());
      cl::sycl::kernel krn = prg.get_kernel<class CompiledKernel>();
      std::string name = krn.get_info<cl::sycl::info::kernel::function_name>();
      assert(prg.has_kernel(name));

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<class CompiledKernel>(krn,
                                              [=]() { acc[0] = acc[0] + 1; });
      });
    }
    assert(data == 1);
  }

  {
    sycl::context context;
    std::vector<sycl::device> devices = context.get_devices();

    sycl::program prg1(context, sycl::property_list{});
    sycl::program prg2(
        context, devices,
        sycl::property_list{sycl::property::buffer::use_host_ptr{}});
    if (!prg2.has_property<sycl::property::buffer::use_host_ptr>()) {
      std::cerr << "Line " << __LINE__ << ": Property was not found"
                << std::endl;
      return 1;
    }

    sycl::property::buffer::use_host_ptr Prop =
        prg2.get_property<sycl::property::buffer::use_host_ptr>();
  }
}
