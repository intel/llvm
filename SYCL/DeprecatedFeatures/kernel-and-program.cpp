// RUN: %clangxx -D__SYCL_INTERNAL_API -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// XFAIL: hip

//==--- kernel-and-program.cpp - SYCL kernel/program test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <utility>

int main() {

  // Single task invocation methods
  {
    cl::sycl::queue q;
    int data = 0;
    // Precompiled kernel invocation
    {
      cl::sycl::buffer<int, 1> buf(&data, cl::sycl::range<1>(1));
      cl::sycl::program prg(q.get_context());
      // Test program building
      assert(prg.get_state() == cl::sycl::program_state::none);
      prg.build_with_kernel_type<class SingleTask>();
      assert(prg.get_state() == cl::sycl::program_state::linked);
      assert(prg.has_kernel<class SingleTask>());
      cl::sycl::kernel krn = prg.get_kernel<class SingleTask>();
      assert(krn.get_context() == q.get_context());
      assert(krn.get_program() == prg);

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.single_task<class SingleTask>(krn, [=]() { acc[0] = acc[0] + 1; });
      });
      if (!q.is_host()) {
        const std::string integrationHeaderKernelName =
            cl::sycl::detail::KernelInfo<SingleTask>::getName();
        const std::string clKerneName =
            krn.get_info<cl::sycl::info::kernel::function_name>();
        assert(integrationHeaderKernelName == clKerneName);
      }
    }
    assert(data == 1);
  }
  // Parallel for with range
  {
    cl::sycl::queue q;
    std::vector<int> dataVec(10);
    std::iota(dataVec.begin(), dataVec.end(), 0);
    // Precompiled kernel invocation
    {
      cl::sycl::range<1> numOfItems(dataVec.size());
      cl::sycl::buffer<int, 1> buf(dataVec.data(), numOfItems);
      cl::sycl::program prg(q.get_context());
      assert(prg.get_state() == cl::sycl::program_state::none);
      // Test compiling -> linking
      prg.compile_with_kernel_type<class ParallelFor>();
      assert(prg.get_state() == cl::sycl::program_state::compiled);
      prg.link();
      assert(prg.get_state() == cl::sycl::program_state::linked);
      assert(prg.has_kernel<class ParallelFor>());
      cl::sycl::kernel krn = prg.get_kernel<class ParallelFor>();
      assert(krn.get_context() == q.get_context());
      assert(krn.get_program() == prg);

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class ParallelFor>(
            krn, numOfItems,
            [=](cl::sycl::id<1> wiID) { acc[wiID] = acc[wiID] + 1; });
      });
    }
    for (size_t i = 0; i < dataVec.size(); ++i) {
      assert(dataVec[i] == i + 1);
    }
  }

  // Parallel for with nd_range
  {
    cl::sycl::queue q;
    std::vector<int> dataVec(10);
    std::iota(dataVec.begin(), dataVec.end(), 0);

    // Precompiled kernel invocation
    // TODO run on host as well once local barrier is supported
    if (!q.is_host()) {
      {
        cl::sycl::range<1> numOfItems(dataVec.size());
        cl::sycl::range<1> localRange(2);
        cl::sycl::buffer<int, 1> buf(dataVec.data(), numOfItems);
        cl::sycl::program prg(q.get_context());
        assert(prg.get_state() == cl::sycl::program_state::none);
        prg.build_with_kernel_type<class ParallelForND>();
        assert(prg.get_state() == cl::sycl::program_state::linked);
        assert(prg.has_kernel<class ParallelForND>());
        cl::sycl::kernel krn = prg.get_kernel<class ParallelForND>();
        assert(krn.get_context() == q.get_context());
        assert(krn.get_program() == prg);

        q.submit([&](cl::sycl::handler &cgh) {
          auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
          cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local>
              localAcc(localRange, cgh);

          cgh.parallel_for<class ParallelForND>(
              krn, cl::sycl::nd_range<1>(numOfItems, localRange),
              [=](cl::sycl::nd_item<1> item) {
                size_t idx = item.get_global_linear_id();
                int pos = idx & 1;
                int opp = pos ^ 1;
                localAcc[pos] = acc[item.get_global_linear_id()];

                item.barrier(cl::sycl::access::fence_space::local_space);

                acc[idx] = localAcc[opp];
              });
        });
      }
      q.wait();
      for (size_t i = 0; i < dataVec.size(); ++i) {
        assert(dataVec[i] == (i ^ 1));
      }
    }
  }
}
