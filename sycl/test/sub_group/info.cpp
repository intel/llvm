// REQUIRES: opencl

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==------------- info.cpp - SYCL sub_group parameters test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <CL/sycl.hpp>
class kernel_sg;
using namespace cl::sycl;

int main() {
  queue Queue;
  device Device = Queue.get_device();

  /* Basic sub-group functionality is supported as part of cl_khr_subgroups
   * extension or as core OpenCL 2.1 feature. */
  if (!core_sg_supported(Device)) {
    std::cout << "Skipping test\n";
    return 0;
  }
  /* Check info::device parameters. */
  Device.get_info<info::device::sub_group_independent_forward_progress>();
  Device.get_info<info::device::max_num_sub_groups>();
  /* sub_group_sizes can be quared only of cl_intel_required_subgroup_size
   * extention is supported by device*/
  if (Device.has_extension("cl_intel_required_subgroup_size"))
    Device.get_info<info::device::sub_group_sizes>();

  try {
    size_t max_sg_num = get_sg_size(Device);
    size_t max_wg_size = Device.get_info<info::device::max_work_group_size>();
    program Prog(Queue.get_context());
    /* TODO: replace with pure SYCL code when fixed problem with consumption
     * kernels defined using program objects on GPU device
    Prog.build_with_kernel_type<kernel_sg>();
    kernel Kernel = Prog.get_kernel<kernel_sg>();

    Queue.submit([&](cl::sycl::handler &cgh) {
      cgh.parallel_for<kernel_sg>(
          nd_range<2>(range<2>(50, 40), range<2>(10, 20)), Kernel,
          [=](nd_item<2> index) {});
    });*/
    Prog.build_with_source("kernel void "
                           "kernel_sg(global double* a, global double* b, "
                           "global double* c) {*a=*b+*c; }\n");
    kernel Kernel = Prog.get_kernel("kernel_sg");
    uint32_t Res = 0;
    for (auto r : {range<3>(3, 4, 5), range<3>(1, 1, 1), range<3>(4, 2, 1),
                   range<3>(32, 3, 4), range<3>(7, 9, 11)}) {
      Res = Kernel.get_sub_group_info<
          info::kernel_sub_group::max_sub_group_size>(Device, r);
      bool Expected = (Res == r.size() || Res == max_sg_num);
      exit_if_not_equal<bool>(Expected, true,
                              "max_sub_group_size");
    }

    Res = Kernel.get_sub_group_info<
        info::kernel_sub_group::compile_num_sub_groups>(Device);

    /* Sub-group size is not specified in kernel or IL*/
    exit_if_not_equal<uint32_t>(Res, 0, "compile_num_sub_groups");

    // According to specification, this kernel query requires `cl_khr_subgroups`
    // or `cl_intel_subgroups`
    if ((Device.has_extension("cl_khr_subgroups") ||
         Device.has_extension("cl_intel_subgroups")) &&
        Device.has_extension("cl_intel_required_subgroup_size")) {
      Res = Kernel.get_sub_group_info<
          info::kernel_sub_group::compile_sub_group_size>(Device);

      /* Required sub-group size is not specified in kernel or IL*/
      exit_if_not_equal<uint32_t>(Res, 0, "compile_sub_group_size");
    }

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  std::cout << "Test passed.\n";
  return 0;
}
