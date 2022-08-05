// See https://github.com/intel/llvm/issues/2922 for more info
// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==------------- info.cpp - SYCL sub_group parameters test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
#include <sycl/sycl.hpp>
class kernel_sg;
using namespace sycl;

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

  try {
    size_t max_wg_size = Device.get_info<info::device::max_work_group_size>();
    auto KernelID = get_kernel_id<kernel_sg>();
    auto KB = get_kernel_bundle<bundle_state::executable>(Queue.get_context(),
                                                          {KernelID});
    auto Kernel = KB.get_kernel(KernelID);
    range<2> GlobalRange{50, 40};

    buffer<double, 2> ABuf{GlobalRange}, BBuf{GlobalRange}, CBuf{GlobalRange};

    Queue.submit([&](sycl::handler &cgh) {
      auto A = ABuf.get_access<access::mode::read_write>(cgh);
      auto B = BBuf.get_access<access::mode::read>(cgh);
      auto C = CBuf.get_access<access::mode::read>(cgh);
      cgh.parallel_for<kernel_sg>(
          nd_range<2>(GlobalRange, range<2>(10, 20)), [=](nd_item<2> index) {
            const id<2> GlobalID = index.get_global_id();
            A[GlobalID] = B[GlobalID] * C[GlobalID];
          });
    });
    uint32_t Res = 0;

    /* sub_group_sizes can be queried only if cl_intel_required_subgroup_size
     * extension is supported by device*/
    auto Vec = Device.get_info<info::device::extensions>();
    if (std::find(Vec.begin(), Vec.end(), "cl_intel_required_subgroup_size") !=
        std::end(Vec)) {
      auto sg_sizes = Device.get_info<info::device::sub_group_sizes>();
      for (auto r : {range<3>(3, 4, 5), range<3>(1, 1, 1), range<3>(4, 2, 1),
                     range<3>(32, 3, 4), range<3>(7, 9, 11)}) {
        Res = Kernel.get_info<info::kernel_device_specific::max_sub_group_size>(
            Device, r);
        bool Expected =
            std::find(sg_sizes.begin(), sg_sizes.end(), Res) != sg_sizes.end();
        exit_if_not_equal<bool>(Expected, true, "max_sub_group_size");
      }
    }

    Res = Kernel.get_info<info::kernel_device_specific::compile_num_sub_groups>(
        Device);

    /* Sub-group size is not specified in kernel or IL*/
    exit_if_not_equal<uint32_t>(Res, 0, "compile_num_sub_groups");

    // According to specification, this kernel query requires `cl_khr_subgroups`
    // or `cl_intel_subgroups`
    if ((std::find(Vec.begin(), Vec.end(), "cl_khr_subgroups") !=
         std::end(Vec)) ||
        std::find(Vec.begin(), Vec.end(), "cl_intel_subgroups") !=
                std::end(Vec) &&
            std::find(Vec.begin(), Vec.end(),
                      "cl_intel_required_subgroup_size") != std::end(Vec)) {
      Res =
          Kernel.get_info<info::kernel_device_specific::compile_sub_group_size>(
              Device);

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
