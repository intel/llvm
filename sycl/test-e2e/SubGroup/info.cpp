// UNSUPPORTED: accelerator
// TODO: FPGAs currently report supported subgroups as {4,8,16,32,64}, causing
// this test to fail. Additionally, the kernel max_sub_group_size checks
// crash on FPGAs
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==------------- info.cpp - SYCL sub_group parameters test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "helper.hpp"
class kernel_sg;
using namespace sycl;

int main() {
  queue Queue;
  device Device = Queue.get_device();

  bool old_opencl = false;
  if (Device.get_backend() == sycl::backend::opencl) {
    // Extract the numerical version from the version string, OpenCL version
    // string have the format "OpenCL <major>.<minor> <vendor specific data>".
    std::string ver = Device.get_info<info::device::version>().substr(7, 3);
    old_opencl = (ver < "2.1");
  }

  /* Check info::device parameters. */
  if (!old_opencl) {
    // Independent forward progress is missing on OpenCL backend prior to
    // version 2.1
    Device.get_info<info::device::sub_group_independent_forward_progress>();
  }
  Device.get_info<info::device::max_num_sub_groups>();

  try {
    size_t max_wg_size = Device.get_info<info::device::max_work_group_size>();
    auto KernelID = get_kernel_id<kernel_sg>();
    auto KB = get_kernel_bundle<bundle_state::executable>(Queue.get_context(),
                                                          {KernelID});
    auto Kernel = KB.get_kernel(KernelID);
    range<2> GlobalRange{50, 40};

    buffer<float, 2> ABuf{GlobalRange}, BBuf{GlobalRange}, CBuf{GlobalRange};

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

    auto sg_sizes = Device.get_info<info::device::sub_group_sizes>();

    // Max sub-group size for a particular kernel might not be the max
    // supported size on the device in general. Can only check that it is
    // contained in list of valid sizes.
    Res = Kernel.get_info<info::kernel_device_specific::max_sub_group_size>(
        Device);
    bool Expected =
        std::find(sg_sizes.begin(), sg_sizes.end(), Res) != sg_sizes.end();
    exit_if_not_equal<bool>(Expected, true, "max_sub_group_size");

    for (auto r : {range<3>(3, 4, 5), range<3>(1, 1, 1), range<3>(4, 2, 1),
                   range<3>(32, 3, 4), range<3>(7, 9, 11)}) {
      Res = Kernel.get_info<info::kernel_device_specific::max_sub_group_size>(
          Device);
      Expected =
          std::find(sg_sizes.begin(), sg_sizes.end(), Res) != sg_sizes.end();
      exit_if_not_equal<bool>(Expected, true, "max_sub_group_size");
    }

    Res = Kernel.get_info<info::kernel_device_specific::compile_num_sub_groups>(
        Device);

    /* Sub-group size is not specified in kernel or IL*/
    exit_if_not_equal<uint32_t>(Res, 0, "compile_num_sub_groups");

    Res = Kernel.get_info<info::kernel_device_specific::compile_sub_group_size>(
        Device);

    /* Required sub-group size is not specified in kernel or IL*/
    exit_if_not_equal<uint32_t>(Res, 0, "compile_sub_group_size");

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  std::cout << "Test passed.\n";
  return 0;
}
