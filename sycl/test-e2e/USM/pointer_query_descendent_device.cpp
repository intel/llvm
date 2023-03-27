// RUN: %clangxx -fsycl %s -o %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

// Use of descendent devices in opencl contexts is not supported yet.
// UNSUPPORTED: opencl
//==------ pointer_query_descendent_device.cpp - Pointer Query test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <cassert>

std::vector<sycl::device> getSubDevices(sycl::device &Dev) {
  uint32_t MaxSubDevices =
      Dev.get_info<sycl::info::device::partition_max_sub_devices>();
  if (MaxSubDevices == 0)
    return {};
  try {
    return Dev
        .create_sub_devices<sycl::info::partition_property::partition_equally>(
            1);
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::feature_not_supported);
  }
  try {
    std::vector<size_t> Counts(MaxSubDevices, 1);
    return Dev.create_sub_devices<
        sycl::info::partition_property::partition_by_counts>(Counts);
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::feature_not_supported);
  }
  try {
    return Dev.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::next_partitionable);
  } catch (sycl::exception &e) {
    assert(e.code() == sycl::errc::feature_not_supported);
  }
  assert(false &&
         "MaxSubDevices is not 0, but none of the partition types worked");
  return {};
}

int main() {
  sycl::device Dev;
  sycl::context Ctx{Dev};

  std::vector<sycl::device> SubDevices = getSubDevices(Dev);
  if (SubDevices.empty())
    return 0;
  sycl::queue Q{Ctx, SubDevices[0]};
  void *Data = sycl::malloc_device(64, Q);
  assert(sycl::get_pointer_type(Data, Ctx) == sycl::usm::alloc::device);
  assert(sycl::get_pointer_device(Data, Ctx) == SubDevices[0]);

  free(Data, Ctx);
}
