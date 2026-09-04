// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: level_zero, gpu
// RUN: env SYCL_UR_USE_LEVEL_ZERO_V2=1 SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P=1 %{run} %t.out
//
// Tests the host-mediated buffer migration fallback in
// ur_discrete_buffer_handle_t::getDevicePtr, taken when a buffer must move
// between two devices that lack P2P access.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

buffer<int, 1> createInitializedBuffer(std::size_t Size, int Value) {
  buffer<int, 1> Buf{range<1>(Size)};
  host_accessor Init{Buf, write_only};
  for (std::size_t I = 0; I < Size; ++I)
    Init[I] = Value;
  return Buf;
}

property_list makeQueueProps(bool InOrder, bool Batched) {
  if (InOrder && Batched)
    return {property::queue::in_order(),
            ext::intel::property::queue::no_immediate_command_list()};
  if (InOrder)
    return {property::queue::in_order()};
  if (Batched)
    return {ext::intel::property::queue::no_immediate_command_list()};
  return {};
}

int runMigration(const context &Ctx, const std::vector<device> &Devices,
                 bool InOrder, bool Batched) {
  constexpr std::size_t NumBuffers = 8;
  constexpr std::size_t Size = 256;

  std::vector<buffer<int, 1>> Bufs;
  for (std::size_t B = 0; B < NumBuffers; ++B)
    Bufs.push_back(createInitializedBuffer(Size, 0));

  device DevA = Devices[0], DevB = Devices[1];
  property_list Props = makeQueueProps(InOrder, Batched);
  queue QA(Ctx, DevA, Props);
  queue QB(Ctx, DevB, Props);

  for (auto &Buf : Bufs)
    QA.submit([&](handler &H) {
      accessor Acc{Buf, H, read_write};
      H.parallel_for(range<1>(Size), [=](id<1> Id) { Acc[Id] = Id; });
    });
  QA.wait();

  for (auto &Buf : Bufs)
    QB.submit([&](handler &H) {
      accessor Acc{Buf, H, read_write};
      H.parallel_for(range<1>(Size), [=](id<1> Id) { Acc[Id] += 1; });
    });
  QB.wait();

  std::cout << (Batched ? "batched " : "immediate ")
            << (InOrder ? "in-order: " : "out-of-order: ");
  for (auto &Buf : Bufs) {
    host_accessor HostAcc{Buf, read_only};
    for (std::size_t I = 0; I < Size; ++I)
      if (HostAcc[I] != static_cast<int>(I) + 1) {
        std::cout << "FAILED" << std::endl;
        return 1;
      }
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}

int main() {
  auto Devices = platform(gpu_selector_v).get_devices(info::device_type::gpu);
  if (Devices.size() < 2) {
    std::cout << "Test requires at least two devices, skipping." << std::endl;
    return 0;
  }

  context Ctx(Devices);

  int Result = 0;
  for (bool Batched : {false, true})
    for (bool InOrder : {false, true})
      Result |= runMigration(Ctx, Devices, InOrder, Batched);
  return Result;
}
