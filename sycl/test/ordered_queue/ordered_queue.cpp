// RUN: %clangxx -fsycl %s -o %t.out -L %opencl_libs_dir -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
//==---------- ordered_queue.cpp - SYCL ordered queue test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../helpers.hpp"
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

string_class get_type(const device &Dev) {
  return ((Dev.is_host())
              ? "host"
              : (Dev.is_gpu()
                     ? "OpenCL.GPU"
                     : (Dev.is_accelerator()) ? "OpenCL.ACC" : "OpenCL.CPU"));
}

void print_queue_info(const ordered_queue &q) {
  std::cout << "ID=" << std::hex
            << ((q.get_device().is_host()) ? nullptr : q.get()) << std::endl;
  std::cout << "ordered queue wraps " << get_type(q.get_device()) << " device"
            << std::endl;
}
int main() {
  {
    std::cout << "Create default queue." << std::endl;
    ordered_queue q;
    print_queue_info(q);
    if (!q.is_host()) {
      cl_command_queue_properties reportedProps;
      cl_int iRet =
          clGetCommandQueueInfo(q.get(), CL_QUEUE_PROPERTIES,
                                sizeof(reportedProps), &reportedProps, NULL);
      CHECK(CL_SUCCESS == iRet &&
            "Failed to obtain queue info from ocl device");
      std::cout << "Queue properties bits are " << reportedProps
                << " and OOO bit is " << CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
                << std::endl;
    }
  }

  auto Devices = device::get_devices();
  device &DeviceA = Devices[0];
  device &DeviceB = (Devices.size() > 1 ? Devices[1] : Devices[0]);
  {
    std::cout << "move constructor" << std::endl;
    ordered_queue Queue(DeviceA);
    size_t Hash = hash_class<ordered_queue>()(Queue);
    ordered_queue MovedQueue(std::move(Queue));
    CHECK(Hash == hash_class<ordered_queue>()(MovedQueue));
    CHECK(DeviceA.is_host() == MovedQueue.is_host());
    if (!DeviceA.is_host()) {
      CHECK(MovedQueue.get() != nullptr);
    }
  }
  {
    std::cout << "move assignment operator" << std::endl;
    ordered_queue Queue(DeviceA);
    size_t Hash = hash_class<ordered_queue>()(Queue);
    ordered_queue WillMovedQueue(DeviceB);
    WillMovedQueue = std::move(Queue);
    CHECK(Hash == hash_class<ordered_queue>()(WillMovedQueue));
    CHECK(DeviceA.is_host() == WillMovedQueue.is_host());
    if (!DeviceA.is_host()) {
      CHECK(WillMovedQueue.get() != nullptr);
    }
  }
  {
    std::cout << "copy constructor" << std::endl;
    ordered_queue Queue(DeviceA);
    size_t Hash = hash_class<ordered_queue>()(Queue);
    ordered_queue QueueCopy(Queue);
    CHECK(Hash == hash_class<ordered_queue>()(Queue));
    CHECK(Hash == hash_class<ordered_queue>()(QueueCopy));
    CHECK(Queue == QueueCopy);
    CHECK(Queue.is_host() == QueueCopy.is_host());
  }
  {
    std::cout << "copy assignment operator" << std::endl;
    ordered_queue Queue(DeviceA);
    size_t Hash = hash_class<ordered_queue>()(Queue);
    ordered_queue WillQueueCopy(DeviceB);
    WillQueueCopy = Queue;
    CHECK(Hash == hash_class<ordered_queue>()(Queue));
    CHECK(Hash == hash_class<ordered_queue>()(WillQueueCopy));
    CHECK(Queue == WillQueueCopy);
    CHECK(Queue.is_host() == WillQueueCopy.is_host());
  }

  {
    property_list pl = {};
    ordered_queue Queue(pl);
    try {
      Queue.throw_asynchronous();
    } catch (const std::bad_function_call &e) {
      std::cout << "Default asynchronous handler call failed: " << e.what()
                << std::endl;
      throw;
    }
  }

  {
    default_selector Selector;
    device Device = Selector.select_device();
    context Context(Device);
    ordered_queue Queue(Context, Selector);
    CHECK(Context == Queue.get_context());
  }
}
