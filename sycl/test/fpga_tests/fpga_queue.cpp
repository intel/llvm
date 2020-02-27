// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -L %opencl_libs_dir -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda
//==------------- fpga_queue.cpp - SYCL FPGA queues test -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>
#include <set>

using namespace cl::sycl;

const int dataSize = 32;
const int maxNumQueues = 256;

void GetCLQueue(event sycl_event, std::set<cl_command_queue>& cl_queues) {
  try {
    cl_command_queue cl_queue;
    cl_event cl_event = sycl_event.get();
    cl_int error = clGetEventInfo(cl_event, CL_EVENT_COMMAND_QUEUE,
                                  sizeof(cl_queue), &cl_queue, nullptr);
    assert(CL_SUCCESS == error && "Failed to obtain queue from OpenCL event");

    cl_queues.insert(cl_queue);
  } catch (invalid_object_error e) {
    std::cout << "Failed to get OpenCL queue from SYCL event: " << e.what()
              << std::endl;
  }
}

int getExpectedQueueNumber(cl_device_id device_id, int default_value) {
   cl_command_queue_properties reportedProps;
   cl_int iRet = clGetDeviceInfo(device_id,
                                 CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                                 sizeof(reportedProps),
                                 &reportedProps,
                                 NULL);
   assert(CL_SUCCESS == iRet && "Failed to obtain queue info from ocl device");
   return (reportedProps & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
              ? 1 : default_value;
}

int main() {
  int data[dataSize] = {0};

  {
    queue Queue;
    std::set<cl_command_queue> cl_queues;
    event sycl_event;

    // Purpose of this test is to check how many OpenCL queues are being
    // created from 1 SYCL queue for FPGA device. For that we submit 3 kernels
    // expecting 3 OpenCL queues created as a result.
    buffer<int, 1> bufA (data, range<1>(dataSize));
    buffer<int, 1> bufB (data, range<1>(dataSize));
    buffer<int, 1> bufC (data, range<1>(dataSize));

    sycl_event = Queue.submit([&](handler& cgh) {
      auto writeBuffer = bufA.get_access<access::mode::write>(cgh);

      // Create a range.
      auto myRange = range<1>(dataSize);

      // Create a kernel.
      auto myKernel = ([=](id<1> idx) {
        writeBuffer[idx] = idx[0];
      });

      cgh.parallel_for<class fpga_writer_1>(myRange, myKernel);
    });
    GetCLQueue(sycl_event, cl_queues);

    sycl_event = Queue.submit([&](handler& cgh) {
      auto writeBuffer = bufB.get_access<access::mode::write>(cgh);

      // Create a range.
      auto myRange = range<1>(dataSize);

      // Create a kernel.
      auto myKernel = ([=](id<1> idx) {
        writeBuffer[idx] = idx[0];
      });

      cgh.parallel_for<class fpga_writer_2>(myRange, myKernel);
    });
    GetCLQueue(sycl_event, cl_queues);

    sycl_event = Queue.submit([&](handler& cgh) {
      auto readBufferA = bufA.get_access<access::mode::read>(cgh);
      auto readBufferB = bufB.get_access<access::mode::read>(cgh);
      auto writeBuffer = bufC.get_access<access::mode::write>(cgh);

      // Create a range.
      auto myRange = range<1>(dataSize);

      // Create a kernel.
      auto myKernel = ([=](id<1> idx) {
        writeBuffer[idx] = readBufferA[idx] + readBufferB[idx];
      });

      cgh.parallel_for<class fpga_calculator>(myRange, myKernel);
    });
    GetCLQueue(sycl_event, cl_queues);

    int result = cl_queues.size();
    device dev = Queue.get_device();
    int expected_result = dev.is_host() ? 0 : getExpectedQueueNumber(dev.get(), 3);

    if (expected_result != result) {
      std::cout << "Result Num of queues = " << result << std::endl
                << "Expected Num of queues = "<< expected_result << std::endl;

      return -1;
    }

    auto readBufferC = bufC.get_access<access::mode::read>();
    for (size_t i = 0; i != dataSize; ++i) {
      if (readBufferC[i] != 2 * i) {
        std::cout << "Result mismatches " << readBufferC[i] << " Vs expected "
                  << 2 * i << " for index " << i << std::endl;
      }
    }
  }

  {
    queue Queue;
    std::set<cl_command_queue> cl_queues;
    event sycl_event;

    // Check limits of OpenCL queues creation for accelerator device.
    buffer<int, 1> buf (&data[0], range<1>(1));

    for (size_t i = 0; i != maxNumQueues + 1; ++i) {
      sycl_event = Queue.submit([&](handler& cgh) {
        auto Buffer = buf.get_access<access::mode::write>(cgh);

        // Create a kernel.
        auto myKernel = ([=]() {
          Buffer[0] = 0;
        });

        cgh.single_task<class fpga_kernel>(myKernel);
      });
      GetCLQueue(sycl_event, cl_queues);
    }

    int result = cl_queues.size();
    device dev = Queue.get_device();
    int expected_result = dev.is_host() ? 0 : getExpectedQueueNumber(dev.get(), maxNumQueues);

    if (expected_result != result) {
      std::cout << "Result Num of queues = " << result << std::endl
                << "Expected Num of queues = " << expected_result << std::endl;

      return -1;
    }
  }

  return 0;
}
