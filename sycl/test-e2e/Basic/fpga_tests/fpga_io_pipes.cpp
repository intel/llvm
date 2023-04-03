// RUN: %clangxx -fsycl %s -o %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//==------------ fpga_io_pipes.cpp - SYCL FPGA pipes test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "io_pipe_def.h"

// TODO: run is disabled, since no support added in FPGA backend yet. Check
// implementation correctness from CXX and SYCL languages perspective.

// This test is supposed to be run only on Intel FPGA emulator. Change it when
// we have more experience with IO pipe feature in SYCL.
// The emulator creates files (one for I pipe, another for O pipe) with the
// appropriate naming, where a data flowing through a pipe can be stored.
// So in the test we need to create these files and use them appropriately.
// The name is taken as IO pipe ID.
const size_t InputData = 42;
const std::string InputFileName = "0.txt";
const std::string OutputFileName = "1.txt";

void createInputFile(const std::string &filename) {
  std::ofstream Input(filename);
  if (Input.is_open()) {
    Input << InputData;
    Input.close();
  }
}

int validateOutputFile(const std::string &filename) {
  std::ifstream Output(filename);
  std::string Line;
  std::vector<size_t> Result;
  if (Output.is_open()) {
    // In the test we write only one number into the pipe, but a backend might
    // have a bug of incorrect interpretetion of capacity of the pipe. In this
    // case let's read all the lines of the output file to catch this.
    while (std::getline(Output, Line))
      Result.push_back(stoi(Line));
    Output.close();
  }
  if (Result.size() != 1 || Result[0] != InputData) {
    std::cout << "Result mismatches " << Result[0] << " Vs expected "
              << InputData << std::endl;
    return -1;
  }

  return 0;
}

// Test for simple non-blocking pipes
int test_io_nb_pipe(sycl::queue Queue) {
  createInputFile(InputFileName);

  sycl::buffer<int, 1> writeBuf(1);
  Queue.submit([&](sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<sycl::access::mode::write>(cgh);

    cgh.single_task<class nb_io_transfer>([=]() {
      bool SuccessCodeI = false;
      do {
        write_acc[0] = intelfpga::ethernet_read_pipe::read(SuccessCodeI);
      } while (!SuccessCodeI);
      bool SuccessCodeO = false;
      do {
        intelfpga::ethernet_write_pipe::write(write_acc[0], SuccessCodeO);
      } while (!SuccessCodeO);
    });
  });

  auto readHostBuffer = writeBuf.get_access<sycl::access::mode::read>();
  if (readHostBuffer[0] != InputData) {
    std::cout << "Read from a file mismatches " << readHostBuffer[0]
              << " Vs expected " << InputData << std::endl;

    return -1;
  }

  return validateOutputFile(OutputFileName);
}

// Test for simple blocking pipes
int test_io_bl_pipe(sycl::queue Queue) {
  createInputFile(InputFileName);

  sycl::buffer<int, 1> writeBuf(1);
  Queue.submit([&](sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<sycl::access::mode::write>(cgh);

    cgh.single_task<class bl_io_transfer>([=]() {
      write_acc[0] = intelfpga::ethernet_read_pipe::read();
      intelfpga::ethernet_write_pipe::write(write_acc[0]);
    });
  });

  auto readHostBuffer = writeBuf.get_access<sycl::access::mode::read>();
  if (readHostBuffer[0] != InputData) {
    std::cout << "Read from a file mismatches " << readHostBuffer[0]
              << " Vs expected " << InputData << std::endl;

    return -1;
  }

  return validateOutputFile(OutputFileName);
}

int main() {
  sycl::queue Queue{sycl::ext::intel::fpga_emulator_selector{}};

  if (!Queue.get_device()
           .get_info<sycl::info::device::kernel_kernel_pipe_support>()) {
    std::cout << "SYCL_ext_intel_data_flow_pipes not supported, skipping"
              << std::endl;
    return 0;
  }

  // Non-blocking pipes
  int Result = test_io_nb_pipe(Queue);

  // Blocking pipes
  Result &= test_io_bl_pipe(Queue);

  return Result;
}
