//==----- multiple-devices.cpp - Appropriate AOT-compiled image selection  ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

// REQUIRES: ioc64, ocloc, aoc

// 1-command compilation case
// Targeting CPU, GPU, FPGA
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-linux-sycldevice,spir64_gen-unknown-linux-sycldevice,spir64_fpga-unknown-linux-sycldevice -Xsycl-target-backend=spir64_gen-unknown-linux-sycldevice "-device skl" %s -o %t_all.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t_all.out
// RUN: %CPU_RUN_PLACEHOLDER %t_all.out
// RUN: %GPU_RUN_PLACEHOLDER %t_all.out
// RUN: %ACC_RUN_PLACEHOLDER %t_all.out

// Produce object file, spirv, device images to combine these differently
// at link-time, thus testing various AOT-compiled images configurations
// RUN: %clangxx -fsycl %s -c -o %t.o
// RUN: %clangxx -fsycl -fsycl-link-targets=spir64-unknown-linux-sycldevice %t.o -o %t.spv
// AOT-compile device binary images
// RUN: ioc64 -cmd=build -binary=%t.spv -ir=%t_cpu.ir -device=cpu
// RUN: ocloc -file %t.spv -spirv_input -output %t_gen.out -output_no_suffix -device cfl
// RUN: aoc %t.spv -o %t_fpga.aocx -sycl -dep-files=%t.d

// CPU, GPU
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64_x86_64:%t_cpu.ir,spir64_gen:%t_gen.out %t.o -o %t_cpu_gpu.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t_cpu_gpu.out
// RUN: %CPU_RUN_PLACEHOLDER %t_cpu_gpu.out
// RUN: %GPU_RUN_PLACEHOLDER %t_cpu_gpu.out

// CPU, FPGA
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64_x86_64:%t_cpu.ir,spir64_fpga:%t_fpga.aocx %t.o -o %t_cpu_fpga.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t_cpu_fpga.out
// RUN: %CPU_RUN_PLACEHOLDER %t_cpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_cpu_fpga.out

// GPU, FPGA
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64_gen:%t_gen.out,spir64_fpga:%t_fpga.aocx %t.o -o %t_gpu_fpga.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t_gpu_fpga.out
// RUN: %GPU_RUN_PLACEHOLDER %t_gpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_gpu_fpga.out

// No AOT-compiled image for CPU
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64:%t.spv,spir64_gen:%t_gen.out,spir64_fpga:%t_fpga.aocx %t.o -o %t_spv_gpu_fpga.out
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_gpu_fpga.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_gpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_gpu_fpga.out

// No AOT-compiled image for GPU
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64:%t.spv,spir64_x86_64:%t_cpu.ir,spir64_fpga:%t_fpga.aocx %t.o -o %t_spv_cpu_fpga.out
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_cpu_fpga.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_cpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_cpu_fpga.out

// No AOT-compiled image for FPGA
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64:%t.spv,spir64_x86_64:%t_cpu.ir,spir64_gen:%t_gen.out %t.o -o %t_spv_cpu_gpu.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_cpu_gpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_cpu_gpu.out
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_cpu_gpu.out

#include <CL/sycl.hpp>

#include <array>
#include <iostream>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC) {
  cl::sycl::queue deviceQueue([](cl::sycl::exception_list ExceptionList) {
      for (cl::sycl::exception_ptr_class ExceptionPtr : ExceptionList) {
        try {
          std::rethrow_exception(ExceptionPtr);
        } catch (cl::sycl::exception &E) {
          std::cerr << E.what();
        } catch (...) {
          std::cerr << "Unknown async exception was caught." << std::endl;
        }
      }
    });

  cl::sycl::range<1> numOfItems{N};
  cl::sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  cl::sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  cl::sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class SimpleVadd<T>>(numOfItems,
    [=](cl::sycl::id<1> wiID) {
        accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    });
  });

  deviceQueue.wait_and_throw();
}

int main() {
  const size_t array_size = 4;
  std::array<cl::sycl::cl_int, array_size> A = {{1, 2, 3, 4}},
                                           B = {{1, 2, 3, 4}}, C;
  std::array<cl::sycl::cl_float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                             E = {{1.f, 2.f, 3.f, 4.f}}, F;
  simple_vadd(A, B, C);
  simple_vadd(D, E, F);
  for (unsigned int i = 0; i < array_size; i++) {
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
    if (F[i] != D[i] + E[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << F[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}
