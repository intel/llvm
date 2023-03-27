// RUN: %clangxx -fsycl %s -o %t.out
//-fsycl-targets=%sycl_triple
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==------------- fpga_pipes.cpp - SYCL FPGA pipes test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

// Size of an array passing through a pipe
constexpr size_t N = 10;

// For simple non-blocking pipes with explicit type
class some_nb_pipe;

// For non-blocking pipes created with namespaces set
namespace some {
class nb_pipe;
}

// For non-blocking template pipes
template <int N> class templ_nb_pipe;

// For non-blocking multiple pipes
template <int N>
using PipeMulNb = sycl::ext::intel::pipe<class templ_nb_pipe<N>, int>;
static_assert(std::is_same_v<typename PipeMulNb<0>::value_type, int>);
static_assert(PipeMulNb<0>::min_capacity == 0);

// For simple blocking pipes with explicit type
class some_bl_pipe;

// For blocking pipes created with namespaces set
namespace some {
class bl_pipe;
}

// For blocking template pipes
template <int N> class templ_bl_pipe;

// For blocking multiple pipes
template <int N>
using PipeMulBl = sycl::ext::intel::pipe<class templ_bl_pipe<N>, int>;
static_assert(std::is_same_v<typename PipeMulBl<0>::value_type, int>);
static_assert(PipeMulBl<0>::min_capacity == 0);

// Kernel names
template <int TestNumber, int KernelNumber = 0> class writer;
template <int TestNumber, int KernelNumber = 0> class reader;

// Test for simple non-blocking pipes
template <typename PipeName, int TestNumber>
int test_simple_nb_pipe(sycl::queue Queue) {
  int data[] = {0};

  using Pipe = sycl::ext::intel::pipe<PipeName, int>;
  static_assert(std::is_same_v<typename Pipe::value_type, int>);
  static_assert(Pipe::min_capacity == 0);

  sycl::buffer<int, 1> readBuf(data, 1);
  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class writer<TestNumber>>([=]() {
      bool SuccessCode = false;
      do {
        Pipe::write(42, SuccessCode);
      } while (!SuccessCode);
    });
  });

  sycl::buffer<int, 1> writeBuf(data, 1);
  Queue.submit([&](sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<sycl::access::mode::write>(cgh);

    cgh.single_task<class reader<TestNumber>>([=]() {
      bool SuccessCode = false;
      do {
        write_acc[0] = Pipe::read(SuccessCode);
      } while (!SuccessCode);
    });
  });

  auto readHostBuffer = writeBuf.get_access<sycl::access::mode::read>();
  if (readHostBuffer[0] != 42) {
    std::cout << "Test: " << TestNumber << "\nResult mismatches "
              << readHostBuffer[0] << " Vs expected " << 42 << std::endl;

    return -1;
  }

  return 0;
}

// Test for multiple non-blocking pipes
template <int TestNumber> int test_multiple_nb_pipe(sycl::queue Queue) {
  int data[] = {0};

  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class writer<TestNumber, /*KernelNumber*/ 1>>([=]() {
      bool SuccessCode = false;
      do {
        PipeMulNb<1>::write(19, SuccessCode);
      } while (!SuccessCode);
    });
  });

  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class writer<TestNumber, /*KernelNumber*/ 2>>([=]() {
      bool SuccessCode = false;
      do {
        PipeMulNb<2>::write(23, SuccessCode);
      } while (!SuccessCode);
    });
  });

  sycl::buffer<int, 1> writeBuf(data, 1);
  Queue.submit([&](sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<class reader<TestNumber>>([=]() {
      bool SuccessCodeA = false;
      int Value = 0;
      do {
        Value = PipeMulNb<1>::read(SuccessCodeA);
      } while (!SuccessCodeA);
      write_acc[0] = Value;
      bool SuccessCodeB = false;
      do {
        Value = PipeMulNb<2>::read(SuccessCodeB);
      } while (!SuccessCodeB);
      write_acc[0] += Value;
    });
  });

  auto readHostBuffer = writeBuf.get_access<sycl::access::mode::read>();
  if (readHostBuffer[0] != 42) {
    std::cout << "Test: " << TestNumber << "\nResult mismatches "
              << readHostBuffer[0] << " Vs expected " << 42 << std::endl;

    return -1;
  }

  return 0;
}

// Test for array passing through a non-blocking pipe
template <int TestNumber> int test_array_th_nb_pipe(sycl::queue Queue) {
  int data[N] = {0};
  using AnotherNbPipe = sycl::ext::intel::pipe<class another_nb_pipe, int>;
  static_assert(std::is_same_v<typename AnotherNbPipe::value_type, int>);
  static_assert(AnotherNbPipe::min_capacity == 0);

  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class writer<TestNumber>>([=]() {
      bool SuccessCode = false;
      for (size_t i = 0; i != N; ++i) {
        do {
          AnotherNbPipe::write(i, SuccessCode);
        } while (!SuccessCode);
      }
    });
  });

  sycl::buffer<int, 1> writeBuf(data, N);
  Queue.submit([&](sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<class reader<TestNumber>>([=]() {
      for (size_t i = 0; i != N; ++i) {
        bool SuccessCode = false;
        do {
          write_acc[i] = AnotherNbPipe::read(SuccessCode);
        } while (!SuccessCode);
      }
    });
  });

  auto readHostBuffer = writeBuf.get_access<sycl::access::mode::read>();
  for (size_t i = 0; i != N; ++i) {
    if (readHostBuffer[i] != i)
      std::cout << "Test: " << TestNumber << "\nResult mismatches "
                << readHostBuffer[i] << " Vs expected " << i << std::endl;
    return -1;
  }

  return 0;
}

// Test for simple blocking pipes
template <typename PipeName, int TestNumber>
int test_simple_bl_pipe(sycl::queue Queue) {
  int data[] = {0};

  using Pipe = sycl::ext::intel::pipe<PipeName, int>;
  static_assert(std::is_same_v<typename Pipe::value_type, int>);
  static_assert(Pipe::min_capacity == 0);

  sycl::buffer<int, 1> readBuf(data, 1);
  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class writer<TestNumber>>([=]() {
      Pipe::write(42);
    });
  });

  sycl::buffer<int, 1> writeBuf(data, 1);
  Queue.submit([&](sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<sycl::access::mode::write>(cgh);

    cgh.single_task<class reader<TestNumber>>([=]() {
      write_acc[0] = Pipe::read();
    });
  });

  auto readHostBuffer = writeBuf.get_access<sycl::access::mode::read>();
  if (readHostBuffer[0] != 42) {
    std::cout << "Test: " << TestNumber << "\nResult mismatches "
              << readHostBuffer[0] << " Vs expected " << 42 << std::endl;

    return -1;
  }

  return 0;
}

// Test for multiple blocking pipes
template <int TestNumber> int test_multiple_bl_pipe(sycl::queue Queue) {
  int data[] = {0};

  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class writer<TestNumber, /*KernelNumber*/ 1>>([=]() {
      PipeMulBl<1>::write(19);
    });
  });

  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class writer<TestNumber, /*KernelNumber*/ 2>>([=]() {
      PipeMulBl<2>::write(23);
    });
  });

  sycl::buffer<int, 1> writeBuf(data, 1);
  Queue.submit([&](sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<class reader<TestNumber>>([=]() {
      write_acc[0] = PipeMulBl<1>::read();
      write_acc[0] += PipeMulBl<2>::read();
    });
  });

  auto readHostBuffer = writeBuf.get_access<sycl::access::mode::read>();
  if (readHostBuffer[0] != 42) {
    std::cout << "Test: " << TestNumber << "\nResult mismatches "
              << readHostBuffer[0] << " Vs expected " << 42 << std::endl;

    return -1;
  }

  return 0;
}

// Test for array passing through a blocking pipe
template <int TestNumber> int test_array_th_bl_pipe(sycl::queue Queue) {
  int data[N] = {0};
  using AnotherBlPipe = sycl::ext::intel::pipe<class another_bl_pipe, int>;
  static_assert(std::is_same_v<typename AnotherBlPipe::value_type, int>);
  static_assert(AnotherBlPipe::min_capacity == 0);

  Queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<class writer<TestNumber>>([=]() {
      for (size_t i = 0; i != N; ++i)
        AnotherBlPipe::write(i);
    });
  });

  sycl::buffer<int, 1> writeBuf(data, N);
  Queue.submit([&](sycl::handler &cgh) {
    auto write_acc = writeBuf.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<class reader<TestNumber>>([=]() {
      for (size_t i = 0; i != N; ++i)
        write_acc[i] = AnotherBlPipe::read();
    });
  });

  auto readHostBuffer = writeBuf.get_access<sycl::access::mode::read>();
  for (size_t i = 0; i != N; ++i) {
    if (readHostBuffer[i] != i)
      std::cout << "Test: " << TestNumber << "\nResult mismatches "
                << readHostBuffer[i] << " Vs expected " << i << std::endl;
    return -1;
  }

  return 0;
}

int main() {
  sycl::queue Queue;

  if (!Queue.get_device()
           .get_info<sycl::info::device::kernel_kernel_pipe_support>()) {
    std::cout << "SYCL_ext_intel_data_flow_pipes not supported, skipping"
              << std::endl;
    return 0;
  }

  // Non-blocking pipes
  int Result = test_simple_nb_pipe<some_nb_pipe, /*test number*/ 1>(Queue);
  Result &= test_simple_nb_pipe<some::nb_pipe, /*test number*/ 2>(Queue);
  class forward_nb_pipe;
  Result &= test_simple_nb_pipe<forward_nb_pipe, /*test number*/ 3>(Queue);
  Result &= test_simple_nb_pipe<templ_nb_pipe<0>, /*test number*/ 4>(Queue);
  Result &= test_multiple_nb_pipe</*test number*/ 5>(Queue);

  // Blocking pipes
  Result &= test_simple_bl_pipe<some_bl_pipe, /*test number*/ 6>(Queue);
  Result &= test_simple_bl_pipe<some::bl_pipe, /*test number*/ 7>(Queue);
  class forward_bl_pipe;
  Result &= test_simple_bl_pipe<forward_bl_pipe, /*test number*/ 8>(Queue);
  Result &= test_simple_bl_pipe<templ_bl_pipe<0>, /*test number*/ 9>(Queue);
  Result &= test_multiple_bl_pipe</*test number*/ 10>(Queue);

  // Test for an array data passing through a pipe
  Result &= test_array_th_nb_pipe</*test number*/ 11>(Queue);
  Result &= test_array_th_bl_pipe</*test number*/ 12>(Queue);

  return Result;
}
