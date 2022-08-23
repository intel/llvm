#pragma once
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <sycl/sycl.hpp>

template <class InputTy, class OutputTy, class FuncTy>
void test(sycl::queue &q, std::initializer_list<InputTy> Input,
          std::initializer_list<OutputTy> RefOutput, FuncTy Func,
          int Line = __builtin_LINE()) {
  auto Size = Input.size();
  assert(RefOutput.size() == Size);

  sycl::buffer<InputTy> InBuf(Size);
  {
    sycl::host_accessor InAcc(InBuf, sycl::write_only);
    int i = 0;
    for (auto x : Input)
      InAcc[i++] = x;
  }

  sycl::buffer<OutputTy> OutBuf(Size);
  q.submit([&](sycl::handler &CGH) {
     sycl::accessor InAcc(InBuf, CGH, sycl::read_only);
     sycl::accessor OutAcc(OutBuf, CGH, sycl::write_only);
     CGH.parallel_for(Size,
                      [=](sycl::id<1> Id) { OutAcc[Id] = Func(InAcc[Id]); });
   }).wait();

  sycl::host_accessor Acc(OutBuf, sycl::read_only);
  for (int i = 0; i < Size; ++i) {
    auto Expected = *(std::begin(RefOutput) + i);
    if (Expected == Acc[i])
      continue;

    std::cout << "Mismatch at line " << Line << "[" << i << "]: " << Acc[i]
              << " != " << Expected << ", input was "
              << *(std::begin(Input) + i) << std::endl;
    assert(false);
  }
}

template <class InputTy, class OutputTy, class FuncTy>
void test2(sycl::queue &q, std::initializer_list<InputTy> Input1,
           std::initializer_list<InputTy> Input2,
           std::initializer_list<OutputTy> RefOutput, FuncTy Func,
           int Line = __builtin_LINE()) {
  auto Size = Input1.size();
  assert(Size == Input2.size());
  assert(RefOutput.size() == Size);

  sycl::buffer<InputTy> InBuf1(Size);
  sycl::buffer<InputTy> InBuf2(Size);
  {
    sycl::host_accessor InAcc1(InBuf1, sycl::write_only);
    sycl::host_accessor InAcc2(InBuf2, sycl::write_only);
    for (int i = 0; i < Size; ++i) {
      InAcc1[i] = *(std::begin(Input1) + i);
      InAcc2[i] = *(std::begin(Input2) + i);
    }
  }

  sycl::buffer<OutputTy> OutBuf(Size);
  q.submit([&](sycl::handler &CGH) {
     sycl::accessor InAcc1(InBuf1, CGH, sycl::read_only);
     sycl::accessor InAcc2(InBuf2, CGH, sycl::read_only);
     sycl::accessor OutAcc(OutBuf, CGH, sycl::write_only);
     CGH.parallel_for(Size, [=](sycl::id<1> Id) {
       OutAcc[Id] = Func(InAcc1[Id], InAcc2[Id]);
     });
   }).wait();

  sycl::host_accessor Acc(OutBuf, sycl::read_only);
  for (int i = 0; i < Size; ++i) {
    auto Expected = *(std::begin(RefOutput) + i);
    if (Expected == Acc[i])
      continue;

    std::cout << "Mismatch at line " << Line << "[" << i << "]: " << Acc[i]
              << " != " << Expected << ", input was "
              << *(std::begin(Input1) + i) << ", " << *(std::begin(Input2) + i)
              << std::endl;
    assert(false);
  }
}

template <class InputTy1, class InputTy2, class InputTy3, class OutputTy,
          class FuncTy>
void test3(sycl::queue &q, std::initializer_list<InputTy1> Input1,
           std::initializer_list<InputTy2> Input2,
           std::initializer_list<InputTy3> Input3,
           std::initializer_list<OutputTy> RefOutput, FuncTy Func,
           int Line = __builtin_LINE()) {
  auto Size = Input1.size();
  assert((Size == Input2.size()) && (Size == Input3.size()));
  assert(RefOutput.size() == Size);

  sycl::buffer<InputTy1> InBuf1(Size);
  sycl::buffer<InputTy2> InBuf2(Size);
  sycl::buffer<InputTy3> InBuf3(Size);
  {
    sycl::host_accessor InAcc1(InBuf1, sycl::write_only);
    sycl::host_accessor InAcc2(InBuf2, sycl::write_only);
    sycl::host_accessor InAcc3(InBuf3, sycl::write_only);
    for (int i = 0; i < Size; ++i) {
      InAcc1[i] = *(std::begin(Input1) + i);
      InAcc2[i] = *(std::begin(Input2) + i);
      InAcc3[i] = *(std::begin(Input3) + i);
    }
  }

  sycl::buffer<OutputTy> OutBuf(Size);
  q.submit([&](sycl::handler &CGH) {
     sycl::accessor InAcc1(InBuf1, CGH, sycl::read_only);
     sycl::accessor InAcc2(InBuf2, CGH, sycl::read_only);
     sycl::accessor InAcc3(InBuf3, CGH, sycl::read_only);
     sycl::accessor OutAcc(OutBuf, CGH, sycl::write_only);
     CGH.parallel_for(Size, [=](sycl::id<1> Id) {
       OutAcc[Id] = Func(InAcc1[Id], InAcc2[Id], InAcc3[Id]);
     });
   }).wait();

  sycl::host_accessor Acc(OutBuf, sycl::read_only);
  for (int i = 0; i < Size; ++i) {
    auto Expected = *(std::begin(RefOutput) + i);
    if (Expected == Acc[i])
      continue;

    std::cout << "Mismatch at line " << Line << "[" << i << "]: " << Acc[i]
              << " != " << Expected << ", input was "
              << *(std::begin(Input1) + i) << ", " << *(std::begin(Input2) + i)
              << ", " << *(std::begin(Input3) + i) << std::endl;
    assert(false);
  }
}

#define F(Name) [](auto x) { return (Name)(x); }
#define FT(T, Name) [](auto x) { return __builtin_bit_cast(T, (Name)(x)); }
#define F2(Name) [](auto x, auto y) { return (Name)(x, y); }
#define F3(Name) [](auto x, auto y, auto z) { return (Name)(x, y, z); }
