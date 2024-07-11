#pragma once
#include <cassert>
#include <climits>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <type_traits>

#if defined(__SPIR__) || defined(__SPIRV__)
typedef _Float16 _iml_half_internal;
#else
typedef uint16_t _iml_half_internal;
#endif

template <class Ty> class imf_utils_default_equ {
public:
  bool operator()(Ty x, Ty y) {
    if constexpr (std::is_same_v<Ty, sycl::half2>) {
      return (x.s0() == y.s0()) && (x.s1() == y.s1());
    } else
      return x == y;
  };
};

// Used to test half precision utils
template <class InputTy, class OutputTy, class FuncTy,
          class EquTy = imf_utils_default_equ<OutputTy>>
void test_host(std::initializer_list<InputTy> Input,
               std::initializer_list<OutputTy> RefOutput, FuncTy Func,
               int Line = __builtin_LINE()) {
  auto Size = Input.size();
  assert(RefOutput.size() == Size);

  for (int i = 0; i < Size; ++i) {
    auto Expected = *(std::begin(RefOutput) + i);
    auto Res = Func(*(std::begin(Input) + i));
    if (EquTy()(Expected, Res))
      continue;

    std::cout << "Mismatch at line " << Line << "[" << i << "]: " << Res
              << " != " << Expected << std::endl;
    assert(false);
  }
}

template <class InputTy, class OutputTy, class FuncTy,
          class EquTy = imf_utils_default_equ<OutputTy>>
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
    if constexpr (std::is_same_v<OutputTy, sycl::half2>) {
      if (EquTy()(Expected, Acc[i]))
        continue;
      std::cout << "Mismatch at line " << Line << "[" << i << "]: ("
                << Acc[i].s0() << ", " << Acc[i].s1() << ")"
                << " != (" << Expected.s0() << ", " << Expected.s1() << ")"
                << ", input was (" << (*(std::begin(Input) + i)).s0() << ", "
                << (*(std::begin(Input) + i)).s1() << ")" << std::endl;
    } else {
      if (EquTy()(Expected, Acc[i]))
        continue;
      std::cout << "Mismatch at line " << Line << "[" << i << "]: " << Acc[i]
                << " != " << Expected << ", input was "
                << *(std::begin(Input) + i) << std::endl;
    }
    assert(false);
  }
}

template <class InputTy, class OutputTy, class FuncTy,
          class EquTy = imf_utils_default_equ<OutputTy>>
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
    if constexpr (std::is_same_v<OutputTy, sycl::half2>) {
      if (EquTy()(Expected, Acc[i]))
        continue;
      std::cout << "Mismatch at line " << Line << "[" << i << "]: ("
                << Acc[i].s0() << ", " << Acc[i].s1() << ")"
                << " != (" << Expected.s0() << ", " << Expected.s1()
                << "), input idx was " << i << std::endl;
    } else {
      if (EquTy()(Expected, Acc[i]))
        continue;
      std::cout << "Mismatch at line " << Line << "[" << i << "]: " << Acc[i]
                << " != " << Expected << ", input idx was " << i << std::endl;
    }
    assert(false);
  }
}

template <class InputTy1, class InputTy2, class InputTy3, class OutputTy,
          class FuncTy, class EquTy = imf_utils_default_equ<OutputTy>>
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
    if constexpr (std::is_same_v<OutputTy, sycl::half2>) {
      if (EquTy()(Expected, Acc[i]))
        continue;
      std::cout << "Mismatch at line " << Line << "[" << i << "]: ("
                << Acc[i].s0() << ", " << Acc[i].s1() << ")"
                << " != (" << Expected.s0() << ", " << Expected.s1()
                << "), input idx was " << i << std::endl;
    } else {
      if (EquTy()(Expected, Acc[i]))
        continue;
      std::cout << "Mismatch at line " << Line << "[" << i << "]: " << Acc[i]
                << " != " << Expected << ", input was "
                << *(std::begin(Input1) + i) << ", "
                << *(std::begin(Input2) + i) << ", "
                << *(std::begin(Input3) + i) << std::endl;
    }
    assert(false);
  }
}

#define F(Name) [](auto x) { return (Name)(x); }
#define FT(T, Name) [](auto x) { return __builtin_bit_cast(T, (Name)(x)); }
#define FT1(T, Name) [](auto x) { return (Name)(__builtin_bit_cast(T, x)); }
#define FT2(T1, T2, Name)                                                      \
  [](auto x) {                                                                 \
    return __builtin_bit_cast(T1, (Name)(__builtin_bit_cast(T2, x)));          \
  }
#define F2(Name) [](auto x, auto y) { return (Name)(x, y); }
#define F2T(T, Name)                                                           \
  [](auto x, auto y) { return __builtin_bit_cast(T, (Name)(x, y)); }
#define F3(Name) [](auto x, auto y, auto z) { return (Name)(x, y, z); }
#define F3T(T, Name)                                                           \
  [](auto x, auto y, auto z) { return __builtin_bit_cast(T, (Name)(x, y, z)); }
#if defined(__SPIR__) || defined(__SPIRV__)
#define F_Half1(Name)                                                          \
  [](uint16_t x) { return (Name)(__builtin_bit_cast(_Float16, x)); }
#define F_Half2(Name)                                                          \
  [](auto x) { return __builtin_bit_cast(uint16_t, (Name)(x)); }
#define F_Half3(Name)                                                          \
  [](unsigned int x) {                                                         \
    return __builtin_bit_cast(uint16_t, (Name)(__builtin_bit_cast(float, x))); \
  }
#else
#define F_Half1(Name) [](uint16_t x) { return (Name)(x); }
#define F_Half2(Name) [](auto x) { return (Name)(x); }
#define F_Half3(Name)                                                          \
  [](unsigned int x) { return (Name)(__builtin_bit_cast(float, x)); }
#endif
