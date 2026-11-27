// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks a kernel argument with a sycl::accessor base at a non-zero offset:
// __init must run on the accessor base subobject, not on the start of the
// derived object, which would leave the base accessor uninitialized and
// overwrite the preceding base.

#include <sycl/detail/core.hpp>

#include <array>
#include <cstddef>
#include <iostream>

using AccT = sycl::accessor<unsigned, 1, sycl::access_mode::read_write>;

// A, B and C are canaries around both accessors: initializing an accessor at
// the wrong address overwrites them.
struct Base {
  unsigned A;
  AccT AccField;
  unsigned B;
};

// The AccT base comes after Base, so it is at a non-zero offset.
struct Captured : Base, AccT {
  unsigned C;
};

int main() {
  // Distinct sizes, so each accessor's size() identifies its buffer.
  constexpr std::size_t FieldSize = 2, BaseSize = 3, OutSize = 5;
  constexpr unsigned ValA = 0x600D1111, ValB = 0x600D2222, ValC = 0x600D3333;
  // Every buffer element starts as Canary; each accessor writes its own
  // marker to its last element, so a write to the wrong buffer or element
  // shows up on the host.
  constexpr unsigned Canary = 0x600D600D;
  constexpr unsigned FieldMarker = 0xF1E1D000, BaseMarker = 0xBA5E0000;

  std::array<unsigned, FieldSize> FieldData;
  std::array<unsigned, BaseSize> BaseData;
  std::array<unsigned, OutSize> Out;
  FieldData.fill(Canary);
  BaseData.fill(Canary);
  Out.fill(0);

  sycl::queue Q;
  {
    sycl::buffer<unsigned, 1> FieldBuf(FieldData.data(),
                                       sycl::range<1>(FieldSize));
    sycl::buffer<unsigned, 1> BaseBuf(BaseData.data(),
                                      sycl::range<1>(BaseSize));
    sycl::buffer<unsigned, 1> OutBuf(Out.data(), sycl::range<1>(OutSize));

    Q.submit([&](sycl::handler &CGH) {
      Captured Obj{{ValA, AccT(FieldBuf, CGH), ValB}, AccT(BaseBuf, CGH), ValC};
      sycl::accessor OutAcc(OutBuf, CGH, sycl::write_only);
      CGH.single_task([=] {
        const AccT &BaseAcc = Obj;
        Obj.AccField[FieldSize - 1] = FieldMarker;
        BaseAcc[BaseSize - 1] = BaseMarker;
        OutAcc[0] = Obj.A;
        OutAcc[1] = Obj.B;
        OutAcc[2] = Obj.C;
        OutAcc[3] = Obj.AccField.size();
        OutAcc[4] = BaseAcc.size();
      });
    });
  }

  int Failures = 0;
  auto Check = [&](const char *What, unsigned Got, unsigned Expected) {
    if (Got != Expected) {
      std::cout << "FAIL: " << What << ": got 0x" << std::hex << Got
                << ", expected 0x" << Expected << std::dec << "\n";
      ++Failures;
    }
  };

  Check("A", Out[0], ValA);
  Check("B", Out[1], ValB);
  Check("C", Out[2], ValC);
  Check("AccField.size()", Out[3], FieldSize);
  Check("base accessor size()", Out[4], BaseSize);
  Check("AccField data[0]", FieldData[0], Canary);
  Check("AccField data[1]", FieldData[1], FieldMarker);
  Check("base accessor data[0]", BaseData[0], Canary);
  Check("base accessor data[1]", BaseData[1], Canary);
  Check("base accessor data[2]", BaseData[2], BaseMarker);

  if (Failures == 0)
    std::cout << "Passed\n";
  return Failures;
}
