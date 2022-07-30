//==------------------- helpers.hpp -  test helpers ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

template <class VecT, int EndIdx = VecT::get_count(), int StartIdx = 0>
class VecPrinter {
public:
  VecPrinter(const VecT &Vec) : MVec(Vec) {}

  void print(std::ostream &Out) const {
    std::cout << "[ ";
    printHelper<StartIdx>(Out, MVec);
    std::cout << " ]";
  }

  static void print(const VecT &Elem1) {
    std::cout << "[ ";
    printHelper<StartIdx>(std::cout, Elem1);
    std::cout << " ]";
  }

private:
  template <int Idx>
  static void printHelper(std::ostream &Out, const VecT &Elem1) {
    std::cout << (typename VecT::element_type)(Elem1.template swizzle<Idx>());
    if (Idx + 1 != EndIdx)
      std::cout << ", ";
    printHelper<Idx + 1>(Out, Elem1);
  }
  template <>
  static void printHelper<EndIdx>(std::ostream &Out, const VecT &Elem1) {}

  VecT MVec;
};

template <class VecT, int EndIdx = VecT::get_count(), int StartIdx = 0>
VecPrinter<VecT, EndIdx, StartIdx> printableVec(const VecT &Vec) {
  return VecPrinter<VecT, EndIdx, StartIdx>(Vec);
}

template <class VecT, int EndIdx, int StartIdx>
std::ostream &operator<<(std::ostream &Out,
                         const VecPrinter<VecT, EndIdx, StartIdx> &VecP) {
  VecP.print(Out);
  return Out;
}

class TestQueue : public sycl::queue {
public:
  TestQueue(const sycl::device_selector &DevSelector,
            const sycl::property_list &PropList = {})
      : sycl::queue(
            DevSelector,
            [](sycl::exception_list ExceptionList) {
              for (std::exception_ptr ExceptionPtr : ExceptionList) {
                try {
                  std::rethrow_exception(ExceptionPtr);
                } catch (sycl::exception &E) {
                  std::cerr << E.what() << std::endl;
                }
              }
              abort();
            },
            PropList) {}

  ~TestQueue() { wait_and_throw(); }
};
