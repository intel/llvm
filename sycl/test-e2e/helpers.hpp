//==------------------- helpers.hpp -  test helpers ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/detail/core.hpp>

template <class VecT, int EndIdx = VecT::size(), int StartIdx = 0>
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
  inline void printHelper<EndIdx>(std::ostream &Out, const VecT &Elem1) {}

  VecT MVec;
};

template <class VecT, int EndIdx = VecT::size(), int StartIdx = 0>
VecPrinter<VecT, EndIdx, StartIdx> printableVec(const VecT &Vec) {
  return VecPrinter<VecT, EndIdx, StartIdx>(Vec);
}

template <class VecT, int EndIdx, int StartIdx>
std::ostream &operator<<(std::ostream &Out,
                         const VecPrinter<VecT, EndIdx, StartIdx> &VecP) {
  VecP.print(Out);
  return Out;
}

void exceptionHandlerHelper(sycl::exception_list ExceptionList) {
  for (std::exception_ptr ExceptionPtr : ExceptionList) {
    try {
      std::rethrow_exception(ExceptionPtr);
    } catch (sycl::exception &E) {
      std::cerr << E.what() << std::endl;
    }
  }
  abort();
}

class TestQueue : public sycl::queue {
public:
  TestQueue(const sycl::detail::DSelectorInvocableType &DevSelector,
            const sycl::property_list &PropList = {})
      : sycl::queue(DevSelector, exceptionHandlerHelper, PropList) {}

  ~TestQueue() { wait_and_throw(); }
};

namespace emu {

// std::exclusive_scan/inclusive_scan are not supported GCC 7.4,
// so use our own implementations
template <typename InputIterator, typename OutputIterator,
          class BinaryOperation, typename T>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, T init,
                              BinaryOperation binary_op) {
  T partial = init;
  for (InputIterator it = first; it != last; ++it) {
    *(result++) = partial;
    partial = binary_op(partial, *it);
  }
  return result;
}

template <typename InputIterator, typename OutputIterator,
          class BinaryOperation, typename T>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, BinaryOperation binary_op,
                              T init) {
  T partial = init;
  for (InputIterator it = first; it != last; ++it) {
    partial = binary_op(partial, *it);
    *(result++) = partial;
  }
  return result;
}
} // namespace emu

namespace env {

bool isDefined(const char *name) {
  char *buf = nullptr;
#ifdef _WIN32
  size_t sz;
  _dupenv_s(&buf, &sz, name);
  free(buf);
#else
  buf = getenv(name);
#endif
  return buf != nullptr;
}

std::string getVal(const char *name) {
  char *buf = nullptr;
  std::string res = "";
#ifdef _WIN32
  size_t sz;
  _dupenv_s(&buf, &sz, name);
  if (buf != nullptr)
    res = std::string(buf);
  free(buf);
#else
  buf = getenv(name);
  if (buf != nullptr)
    res = std::string(buf);
#endif
  return res;
}
} // namespace env
