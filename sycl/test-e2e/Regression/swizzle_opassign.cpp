// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{  %{build} -fpreview-breaking-changes -o %t2.out   %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out  %}

// Tests that the mutating operators (+=, -=, ..., ++, --) on swizzles compile
// and correctly mutate the elements in the corresponding vector.

#include <sycl/detail/core.hpp>
#include <sycl/types.hpp>
#include <sycl/usm.hpp>

constexpr std::string_view OpNames[] = {
    "+=", "-=",  "*=",  "/=",        "%=",        "&=",         "|=",
    "^=", "<<=", ">>=", "prefix ++", "prefix --", "postfix ++", "prefix ++"};
constexpr size_t NumOps = std::size(OpNames);

int main() {
  sycl::queue Q;
  bool Results[NumOps] = {false};

  {
    sycl::buffer<bool> ResultsBuff{Results, NumOps};

    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor ResultsAcc{ResultsBuff, CGH, sycl::write_only};

      CGH.single_task([=]() {
        int I = 0;
#define TestCase(OP)                                                           \
  {                                                                            \
    sycl::vec<int, 4> VecVal{1, 2, 3, 4};                                      \
    int ExpectedRes = VecVal[1] OP 2;                                          \
    ResultsAcc[I++] = (VecVal.swizzle<1>() OP## = 2)[0] == ExpectedRes &&      \
                      VecVal[1] == ExpectedRes;                                \
  }
        TestCase(+);
        TestCase(-);
        TestCase(*);
        TestCase(/);
        TestCase(%);
        TestCase(&);
        TestCase(|);
        TestCase(^);
        TestCase(<<);
        TestCase(>>);
        {
          sycl::vec<int, 4> VecVal{1, 2, 3, 4};
          int ExpectedRes = VecVal[1] + 1;
          ResultsAcc[I++] = (++VecVal.swizzle<1>())[0] == ExpectedRes &&
                            VecVal[1] == ExpectedRes;
        }
        {
          sycl::vec<int, 4> VecVal{1, 2, 3, 4};
          int ExpectedRes = VecVal[1] - 1;
          ResultsAcc[I++] = (--VecVal.swizzle<1>())[0] == ExpectedRes &&
                            VecVal[1] == ExpectedRes;
        }
        {
          sycl::vec<int, 4> VecVal{1, 2, 3, 4};
          int ExpectedRes = VecVal[1] + 1;
          ResultsAcc[I++] = (VecVal.swizzle<1>()++)[0] == (ExpectedRes - 1) &&
                            VecVal[1] == ExpectedRes;
        }
        {
          sycl::vec<int, 4> VecVal{1, 2, 3, 4};
          int ExpectedRes = VecVal[1] - 1;
          ResultsAcc[I++] = (VecVal.swizzle<1>()--)[0] == (ExpectedRes + 1) &&
                            VecVal[1] == ExpectedRes;
        }
      });
    });
  }

  int Failures = 0;
  for (size_t I = 0; I < NumOps; ++I) {
    if (!Results[I]) {
      std::cout << "Failed for " << OpNames[I] << std::endl;
      ++Failures;
    }
  }

  return Failures;
}
