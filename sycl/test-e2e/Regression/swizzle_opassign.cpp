// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{  %{build} -fpreview-breaking-changes -o %t2.out   %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out  %}

// Tests that the mutating operators (+=, -=, ..., ++, --) on swizzles compile
// and correctly mutate the elements in the corresponding vector.

#include <sycl/sycl.hpp>

constexpr size_t NumOps = 14;
constexpr std::string_view OpNames[NumOps] = {
    "+=", "-=",  "*=",  "/=",        "%=",        "&=",         "|=",
    "^=", "<<=", ">>=", "prefix ++", "prefix --", "postfix ++", "prefix ++"};

int main() {
  sycl::queue Q;
  bool *Results = sycl::malloc_shared<bool>(NumOps, Q);
  for (size_t I = 0; I < NumOps; ++I)
    Results[I] = 0;

  Q.single_task([=]() {
     bool *ResultIt = Results;
#define TestCase(OP)                                                           \
  {                                                                            \
    sycl::vec<int, 4> VecVal{1, 2, 3, 4};                                      \
    int ExpectedRes = VecVal[1] OP 2;                                          \
    *(ResultIt++) = (VecVal.swizzle<1>() OP## = 2)[0] == ExpectedRes &&        \
                    VecVal[1] == ExpectedRes;                                  \
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
       *(ResultIt++) = (++VecVal.swizzle<1>())[0] == ExpectedRes &&
                       VecVal[1] == ExpectedRes;
     }
     {
       sycl::vec<int, 4> VecVal{1, 2, 3, 4};
       int ExpectedRes = VecVal[1] - 1;
       *(ResultIt++) = (--VecVal.swizzle<1>())[0] == ExpectedRes &&
                       VecVal[1] == ExpectedRes;
     }
     {
       sycl::vec<int, 4> VecVal{1, 2, 3, 4};
       int ExpectedRes = VecVal[1] + 1;
       *(ResultIt++) = (VecVal.swizzle<1>()++)[0] == (ExpectedRes - 1) &&
                       VecVal[1] == ExpectedRes;
     }
     {
       sycl::vec<int, 4> VecVal{1, 2, 3, 4};
       int ExpectedRes = VecVal[1] - 1;
       *(ResultIt++) = (VecVal.swizzle<1>()--)[0] == (ExpectedRes + 1) &&
                       VecVal[1] == ExpectedRes;
     }
   }).wait_and_throw();

  int Failures = 0;
  for (size_t I = 0; I < NumOps; ++I) {
    if (!Results[I]) {
      std::cout << "Failed for " << OpNames[I] << std::endl;
      ++Failures;
    }
  }

  sycl::free(Results, Q);
  return Failures;
}
