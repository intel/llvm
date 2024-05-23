// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>

template <typename T>
int CheckResult(const T &Actual, const T &Reference, const char *Category,
                float AllowedErr = 0.0001f) {
  if (sycl::fabs(static_cast<float>(Actual) - static_cast<float>(Reference)) <=
      AllowedErr)
    return 0;

  std::cout << "Failed: " << Category << " - " << Actual << " != " << Reference
            << std::endl;
  return 1;
}

template <typename T, int N>
int CheckResult(const sycl::vec<T, N> &Actual, const sycl::vec<T, N> &Reference,
                const char *Category, float AllowedErr = 0.0001f) {
  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    if (sycl::fabs(static_cast<float>(Actual[I]) -
                   static_cast<float>(Reference[I])) <= AllowedErr)
      continue;

    std::cout << "Failed at index " << I << ": " << Category << " - "
              << Actual[I] << " != " << Reference[I] << std::endl;
    ++Failures;
  }
  return Failures;
}

#define CHECK(Func, NonPtrT, PtrElemT, ...)                                    \
  {                                                                            \
    PtrElemT Y = PtrElemT{0};                                                  \
    NonPtrT R = sycl::Func(__VA_ARGS__, &Y);                                   \
    PtrElemT Ys[3] = {PtrElemT{0}};                                            \
    NonPtrT Rs[3] = {NonPtrT{0.0f}};                                           \
    {                                                                          \
      sycl::buffer<PtrElemT, 1> YsBuff{Ys, 3};                                 \
      sycl::buffer<NonPtrT, 1> RsBuff{Rs, 3};                                  \
      Q.submit([&](sycl::handler &CGH) {                                       \
        sycl::accessor YsAcc{YsBuff, CGH, sycl::read_write};                   \
        sycl::accessor RsAcc{RsBuff, CGH, sycl::write_only};                   \
        sycl::local_accessor<PtrElemT, 0> LocalAcc{CGH};                       \
        CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {      \
          LocalAcc = PtrElemT{0};                                              \
          PtrElemT PrivateVal = PtrElemT{0};                                   \
          /* Global mem raw ptr */                                             \
          PtrElemT *RawGlobalPtr =                                             \
              YsAcc.get_multi_ptr<sycl::access::decorated::no>().get_raw();    \
          RsAcc[0] = sycl::Func(__VA_ARGS__, RawGlobalPtr);                    \
          /* Local mem raw ptr */                                              \
          PtrElemT *RawLocalPtr =                                              \
              LocalAcc.get_multi_ptr<sycl::access::decorated::no>().get_raw(); \
          RsAcc[1] = sycl::Func(__VA_ARGS__, RawLocalPtr);                     \
          YsAcc[1] = LocalAcc;                                                 \
          /* Private mem raw ptr */                                            \
          PtrElemT *RawPrivatePtr = &PrivateVal;                               \
          RsAcc[2] = sycl::Func(__VA_ARGS__, RawPrivatePtr);                   \
          YsAcc[2] = PrivateVal;                                               \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    Failed += CheckResult(                                                     \
        Y, Ys[0], "pointer return from " #Func " with global pointer");        \
    Failed += CheckResult(R, Rs[0],                                            \
                          "return value from " #Func " with global pointer");  \
    Failed += CheckResult(Y, Ys[1],                                            \
                          "pointer return from " #Func " with local pointer"); \
    Failed += CheckResult(R, Rs[1],                                            \
                          "return value from " #Func " with local pointer");   \
    Failed += CheckResult(                                                     \
        Y, Ys[2], "pointer return from " #Func " with private pointer");       \
    Failed += CheckResult(R, Rs[2],                                            \
                          "return value from " #Func " with private pointer"); \
  }

int main() {
  sycl::queue Q;

  int Failed = 0;

  CHECK(fract, float, float, 1.5f)
  CHECK(frexp, float, int, 1.5f)
  CHECK(lgamma_r, float, int, 1.5f)
  CHECK(remquo, float, int, 1.5f, 4.0f)
  CHECK(sincos, float, float, 1.5f)

  sycl::float4 VecA{1.5f, 5.1f, 4.0f, 0.4f};
  sycl::float4 VecB{1.5f, 5.1f, 4.0f, 0.4f};

  CHECK(fract, sycl::float4, sycl::float4, VecA)
  CHECK(frexp, sycl::float4, sycl::int4, VecA)
  CHECK(lgamma_r, sycl::float4, sycl::int4, VecA)
  CHECK(remquo, sycl::float4, sycl::int4, VecA, VecB)
  CHECK(sincos, sycl::float4, sycl::float4, VecA)

  return Failed;
}
