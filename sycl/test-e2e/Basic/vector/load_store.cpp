// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

// Tests load and store on sycl::vec.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/bfloat16_math.hpp>
#include <sycl/types.hpp>

namespace syclex = sycl::ext::oneapi;

template <size_t N, typename T0, typename T1>
int CheckResult(const T0 &Actual, const T1 &Reference, const char *Category) {
  int Failures = 0;
  for (size_t I = 0; I < N; ++I) {
    if (Actual[I] == Reference[I])
      continue;

    std::cout << "Failed at index " << I << ": " << Category << " - "
              << Actual[I] << " != " << Reference[I] << std::endl;
    ++Failures;
  }
  return Failures;
}

template <typename VecT> int RunTest(sycl::queue &Q) {
  using ElemT = typename VecT::element_type;

  int Failures = 0;
  // Load on host.
  // Note: multi_ptr is not usable on host, so only raw pointer is tested.
  {
    const ElemT Ref[] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13};
    VecT V{0};
    V.load(2, Ref);
    Failures += CheckResult<4>(V, Ref + 8, "load with raw pointer on host");
  }

  // Store on host.
  // Note: multi_ptr is not usable on host, so only raw pointer is tested.
  {
    ElemT Out[] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13};
    const VecT V{4, 3, 2, 1};
    V.store(1, Out);
    const ElemT Ref[] = {0, 2, 1, 4, 4, 3, 2, 1, 7, 10, 9, 12, 11, 14, 13};
    Failures +=
        CheckResult<std::size(Ref)>(Out, Ref, "store in raw pointer on host");
  }

  // Load on device.
  {
    const ElemT Ref[] = {0,  2,  1,  4,  3,  6,  5,  8,  7,  10, 9,  12,
                         11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24};
    VecT V[6] = {VecT{0}};

    {
      sycl::buffer<const ElemT, 1> RefBuff{Ref, std::size(Ref)};
      sycl::buffer<VecT, 1> VBuff{V, std::size(V)};

      Q.submit([&](sycl::handler &CGH) {
        sycl::accessor GlobalRefAcc{RefBuff, CGH, sycl::read_only};
        sycl::accessor VAcc{VBuff, CGH, sycl::read_write};
        sycl::local_accessor<ElemT, 1> LocalRefAcc{std::size(Ref), CGH};
        CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
          // Initialize the local and private memory copies.
          ElemT PrivateRef[std::size(Ref)] = {0};
          for (size_t I = 0; I < GlobalRefAcc.size(); ++I) {
            PrivateRef[I] = GlobalRefAcc[I];
            LocalRefAcc[I] = GlobalRefAcc[I];
          }

          // Load with global multi_ptr.
          auto GlobalMPtr =
              GlobalRefAcc
                  .template get_multi_ptr<sycl::access::decorated::no>();
          VAcc[0].load(0, GlobalMPtr);

          // Load with local multi_ptr.
          auto LocalMPtr =
              LocalRefAcc.template get_multi_ptr<sycl::access::decorated::no>();
          VAcc[1].load(1, LocalMPtr);

          // Load with private multi_ptr.
          auto PrivateMPtr = sycl::address_space_cast<
              sycl::access::address_space::private_space,
              sycl::access::decorated::no>(PrivateRef);
          VAcc[2].load(2, PrivateMPtr);

          // Load with global raw pointer.
          const ElemT *GlobalRawPtr = GlobalMPtr.get_raw();
          VAcc[3].load(3, GlobalRawPtr);

          // Load with local raw pointer.
          const ElemT *LocalRawPtr = LocalMPtr.get_raw();
          VAcc[4].load(4, LocalRawPtr);

          // Load with private raw pointer.
          VAcc[5].load(5, PrivateRef);
        });
      });
    }

    Failures +=
        CheckResult<4>(V[0], Ref, "load with global multi_ptr on device");
    Failures +=
        CheckResult<4>(V[1], Ref + 4, "load with local multi_ptr on device");
    Failures +=
        CheckResult<4>(V[2], Ref + 8, "load with private multi_ptr on device");
    Failures += CheckResult<4>(V[3], Ref + 12,
                               "load with global raw pointer on device");
    Failures +=
        CheckResult<4>(V[4], Ref + 16, "load with local raw pointer on device");
    Failures += CheckResult<4>(V[5], Ref + 20,
                               "load with private raw pointer on device");
  }

  // Store on device.
  {
    ElemT Out[24] = {0};
    const VecT V[] = {{0, 2, 1, 4},     {3, 6, 5, 8},     {7, 10, 9, 12},
                      {11, 14, 13, 16}, {15, 18, 17, 20}, {19, 22, 21, 24}};

    {
      sycl::buffer<ElemT, 1> OutBuff{Out, std::size(Out)};

      Q.submit([&](sycl::handler &CGH) {
        sycl::accessor OutAcc{OutBuff, CGH, sycl::read_write};
        sycl::local_accessor<ElemT, 1> LocalOutAcc{std::size(Out), CGH};
        CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
          ElemT PrivateVal[std::size(Out)] = {0};

          // Store in global multi_ptr.
          auto GlobalMPtr =
              OutAcc.template get_multi_ptr<sycl::access::decorated::no>();
          V[0].store(0, GlobalMPtr);

          // Store in local multi_ptr.
          auto LocalMPtr =
              LocalOutAcc.template get_multi_ptr<sycl::access::decorated::no>();
          V[1].store(1, LocalMPtr);

          // Store in private multi_ptr.
          auto PrivateMPtr = sycl::address_space_cast<
              sycl::access::address_space::private_space,
              sycl::access::decorated::no>(PrivateVal);
          V[2].store(2, PrivateMPtr);

          // Store in global raw pointer.
          ElemT *GlobalRawPtr = GlobalMPtr.get_raw();
          V[3].store(3, GlobalRawPtr);

          // Store in local raw pointer.
          ElemT *LocalRawPtr = LocalMPtr.get_raw();
          V[4].store(4, LocalRawPtr);

          // Store in private raw pointer.
          V[5].store(5, PrivateVal);

          // Write local and private results back to the global buffer.
          for (size_t I = 0; I < 4; ++I) {
            OutAcc[4 + I] = LocalMPtr[4 + I];
            OutAcc[8 + I] = PrivateVal[8 + I];
            OutAcc[16 + I] = LocalMPtr[16 + I];
            OutAcc[20 + I] = PrivateVal[20 + I];
          }
        });
      });
    }

    const ElemT Ref[] = {0,  2,  1,  4,  3,  6,  5,  8,  7,  10, 9,  12,
                         11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24};

    Failures += CheckResult<4>(Out, Ref, "store in global multi_ptr on device");
    Failures +=
        CheckResult<4>(Out + 4, Ref + 4, "store in local multi_ptr on device");
    Failures += CheckResult<4>(Out + 8, Ref + 8,
                               "store in private multi_ptr on device");
    Failures += CheckResult<4>(Out + 12, Ref + 12,
                               "store in global raw pointer on device");
    Failures += CheckResult<4>(Out + 16, Ref + 16,
                               "store in local raw pointer on device");
    Failures += CheckResult<4>(Out + 20, Ref + 20,
                               "store in private raw pointer on device");
  }

  return Failures;
}

int main() {
  sycl::queue Q;

  int Failures = 0;

  Failures += RunTest<sycl::int4>(Q);
  Failures += RunTest<sycl::float4>(Q);
  Failures += RunTest<sycl::vec<syclex::bfloat16, 4>>(Q);

  if (Q.get_device().has(sycl::aspect::fp16))
    Failures += RunTest<sycl::half4>(Q);
  if (Q.get_device().has(sycl::aspect::fp64))
    Failures += RunTest<sycl::double4>(Q);

  return Failures;
}
