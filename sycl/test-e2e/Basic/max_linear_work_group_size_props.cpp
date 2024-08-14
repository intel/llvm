// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <iostream>

using namespace sycl;

enum class Variant { Function, Functor, FunctorAndProperty };

template <Variant KernelVariant, bool IsShortcut, size_t>
class ReqdWGSizePositiveA;
template <Variant KernelVariant, bool IsShortcut, size_t>
class ReqdWGSizeNoLocalPositive;
template <Variant KernelVariant, bool IsShortcut, size_t>
class ReqdWGSizeNegativeA;

template <size_t Dims> range<Dims> repeatRange(size_t Val);
template <> range<1> repeatRange<1>(size_t Val) { return range<1>{Val}; }
template <> range<2> repeatRange<2>(size_t Val) { return range<2>{Val, Val}; }
template <> range<3> repeatRange<3>(size_t Val) {
  return range<3>{Val, Val, Val};
}

std::string rangeToString(range<1> Range) {
  return "{1, 1, " + std::to_string(Range[0]) + "}";
}
std::string rangeToString(range<2> Range) {
  return "{1, " + std::to_string(Range[0]) + ", " + std::to_string(Range[1]) +
         "}";
}
std::string rangeToString(range<3> Range) {
  return "{" + std::to_string(Range[0]) + ", " + std::to_string(Range[1]) +
         ", " + std::to_string(Range[2]) + "}";
}

template <size_t I> struct KernelFunctorWithMaxWGSizeProp {
  void operator()(nd_item<1>) const {}
  void operator()(item<1>) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::max_linear_work_group_size<I>};
  }
};

template <Variant KernelVariant, size_t I, typename PropertiesT,
          typename KernelType>
int test(queue &Q, PropertiesT Props, KernelType KernelFunc) {
  constexpr size_t Dims = 1;

  bool IsOpenCL = (Q.get_backend() == backend::opencl);

  // Positive test case: Specify local size that matches required size.
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<ReqdWGSizePositiveA<KernelVariant, false, I>>(
          nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(I)), Props,
          KernelFunc);
    });
    Q.wait_and_throw();
  } catch (exception &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected exception: "
              << E.what() << std::endl;
    return 1;
  }

  // Same as above but using the queue shortcuts.
  try {
    Q.parallel_for<ReqdWGSizePositiveA<KernelVariant, true, I>>(
        nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(I)), Props,
        KernelFunc);
    Q.wait_and_throw();
  } catch (exception &E) {
    std::cerr << "Test case ReqdWGSizePositiveA shortcut failed: unexpected "
                 "exception: "
              << E.what() << std::endl;
    return 1;
  }

  // Kernel that has a required WG size, but no local size is specified.
  //
  // TODO: This fails on OpenCL and should be investigated.
  if (!IsOpenCL) {
    try {
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<ReqdWGSizeNoLocalPositive<KernelVariant, false, I>>(
            repeatRange<Dims>(16), Props, KernelFunc);
      });
      Q.wait_and_throw();
    } catch (exception &E) {
      std::cerr << "Test case ReqdWGSizeNoLocalPositive failed: unexpected "
                   "exception: "
                << E.what() << std::endl;
      return 1;
    }

    try {
      Q.parallel_for<ReqdWGSizeNoLocalPositive<KernelVariant, true, I>>(
          repeatRange<Dims>(16), Props, KernelFunc);
      Q.wait_and_throw();
    } catch (exception &E) {
      std::cerr << "Test case ReqdWGSizeNoLocalPositive shortcut failed: "
                   "unexpected exception: "
                << E.what() << std::endl;
      return 1;
    }
  }

  // Negative test case: Specify local size that does not match required size.
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<ReqdWGSizeNegativeA<KernelVariant, false, I>>(
          nd_range<Dims>(repeatRange<Dims>(16), repeatRange<Dims>(2)), Props,
          KernelFunc);
    });
    Q.wait_and_throw();
    std::cerr << "Test case ReqdWGSizeNegativeA failed: no exception has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (exception &E) {
    if (E.code() != errc::nd_range ||
        std::string(E.what()).find(
            "The specified local size " + rangeToString(repeatRange<Dims>(8)) +
            " doesn't match the required " +
            "work-group size specified in the program source " +
            rangeToString(range<Dims>(I))) == std::string::npos) {
      std::cerr
          << "Test case ReqdWGSizeNegativeA failed: unexpected exception: "
          << E.what() << std::endl;
      return 1;
    }
  }

  // Same as above but using the queue shortcuts.
  try {
    Q.parallel_for<ReqdWGSizeNegativeA<KernelVariant, true, I>>(
        nd_range<Dims>(repeatRange<Dims>(16), repeatRange<Dims>(8)), Props,
        KernelFunc);
    Q.wait_and_throw();
    std::cerr << "Test case ReqdWGSizeNegativeA shortcut failed: no exception "
                 "has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (exception &E) {
    if (E.code() != errc::nd_range ||
        std::string(E.what()).find(
            "The specified local size " + rangeToString(repeatRange<Dims>(8)) +
            " doesn't match the required " +
            "work-group size specified in the program source " +
            rangeToString(range<Dims>(I))) == std::string::npos) {
      std::cerr << "Test case ReqdWGSizeNegativeA shortcut failed: unexpected "
                   "exception: "
                << E.what() << std::endl;
      return 1;
    }
  }

  return 0;
}

template <size_t I> int test_max(queue &Q) {
  auto Props = ext::oneapi::experimental::properties{
      ext::oneapi::experimental::max_linear_work_group_size<I>};
  auto KernelFunction = [](auto) {};

  auto EmptyProps = ext::oneapi::experimental::properties{};
  KernelFunctorWithMaxWGSizeProp<I> KernelFunctor;

  int Res = 0;
  Res += test<Variant::Function, I>(Q, Props, KernelFunction);
  Res += test<Variant::Functor, I>(Q, EmptyProps, KernelFunctor);
  Res += test<Variant::FunctorAndProperty, I>(Q, Props, KernelFunctor);
  return Res;
}

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto &E : ES) {
      std::rethrow_exception(E);
    }
  };

  queue Q(AsyncHandler);

  return test_max<4>(Q);
}
