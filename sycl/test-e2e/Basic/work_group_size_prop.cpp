// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Failing negative test with HIP. Temporarily disabled for CUDA.
// XFAIL: hip, cuda

#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

enum class Variant { Function, Functor, FunctorAndProperty };

template <Variant KernelVariant, bool IsShortcut, size_t... Is>
class ReqdWGSizePositiveA;
template <Variant KernelVariant, bool IsShortcut, size_t... Is>
class ReqdWGSizeNoLocalPositive;
template <Variant KernelVariant, bool IsShortcut, size_t... Is>
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

template <size_t... Is> struct KernelFunctorWithWGSizeProp {
  void operator()(nd_item<sizeof...(Is)>) const {}
  void operator()(item<sizeof...(Is)>) const {}

  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::work_group_size<Is...>};
  }
};

template <Variant KernelVariant, size_t... Is, typename PropertiesT,
          typename KernelType>
int test(queue &Q, PropertiesT Props, KernelType KernelFunc) {
  constexpr size_t Dims = sizeof...(Is);

  bool IsOpenCL = (Q.get_backend() == backend::opencl);

  // Positive test case: Specify local size that matches required size.
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<ReqdWGSizePositiveA<KernelVariant, false, Is...>>(
          nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(Is...)), Props,
          KernelFunc);
    });
    Q.wait_and_throw();
  } catch (nd_range_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected "
                 "nd_range_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected "
                 "runtime_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  // Same as above but using the queue shortcuts.
  try {
    Q.parallel_for<ReqdWGSizePositiveA<KernelVariant, true, Is...>>(
        nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(Is...)), Props,
        KernelFunc);
    Q.wait_and_throw();
  } catch (nd_range_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA shortcut failed: unexpected "
                 "nd_range_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA shortcut failed: unexpected "
                 "runtime_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizePositiveA shortcut failed: something "
                 "unexpected has been caught"
              << std::endl;
    return 1;
  }

  // Kernel that has a required WG size, but no local size is specified.
  //
  // TODO: This fails on OpenCL and should be investigated.
  if (!IsOpenCL) {
    try {
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<
            ReqdWGSizeNoLocalPositive<KernelVariant, false, Is...>>(
            repeatRange<Dims>(16), Props, KernelFunc);
      });
      Q.wait_and_throw();
    } catch (nd_range_error &E) {
      std::cerr << "Test case ReqdWGSizeNoLocalPositive failed: unexpected "
                   "nd_range_error exception: "
                << E.what() << std::endl;
      return 1;
    } catch (runtime_error &E) {
      std::cerr
          << "Test case ReqdWGSizeNoLocalPositive: unexpected runtime_error "
             "exception: "
          << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case ReqdWGSizeNoLocalPositive failed: something "
                   "unexpected has been caught"
                << std::endl;
      return 1;
    }

    try {
      Q.parallel_for<ReqdWGSizeNoLocalPositive<KernelVariant, true, Is...>>(
          repeatRange<Dims>(16), Props, KernelFunc);
      Q.wait_and_throw();
    } catch (nd_range_error &E) {
      std::cerr
          << "Test case ReqdWGSizeNoLocalPositive shortcut failed: unexpected "
             "nd_range_error exception: "
          << E.what() << std::endl;
      return 1;
    } catch (runtime_error &E) {
      std::cerr << "Test case ReqdWGSizeNoLocalPositive shortcut: unexpected "
                   "runtime_error "
                   "exception: "
                << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr
          << "Test case ReqdWGSizeNoLocalPositive shortcut failed: something "
             "unexpected has been caught"
          << std::endl;
      return 1;
    }
  }

  // Negative test case: Specify local size that does not match required size.
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<ReqdWGSizeNegativeA<KernelVariant, false, Is...>>(
          nd_range<Dims>(repeatRange<Dims>(16), repeatRange<Dims>(8)), Props,
          KernelFunc);
    });
    Q.wait_and_throw();
    std::cerr << "Test case ReqdWGSizeNegativeA failed: no exception has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (nd_range_error &E) {
    if (std::string(E.what()).find(
            "The specified local size " + rangeToString(repeatRange<Dims>(8)) +
            " doesn't match the required " +
            "work-group size specified in the program source " +
            rangeToString(range<Dims>(Is...))) == std::string::npos) {
      std::cerr
          << "Test case ReqdWGSizeNegativeA failed: unexpected nd_range_error "
             "exception: "
          << E.what() << std::endl;
      return 1;
    }
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizeNegativeA failed: unexpected "
                 "nd_range_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizeNegativeA failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  // Same as above but using the queue shortcuts.
  try {
    Q.parallel_for<ReqdWGSizeNegativeA<KernelVariant, true, Is...>>(
        nd_range<Dims>(repeatRange<Dims>(16), repeatRange<Dims>(8)), Props,
        KernelFunc);
    Q.wait_and_throw();
    std::cerr << "Test case ReqdWGSizeNegativeA shortcut failed: no exception "
                 "has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (nd_range_error &E) {
    if (std::string(E.what()).find(
            "The specified local size " + rangeToString(repeatRange<Dims>(8)) +
            " doesn't match the required " +
            "work-group size specified in the program source " +
            rangeToString(range<Dims>(Is...))) == std::string::npos) {
      std::cerr << "Test case ReqdWGSizeNegativeA shortcut failed: unexpected "
                   "nd_range_error "
                   "exception: "
                << E.what() << std::endl;
      return 1;
    }
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizeNegativeA shortcut failed: unexpected "
                 "nd_range_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizeNegativeA shortcut failed: something "
                 "unexpected has been caught"
              << std::endl;
    return 1;
  }

  return 0;
}

template <size_t... Is> int test(queue &Q) {
  auto Props = ext::oneapi::experimental::properties{
      ext::oneapi::experimental::work_group_size<Is...>};
  auto KernelFunction = [](auto) {};

  auto EmptyProps = ext::oneapi::experimental::properties{};
  KernelFunctorWithWGSizeProp<Is...> KernelFunctor;

  int Res = 0;
  Res += test<Variant::Function, Is...>(Q, Props, KernelFunction);
  Res += test<Variant::Functor, Is...>(Q, EmptyProps, KernelFunctor);
  Res += test<Variant::FunctorAndProperty, Is...>(Q, Props, KernelFunctor);
  return Res;
}

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto &E : ES) {
      std::rethrow_exception(E);
    }
  };

  queue Q(AsyncHandler);

  int Res = 0;
  Res += test<4>(Q);
  Res += test<4, 4>(Q);
  Res += test<8, 4>(Q);
  Res += test<4, 8>(Q);
  Res += test<4, 4, 4>(Q);
  Res += test<4, 4, 8>(Q);
  Res += test<8, 4, 4>(Q);
  Res += test<4, 8, 4>(Q);
  return Res;
}
