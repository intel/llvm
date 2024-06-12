// This test is adapted from "test-e2e/Basic/work_group_size_prop.cpp"

#include "../graph_common.hpp"

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
int test(queue &Queue, PropertiesT Props, KernelType KernelFunc) {
  constexpr size_t Dims = sizeof...(Is);

  // Positive test case: Specify local size that matches required size.
  exp_ext::command_graph Graph{Queue.get_context(), Queue.get_device()};

  try {

    add_node(Graph, Queue, [&](handler &CGH) {
      CGH.parallel_for<ReqdWGSizePositiveA<KernelVariant, false, Is...>>(
          nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(Is...)), Props,
          KernelFunc);
    });

#ifdef GRAPH_E2E_RECORD_REPLAY
    Graph.begin_recording(Queue);
    Queue.parallel_for<ReqdWGSizePositiveA<KernelVariant, true, Is...>>(
        nd_range<Dims>(repeatRange<Dims>(8), range<Dims>(Is...)), Props,
        KernelFunc);
    Graph.end_recording(Queue);
#endif

    auto ExecGraph = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
    Queue.wait_and_throw();
  } catch (nd_range_error &E) {
    std::cerr << "Test case failed: unexpected "
                 "nd_range_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (runtime_error &E) {
    std::cerr << "Test case failed: unexpected "
                 "runtime_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  // Negative test case: Specify local size that does not match required size.
  exp_ext::command_graph GraphN{Queue.get_context(), Queue.get_device()};
  try {
    add_node(GraphN, Queue, [&](handler &CGH) {
      CGH.parallel_for<ReqdWGSizeNegativeA<KernelVariant, false, Is...>>(
          nd_range<Dims>(repeatRange<Dims>(16), repeatRange<Dims>(8)), Props,
          KernelFunc);
    });
    auto ExecGraph = GraphN.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
    Queue.wait_and_throw();

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

#ifdef GRAPH_E2E_RECORD_REPLAY
  // Same as above but using the queue shortcuts.
  try {
    GraphN.begin_recording(Queue);

    Queue.parallel_for<ReqdWGSizeNegativeA<KernelVariant, true, Is...>>(
        nd_range<Dims>(repeatRange<Dims>(16), repeatRange<Dims>(8)), Props,
        KernelFunc);

    GraphN.end_recording(Queue);
    auto ExecGraph = GraphN.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(ExecGraph); });
    Queue.wait_and_throw();

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
#endif

  return 0;
}

template <size_t... Is> int test(queue &Queue) {
  auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::work_group_size<Is...>};
  auto KernelFunction = [](auto) {};

  auto EmptyProps = sycl::ext::oneapi::experimental::properties{};
  KernelFunctorWithWGSizeProp<Is...> KernelFunctor;

  int Res = 0;
  Res += test<Variant::Function, Is...>(Queue, Props, KernelFunction);
  Res += test<Variant::Functor, Is...>(Queue, EmptyProps, KernelFunctor);
  Res += test<Variant::FunctorAndProperty, Is...>(Queue, Props, KernelFunctor);
  return Res;
}

int main() {
  queue Queue{};

  int Res = 0;
  Res += test<4>(Queue);
  Res += test<4, 4>(Queue);
  Res += test<8, 4>(Queue);
  Res += test<4, 8>(Queue);
  Res += test<4, 4, 4>(Queue);
  Res += test<4, 4, 8>(Queue);
  Res += test<8, 4, 4>(Queue);
  Res += test<4, 8, 4>(Queue);
  return Res;
}
