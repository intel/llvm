// Tests the command_graph information descriptors:
//   info::graph::recording_mode (modifiable and executable state)
//   info::graph::updatable      (executable state)
//
// Native recording is exercised by two wrappers under
// RecordReplay/NativeRecording/: one passing
// property::graph::enable_native_recording, and one setting
// SYCL_GRAPH_FORCE_NATIVE_RECORDING=1 with an empty property list. The latter
// checks that the descriptors report effective state rather than property-list
// contents.

#include "../graph_common.hpp"
#include <sycl/properties/all_properties.hpp>

#include <type_traits>

// Whether native recording is expected to be in effect for the graphs below.
#if defined(GRAPH_E2E_NATIVE_RECORDING) ||                                     \
    defined(GRAPH_E2E_FORCE_NATIVE_RECORDING_ENV)
constexpr bool ExpectNative = true;
#else
constexpr bool ExpectNative = false;
#endif

constexpr auto ExpectedMode = ExpectNative
                                  ? exp_ext::graph_recording_mode::native
                                  : exp_ext::graph_recording_mode::runtime;

int main() {
  device Dev;
  context Ctx{Dev};
  queue Q{Ctx, Dev, {property::queue::in_order{}}};

  // Deliberately empty in the force-env-var configuration: the descriptor must
  // report `native` without the property ever appearing in a property list.
#if defined(GRAPH_E2E_NATIVE_RECORDING) &&                                     \
    !defined(GRAPH_E2E_FORCE_NATIVE_RECORDING_ENV)
  const property_list GraphProps{
      exp_ext::property::graph::enable_native_recording{}};
#else
  const property_list GraphProps{};
#endif

  // Descriptors report the return types they declare.
  static_assert(
      std::is_same_v<exp_ext::info::graph::recording_mode::return_type,
                     exp_ext::graph_recording_mode>);
  static_assert(
      std::is_same_v<exp_ext::info::graph::updatable::return_type, bool>);

  const size_t N = 1024;
  int *Data = malloc_device<int>(N, Q);

  // Test 1: recording_mode is answerable before, during and after recording,
  // and the executable graph reports the same mode as its parent.
  {
    exp_ext::command_graph Graph{Ctx, Dev, GraphProps};
    assert(Graph.get_info<exp_ext::info::graph::recording_mode>() ==
           ExpectedMode);

    Graph.begin_recording(Q);
    assert(Graph.get_info<exp_ext::info::graph::recording_mode>() ==
           ExpectedMode);
    Q.submit([&](handler &CGH) {
      CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] = Idx; });
    });
    Graph.end_recording();
    assert(Graph.get_info<exp_ext::info::graph::recording_mode>() ==
           ExpectedMode);

    auto ExecGraph = Graph.finalize();
    assert(ExecGraph.get_info<exp_ext::info::graph::recording_mode>() ==
           ExpectedMode);

    // Not finalized with property::graph::updatable.
    assert(!ExecGraph.get_info<exp_ext::info::graph::updatable>());

    Q.ext_oneapi_graph(ExecGraph).wait();
  }

  // Test 2: updatable round-trips through finalize(). Native recording rejects
  // the updatable property at finalize(), so `true` is unreachable there.
  if (!ExpectNative) {
    exp_ext::command_graph Graph{Ctx, Dev, GraphProps};
    Graph.begin_recording(Q);
    Q.submit([&](handler &CGH) {
      CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] = Idx + 1; });
    });
    Graph.end_recording();

    auto ExecGraph = Graph.finalize({exp_ext::property::graph::updatable{}});
    assert(ExecGraph.get_info<exp_ext::info::graph::updatable>());
    assert(ExecGraph.get_info<exp_ext::info::graph::recording_mode>() ==
           exp_ext::graph_recording_mode::runtime);

    Q.ext_oneapi_graph(ExecGraph).wait();
  }

  // Test 3: a graph built only through the explicit API and never recorded
  // from a queue still reports a mode, and that mode is `runtime`. Not
  // reachable under native recording, where add() throws.
  if (!ExpectNative) {
    exp_ext::command_graph Graph{Ctx, Dev};
    auto NodeA = Graph.add([&](handler &CGH) {
      CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] = Idx + 2; });
    });
    auto NodeB = Graph.add([&](handler &CGH) {
      CGH.parallel_for(range<1>{N}, [=](id<1> Idx) { Data[Idx] *= 2; });
    });
    Graph.make_edge(NodeA, NodeB);

    assert(Graph.get_info<exp_ext::info::graph::recording_mode>() ==
           exp_ext::graph_recording_mode::runtime);
    assert(Graph.finalize().get_info<exp_ext::info::graph::recording_mode>() ==
           exp_ext::graph_recording_mode::runtime);
  }

  free(Data, Q);
  return 0;
}
