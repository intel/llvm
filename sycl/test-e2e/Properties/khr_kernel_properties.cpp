// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Checks that the sycl_khr_kernel_properties properties embedded in a kernel
// have the same effect as the equivalent kernel attributes, including on the
// kernel information queries and is_compatible. Every get member is declared
// but not defined, so the implementation calling it is a link error.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/khr/kernel_properties.hpp>
#include <sycl/sub_group.hpp>
#include <sycl/usm.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>

namespace kp = sycl::khr::property;
using namespace sycl;

static int Failures = 0;
#define CHECK(Cond)                                                            \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      std::cerr << __LINE__ << ": " #Cond " failed\n";                         \
      ++Failures;                                                              \
    }                                                                          \
  } while (0)

constexpr std::uint32_t SGSizes[] = {1, 2, 4, 8, 16, 32, 64};

// What the first work-item of a kernel records about its launch.
struct Launch {
  std::size_t Local;
  std::size_t SubGroup;
};

template <int D> void record(Launch *Out, nd_item<D> It) {
  if (It.get_global_linear_id() == 0)
    *Out = {It.get_local_range().size(),
            It.get_sub_group().get_max_local_range()[0]};
}

// Kernels with an embedded property, and their equivalents with an attribute.

template <std::size_t... Dims> struct WGKernel {
  Launch *Out;
  void operator()(nd_item<sizeof...(Dims)> It) const { record(Out, It); }
  decltype(kp::work_group_size<Dims...>) get(kp::tag::kernel) const;
};

template <std::size_t... Dims> struct WGAttr;
template <std::size_t X> struct WGAttr<X> {
  Launch *Out;
  [[sycl::reqd_work_group_size(X)]] void operator()(nd_item<1> It) const {
    record(Out, It);
  }
};
template <std::size_t X, std::size_t Y> struct WGAttr<X, Y> {
  Launch *Out;
  [[sycl::reqd_work_group_size(X, Y)]] void operator()(nd_item<2> It) const {
    record(Out, It);
  }
};
template <std::size_t X, std::size_t Y, std::size_t Z> struct WGAttr<X, Y, Z> {
  Launch *Out;
  [[sycl::reqd_work_group_size(X, Y, Z)]] void operator()(nd_item<3> It) const {
    record(Out, It);
  }
};

template <std::size_t... Dims> struct WGHintKernel {
  Launch *Out;
  void operator()(nd_item<sizeof...(Dims)> It) const { record(Out, It); }
  decltype(kp::work_group_size_hint<Dims...>) get(kp::tag::kernel) const;
};

template <std::uint32_t S> struct SGKernel {
  Launch *Out;
  void operator()(nd_item<1> It) const { record(Out, It); }
  decltype(kp::sub_group_size<S>) get(kp::tag::kernel) const;
};

template <std::uint32_t S> struct SGAttr {
  Launch *Out;
  [[sycl::reqd_sub_group_size(S)]] void operator()(nd_item<1> It) const {
    record(Out, It);
  }
};

// The attribute is allowed next to the property with the same value.
struct WGBoth {
  Launch *Out;
  [[sycl::reqd_work_group_size(8)]] void operator()(nd_item<1> It) const {
    record(Out, It);
  }
  decltype(khr::properties{kp::work_group_size<8>}) get(kp::tag::kernel) const;
};

template <std::uint32_t S> struct SGBoth {
  Launch *Out;
  [[sycl::reqd_sub_group_size(S)]] void operator()(nd_item<1> It) const {
    record(Out, It);
  }
  decltype(khr::properties{kp::sub_group_size<S>}) get(kp::tag::kernel) const;
};

// The properties apply to every function call operator.
template <std::size_t W> struct MultiWG {
  Launch *Out;
  void operator()(nd_item<1> It) const { record(Out, It); }
  void operator()(group<1> G) const {
    if (G.get_group_linear_id() == 0)
      Out->Local = G.get_local_range().size();
  }
  decltype(kp::work_group_size<W>) get(kp::tag::kernel) const;
};
template <std::size_t W> class MultiWGNDItem;
template <std::size_t W> class MultiWGGroup;

template <std::uint32_t S> struct MultiSG {
  Launch *Out;
  void operator()(nd_item<1> It) const { record(Out, It); }
  void operator()(nd_item<2> It) const { record(Out, It); }
  decltype(kp::sub_group_size<S>) get(kp::tag::kernel) const;
};
template <std::uint32_t S> class MultiSG1;
template <std::uint32_t S> class MultiSG2;

template <std::uint32_t S> struct AllProps {
  Launch *Out;
  void operator()(nd_item<1> It) const { record(Out, It); }
  decltype(khr::properties{kp::work_group_size<64>,
                           kp::work_group_size_hint<64>,
                           kp::sub_group_size<S>}) get(kp::tag::kernel) const;
};

struct EmptyProps {
  Launch *Out;
  void operator()(nd_item<1> It) const { record(Out, It); }
  khr::empty_properties_t get(kp::tag::kernel) const;
};

struct NoProps {
  Launch *Out;
  void operator()(nd_item<1> It) const { record(Out, It); }
};

// Helpers.

template <int D> range<D> ones() {
  if constexpr (D == 1)
    return {1};
  else if constexpr (D == 2)
    return {1, 1};
  else
    return {1, 1, 1};
}

template <int D> bool supportsWG(const device &Dev, range<D> WG) {
  if (WG.size() > Dev.get_info<info::device::max_work_group_size>())
    return false;
  id<D> Max = Dev.get_info<info::device::max_work_item_sizes<D>>();
  for (int I = 0; I < D; ++I)
    if (WG[I] > Max[I])
      return false;
  return true;
}

bool supportsSG(const device &Dev, std::uint32_t S) {
  auto Sizes = Dev.get_info<info::device::sub_group_sizes>();
  return std::find(Sizes.begin(), Sizes.end(), S) != Sizes.end();
}

template <typename F> bool throws(errc Code, F &&Func) {
  try {
    Func();
  } catch (const sycl::exception &E) {
    return E.code() == Code;
  }
  return false;
}

template <typename Name> kernel getKernel(queue &Q) {
  auto Bundle = get_kernel_bundle<Name, bundle_state::executable>(
      Q.get_context(), {Q.get_device()});
  return Bundle.template get_kernel<Name>();
}

template <typename Name> range<3> compileWG(queue &Q) {
  return getKernel<Name>(Q)
      .template get_info<info::kernel_device_specific::compile_work_group_size>(
          Q.get_device());
}

template <typename Name> std::uint32_t compileSG(queue &Q) {
  return getKernel<Name>(Q)
      .template get_info<info::kernel_device_specific::compile_sub_group_size>(
          Q.get_device());
}

Launch read(queue &Q, Launch *Out) {
  Launch L;
  Q.copy(Out, &L, 1).wait();
  return L;
}

template <typename Name, typename K, int D>
Launch viaHandler(queue &Q, Launch *Out, nd_range<D> R) {
  Q.memset(Out, 0, sizeof(Launch)).wait();
  Q.submit([&](handler &CGH) { CGH.parallel_for<Name>(R, K{Out}); }).wait();
  return read(Q, Out);
}

template <typename Name, typename K, int D>
Launch viaQueue(queue &Q, Launch *Out, nd_range<D> R) {
  Q.memset(Out, 0, sizeof(Launch)).wait();
  Q.parallel_for<Name>(R, K{Out}).wait();
  return read(Q, Out);
}

// Tests.

// K embeds a work-group size WG and A is decorated with the attribute.
template <typename K, typename A, int D>
void checkWG(queue &Q, Launch *Out, range<D> WG) {
  device Dev = Q.get_device();
  CHECK(is_compatible<K>(Dev) == is_compatible<A>(Dev));
  nd_range<D> R{WG, WG};
  if (supportsWG(Dev, WG)) {
    CHECK((viaHandler<K, K>(Q, Out, R).Local == WG.size()));
    CHECK((viaQueue<K, K>(Q, Out, R).Local == WG.size()));
    CHECK((viaHandler<A, A>(Q, Out, R).Local == WG.size()));
    // The runtime only reports a mismatch once per kernel and nd_range, so
    // each launch uses a different local size.
    range<D> Other = ones<D>();
    Other[0] = WG[0] == 1 ? 2 : 1;
    CHECK(throws(errc::nd_range,
                 [&] { viaHandler<K, K>(Q, Out, nd_range<D>{Other, Other}); }));
    Other[0] = WG[0] + 1;
    CHECK(throws(errc::nd_range,
                 [&] { viaQueue<K, K>(Q, Out, nd_range<D>{Other, Other}); }));
    CHECK(compileWG<K>(Q) == compileWG<A>(Q));
  } else {
    CHECK(!is_compatible<K>(Dev));
    CHECK(throws(errc::kernel_not_supported,
                 [&] { viaHandler<K, K>(Q, Out, R); }));
    CHECK(
        throws(errc::kernel_not_supported, [&] { viaQueue<K, K>(Q, Out, R); }));
    CHECK(throws(errc::kernel_not_supported,
                 [&] { viaHandler<A, A>(Q, Out, R); }));
  }
}

template <std::size_t... Dims> void testWorkGroupSize(queue &Q, Launch *Out) {
  checkWG<WGKernel<Dims...>, WGAttr<Dims...>>(Q, Out,
                                              range<sizeof...(Dims)>{Dims...});
}

template <std::size_t... Dims>
void testWorkGroupSizeHint(queue &Q, Launch *Out) {
  constexpr int D = sizeof...(Dims);
  using K = WGHintKernel<Dims...>;
  CHECK(
      (viaHandler<K, K>(Q, Out, nd_range<D>{ones<D>(), ones<D>()}).Local == 1));
  range<D> WG{Dims...};
  if (supportsWG(Q.get_device(), WG))
    CHECK((viaHandler<K, K>(Q, Out, nd_range<D>{WG, WG}).Local == WG.size()));
}

// K embeds a sub-group size S and A is decorated with the attribute.
template <typename K, typename A>
void checkSG(queue &Q, Launch *Out, std::uint32_t S) {
  device Dev = Q.get_device();
  CHECK(is_compatible<K>(Dev) == is_compatible<A>(Dev));
  if (supportsSG(Dev, S)) {
    CHECK((viaHandler<K, K>(Q, Out, nd_range<1>{S, S}).SubGroup == S));
    CHECK((viaHandler<A, A>(Q, Out, nd_range<1>{S, S}).SubGroup == S));
    CHECK(compileSG<K>(Q) == S);
    CHECK(compileSG<K>(Q) == compileSG<A>(Q));
  } else {
    CHECK(!is_compatible<K>(Dev));
    CHECK(throws(errc::kernel_not_supported,
                 [&] { viaHandler<K, K>(Q, Out, nd_range<1>{1, 1}); }));
    CHECK(throws(errc::kernel_not_supported,
                 [&] { viaHandler<A, A>(Q, Out, nd_range<1>{1, 1}); }));
  }
}

template <std::size_t W> void testMultiWG(queue &Q, Launch *Out) {
  if (!supportsWG(Q.get_device(), range<1>{W}))
    return;
  using K = MultiWG<W>;
  CHECK(
      (viaHandler<MultiWGNDItem<W>, K>(Q, Out, nd_range<1>{W, W}).Local == W));
  Q.memset(Out, 0, sizeof(Launch)).wait();
  Q.submit([&](handler &CGH) {
     CGH.parallel_for_work_group<MultiWGGroup<W>>(range<1>{1}, range<1>{W},
                                                  K{Out});
   }).wait();
  CHECK(read(Q, Out).Local == W);
  viaHandler<WGAttr<W>, WGAttr<W>>(Q, Out, nd_range<1>{W, W});
  CHECK(compileWG<MultiWGNDItem<W>>(Q) == compileWG<WGAttr<W>>(Q));
  CHECK(compileWG<MultiWGGroup<W>>(Q) == compileWG<WGAttr<W>>(Q));
  constexpr std::size_t Other = W == 1 ? 2 : 1;
  CHECK(throws(errc::nd_range, [&] {
    viaHandler<MultiWGNDItem<W>, K>(Q, Out, nd_range<1>{Other, Other});
  }));
}

template <std::uint32_t S> void testMultiSG(queue &Q, Launch *Out) {
  if (!supportsSG(Q.get_device(), S))
    return;
  using K = MultiSG<S>;
  CHECK((viaHandler<MultiSG1<S>, K>(Q, Out, nd_range<1>{S, S}).SubGroup == S));
  CHECK((viaHandler<MultiSG2<S>, K>(Q, Out, nd_range<2>{{1, S}, {1, S}})
             .SubGroup == S));
  CHECK(compileSG<MultiSG1<S>>(Q) == S);
  CHECK(compileSG<MultiSG2<S>>(Q) == S);
}

template <std::uint32_t S> void testAllProps(queue &Q, Launch *Out) {
  device Dev = Q.get_device();
  if (!supportsSG(Dev, S) || !supportsWG(Dev, range<1>{64}) ||
      Dev.get_info<info::device::max_num_sub_groups>() < 64 / S)
    return;
  using K = AllProps<S>;
  Launch L = viaHandler<K, K>(Q, Out, nd_range<1>{64, 64});
  CHECK(L.Local == 64);
  CHECK(L.SubGroup == S);
  viaHandler<WGAttr<64>, WGAttr<64>>(Q, Out, nd_range<1>{64, 64});
  CHECK(compileWG<K>(Q) == compileWG<WGAttr<64>>(Q));
  CHECK(compileSG<K>(Q) == S);
}

template <std::uint32_t S> void testSubGroupSize(queue &Q, Launch *Out) {
  checkSG<SGKernel<S>, SGAttr<S>>(Q, Out, S);
  checkSG<SGBoth<S>, SGAttr<S>>(Q, Out, S);
  testMultiSG<S>(Q, Out);
  testAllProps<S>(Q, Out);
}

template <std::size_t... I>
void testSubGroupSizes(queue &Q, Launch *Out, std::index_sequence<I...>) {
  (testSubGroupSize<SGSizes[I]>(Q, Out), ...);
}

int main() {
  queue Q;
  Launch *Out = malloc_device<Launch>(1, Q);

  testWorkGroupSize<8>(Q, Out);
  testWorkGroupSize<4, 2>(Q, Out);
  testWorkGroupSize<2, 2, 2>(Q, Out);
  // No device supports this size.
  testWorkGroupSize<std::size_t{1} << 30>(Q, Out);

  testWorkGroupSizeHint<8>(Q, Out);
  testWorkGroupSizeHint<4, 2>(Q, Out);
  testWorkGroupSizeHint<2, 2, 2>(Q, Out);

  testSubGroupSizes(Q, Out, std::make_index_sequence<std::size(SGSizes)>{});
  checkSG<SGKernel<4099>, SGAttr<4099>>(Q, Out, 4099);

  checkWG<WGBoth, WGAttr<8>>(Q, Out, range<1>{8});

  testMultiWG<1>(Q, Out);
  testMultiWG<8>(Q, Out);

  viaHandler<EmptyProps, EmptyProps>(Q, Out, nd_range<1>{8, 8});
  viaHandler<NoProps, NoProps>(Q, Out, nd_range<1>{8, 8});
  CHECK(compileWG<EmptyProps>(Q) == compileWG<NoProps>(Q));
  CHECK(compileSG<EmptyProps>(Q) == compileSG<NoProps>(Q));

  free(Out, Q);
  return Failures;
}
