// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_PI_TRACE=2 env SYCL_RT_WARNING_LEVEL=1 \
// RUN: SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=16:32:64 %{run} %t.out 2>&1 \
// RUN: | FileCheck %s --implicit-check-not "ERROR: JIT compilation for kernel fusion failed with message:"

// Test complete fusion of kernels with different ND-ranges.

// Kernels with different ND-ranges should be fused.
// CHECK-COUNT-26: piEnqueueKernelLaunch
// CHECK-NOT: piEnqueueKernelLaunch

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

#include <algorithm>

using namespace sycl;

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////

using DataTy = vec<uint16_t, 3>;
using VecTy = std::vector<DataTy>;

template <std::size_t Dimensions> class FillBase {
public:
  FillBase(accessor<DataTy, 1, access_mode::write> GS,
           accessor<DataTy, 1, access_mode::write> LS,
           accessor<DataTy, 1, access_mode::write> GrS,
           accessor<DataTy, 1, access_mode::write> G,
           accessor<DataTy, 1, access_mode::write> L,
           accessor<DataTy, 1, access_mode::write> Gr)
      : GS{GS}, LS{LS}, GrS{GrS}, G{G}, L{L}, Gr{Gr} {}

protected:
  template <typename F> static DataTy getValue(F gen) {
    DataTy x{0};
    for (std::size_t i = 0; i < Dimensions; ++i) {
      x[i] = gen(i);
    }
    return x;
  }

  accessor<DataTy, 1, access_mode::write> GS;
  accessor<DataTy, 1, access_mode::write> LS;
  accessor<DataTy, 1, access_mode::write> GrS;
  accessor<DataTy, 1, access_mode::write> G;
  accessor<DataTy, 1, access_mode::write> L;
  accessor<DataTy, 1, access_mode::write> Gr;
};

template <int Dimensions> class Fill : public FillBase<Dimensions> {
public:
  using FillBase<Dimensions>::FillBase;
  using FillBase<Dimensions>::getValue;

  void operator()(item<Dimensions> i) const {
    const auto lid = i.get_linear_id();

    FillBase<Dimensions>::GS[lid] =
        getValue([i](int arg) { return i.get_range(arg); });
    FillBase<Dimensions>::G[lid] =
        getValue([i](int arg) { return i.get_id(arg); });
  }
};

template <int Dimensions> class FillLS : public FillBase<Dimensions> {
public:
  using FillBase<Dimensions>::FillBase;
  using FillBase<Dimensions>::getValue;

  void operator()(nd_item<Dimensions> nd) const {
    const auto lid = nd.get_global_linear_id();
    FillBase<Dimensions>::GS[lid] =
        getValue([nd](int arg) { return nd.get_global_range(arg); });
    FillBase<Dimensions>::LS[lid] =
        getValue([nd](int arg) { return nd.get_local_range(arg); });
    FillBase<Dimensions>::GrS[lid] =
        getValue([nd](int arg) { return nd.get_group_range(arg); });
    FillBase<Dimensions>::G[lid] =
        getValue([nd](int arg) { return nd.get_global_id(arg); });
    FillBase<Dimensions>::L[lid] =
        getValue([nd](int arg) { return nd.get_local_id(arg); });
    FillBase<Dimensions>::Gr[lid] =
        getValue([nd](int arg) { return nd.get_group(arg); });
  }
};

////////////////////////////////////////////////////////////////////////////////
// Range description
////////////////////////////////////////////////////////////////////////////////

struct RangeDesc {
  using Indices = std::array<std::size_t, 3>;

  constexpr RangeDesc(std::initializer_list<std::size_t> GS)
      : Dimensions{static_cast<int>(GS.size())}, GS{init(GS)},
        LS{std::nullopt} {}
  constexpr RangeDesc(std::initializer_list<std::size_t> GS,
                      std::initializer_list<std::size_t> LS)
      : Dimensions{static_cast<int>(GS.size())}, GS{init(GS)}, LS{init(LS)} {}

  constexpr std::size_t num_work_items() const { return GS[0] * GS[1] * GS[2]; }

  int Dimensions;
  Indices GS;
  std::optional<Indices> LS;

  template <std::size_t D> range<D> get_range() const;
  template <std::size_t D> nd_range<D> get_nd_range() const;

private:
  static constexpr Indices init(std::initializer_list<std::size_t> sizes) {
    Indices res{1, 1, 1};
    std::copy(sizes.begin(), sizes.end(), res.begin());
    return res;
  }
};

template <> range<1> RangeDesc::get_range<1>() const { return {GS[0]}; }

template <> range<2> RangeDesc::get_range<2>() const { return {GS[0], GS[1]}; }

template <> range<3> RangeDesc::get_range<3>() const {
  return {GS[0], GS[1], GS[2]};
}

template <> nd_range<1> RangeDesc::get_nd_range<1>() const {
  return {get_range<1>(), {(*LS)[0]}};
}

template <> nd_range<2> RangeDesc::get_nd_range<2>() const {
  return {get_range<2>(), {(*LS)[0], (*LS)[1]}};
}

template <> nd_range<3> RangeDesc::get_nd_range<3>() const {
  return {get_range<3>(), {(*LS)[0], (*LS)[1], (*LS)[2]}};
}

////////////////////////////////////////////////////////////////////////////////
// Test
////////////////////////////////////////////////////////////////////////////////

using TestResult = std::vector<VecTy>;

TestResult run(const std::vector<RangeDesc> &sizes, bool fuse) {
  const auto numWorkItems =
      std::max_element(sizes.begin(), sizes.end(),
                       [](const auto &LHS, const auto &RHS) {
                         return LHS.num_work_items() < RHS.num_work_items();
                       })
          ->num_work_items();
  TestResult res(6 * sizes.size(), VecTy(numWorkItems));

  {
    queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
    std::vector<buffer<DataTy>> Buffers;
    Buffers.reserve(res.size());
    for (auto &v : res) {
      Buffers.emplace_back(v);
    }
    if (fuse) {
      ext::codeplay::experimental::fusion_wrapper fw{q};
      fw.start_fusion();
    }
    for (std::size_t i = 0; i < sizes.size(); ++i) {
      q.submit([&](handler &cgh) {
        const auto &size = *(sizes.begin() + i);
        const auto j = i * 6;
        accessor GS{Buffers[j], cgh, write_only};
        accessor LS{Buffers[j + 1], cgh, write_only};
        accessor GrS{Buffers[j + 2], cgh, write_only};
        accessor G{Buffers[j + 3], cgh, write_only};
        accessor L{Buffers[j + 4], cgh, write_only};
        accessor Gr{Buffers[j + 5], cgh, write_only};
        if (size.LS) {
          switch (size.Dimensions) {
          case 1:
            cgh.parallel_for(size.template get_nd_range<1>(),
                             FillLS<1>{GS, LS, GrS, G, L, Gr});
            break;
          case 2:
            cgh.parallel_for(size.template get_nd_range<2>(),
                             FillLS<2>{GS, LS, GrS, G, L, Gr});
            break;
          case 3:
            cgh.parallel_for(size.template get_nd_range<3>(),
                             FillLS<3>{GS, LS, GrS, G, L, Gr});
            break;
          }
        } else {
          switch (size.Dimensions) {
          case 1:
            cgh.parallel_for(size.template get_range<1>(),
                             Fill<1>{GS, LS, GrS, G, L, Gr});
            break;
          case 2:
            cgh.parallel_for(size.template get_range<2>(),
                             Fill<2>{GS, LS, GrS, G, L, Gr});
            break;
          case 3:
            cgh.parallel_for(size.template get_range<3>(),
                             Fill<3>{GS, LS, GrS, G, L, Gr});
            break;
          }
        }
      });
    }
    if (fuse) {
      ext::codeplay::experimental::fusion_wrapper fw{q};
      assert(fw.is_in_fusion_mode() && "Fusion failed");
      fw.complete_fusion(
          {ext::codeplay::experimental::property::no_barriers{}});
      assert(!fw.is_in_fusion_mode() && "Fusion failed");
    }
  }

  return res;
}

void test(const std::vector<RangeDesc> &sizes) {
  const auto res = run(sizes, /*fuse*/ false);
  const auto fusedRes = run(sizes, /*fuse*/ true);
  assert(std::equal(res.begin(), res.end(), fusedRes.begin(),
                    [](const auto &LHS, const auto &RHS) {
                      return std::equal(LHS.begin(), LHS.end(), RHS.begin(),
                                        [](const auto &LHS, const auto &RHS) {
                                          return all(LHS == RHS);
                                        });
                    }) &&
         "COMPUTATION ERROR");
}

int main() {
  // 1-D kernels with different global sizes
  test({RangeDesc{10}, RangeDesc{20}});
  test({RangeDesc{10}, RangeDesc{20}, RangeDesc{30}});

  // Two 1-D kernels with different global sizes and a 2-D kernel with more
  // work-items.
  test({RangeDesc{10}, RangeDesc{20}, RangeDesc{10, 10}});

  // Two 1-D kernels with different global sizes and specified (equal) local
  // size.
  const auto R2 = {2ul};
  test({RangeDesc{{10}, R2}, RangeDesc{{20}, R2}});

  // Three 1-D kernels with different global sizes and specified (equal) local
  // size.
  const auto R5 = {5ul};
  test({RangeDesc{{10}, R5}, RangeDesc{{20}, R5}, RangeDesc{{30}, R5}});

  // Test global sizes that trigger the rounded range kernel insertion.
  // Note that we lower the RR threshold when running this test.
  test({RangeDesc{67}, RangeDesc{87}, RangeDesc{64}});

  // Test multi-dimensional range-rounded kernels. Only the first dimension will
  // be rounded up.
  test({RangeDesc{30, 67}, RangeDesc{76, 55}, RangeDesc{64, 64}});
}
