// RUN: %clangxx -fsycl -fsyntax-only %s

// Test checks that the subscript operator and reference alias on accessors
// evaluate to the right types.
#include <sycl/sycl.hpp>

using namespace sycl;

// Trait for getting the return type of a full subscript operation.
template <int Dims, typename AccT> struct FullSubscriptType;
template <typename AccT> struct FullSubscriptType<1, AccT> {
  using type = decltype(std::declval<AccT>()[0]);
};
template <typename AccT> struct FullSubscriptType<2, AccT> {
  using type = decltype(std::declval<AccT>()[0][0]);
};
template <typename AccT> struct FullSubscriptType<3, AccT> {
  using type = decltype(std::declval<AccT>()[0][0][0]);
};
template <int Dims, typename AccT>
using FullSubscriptTypeT = typename FullSubscriptType<Dims, AccT>::type;

// Expected reference type of an accessor given an access mode.
template <access::mode AccessMode, typename DataT>
using ExpectedRefTypeT = std::conditional_t<AccessMode == access::mode::read,
                                            std::add_const_t<DataT> &, DataT &>;

// Trait for getting the expected return type of a full subscript operation.
template <access::mode AccessMode, access::target AccessTarget, typename DataT>
struct ExpectedSubscriptType {
  using type = ExpectedRefTypeT<AccessMode, DataT>;
};
template <typename DataT, access::target AccessTarget>
struct ExpectedSubscriptType<access::mode::atomic, AccessTarget, DataT> {
  using type = atomic<DataT, access::address_space::global_space>;
};
template <typename DataT>
struct ExpectedSubscriptType<access::mode::atomic, access::target::local,
                             DataT> {
  using type = atomic<DataT, access::address_space::local_space>;
};
template <access::mode AccessMode, access::target AccessTarget, typename DataT>
using ExpectedSubscriptTypeT =
    typename ExpectedSubscriptType<AccessMode, AccessTarget, DataT>::type;

template <typename DataT, int Dims, access::mode AccessMode,
          access::target AccessTarget, typename AccT>
void CheckAccRefAndSubscript() {
  static_assert(std::is_same_v<typename AccT::reference,
                               ExpectedRefTypeT<AccessMode, DataT>>);
  static_assert(
      std::is_same_v<FullSubscriptTypeT<Dims, AccT>,
                     ExpectedSubscriptTypeT<AccessMode, AccessTarget, DataT>>);
}

template <typename DataT, int Dims, access::mode AccessMode> void CheckAcc() {
  CheckAccRefAndSubscript<DataT, Dims, AccessMode, access::target::host_buffer,
                          host_accessor<DataT, Dims, AccessMode>>();
  CheckAccRefAndSubscript<
      DataT, Dims, AccessMode, access::target::device,
      accessor<DataT, Dims, AccessMode, access::target::device>>();
  CheckAccRefAndSubscript<
      DataT, Dims, AccessMode, access::target::host_buffer,
      accessor<DataT, Dims, AccessMode, access::target::host_buffer>>();
  if constexpr (AccessMode == access::mode::read_write) {
    CheckAccRefAndSubscript<DataT, Dims, AccessMode, access::target::local,
                            local_accessor<DataT, Dims>>();
  }
  if constexpr (AccessMode == access::mode::read_write ||
                AccessMode == access::mode::atomic) {
    CheckAccRefAndSubscript<
        DataT, Dims, AccessMode, access::target::local,
        accessor<DataT, Dims, AccessMode, access::target::local>>();
  }
}

template <typename DataT, access::mode AccessMode> void CheckAccAllDims() {
  CheckAcc<DataT, 1, AccessMode>();
  CheckAcc<DataT, 2, AccessMode>();
  CheckAcc<DataT, 3, AccessMode>();
}

template <typename DataT> void CheckAccAllAccessModesAndDims() {
  CheckAccAllDims<DataT, access::mode::write>();
  CheckAccAllDims<DataT, access::mode::read_write>();
  CheckAccAllDims<DataT, access::mode::discard_write>();
  CheckAccAllDims<DataT, access::mode::discard_read_write>();
  CheckAccAllDims<DataT, access::mode::read>();
  if constexpr (!std::is_const_v<DataT>)
    CheckAccAllDims<DataT, access::mode::atomic>();
}

int main() {
  CheckAccAllAccessModesAndDims<int>();
  CheckAccAllAccessModesAndDims<const int>();
  return 0;
}
