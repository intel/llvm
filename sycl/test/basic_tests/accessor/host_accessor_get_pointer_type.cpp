// RUN: %clangxx -fsycl -fsyntax-only -sycl-std=2020 %s

// Tests the type of the get_pointer method of host_accessor.

#include <sycl/sycl.hpp>

#include <type_traits>

template <typename DataT, int Dims, sycl::access::mode Mode>
void CheckHostAccessor() {
  using HostAccessorT = sycl::host_accessor<DataT, Dims, Mode>;
  using HostAccessorGetPointerT =
      decltype(std::declval<HostAccessorT>().get_pointer());
  static_assert(
      std::is_same_v<HostAccessorGetPointerT,
                     std::add_pointer_t<typename HostAccessorT::value_type>>);
}

template <typename DataT, int Dims> void CheckHostAccessorForModes() {
  CheckHostAccessor<DataT, Dims, sycl::access::mode::read>();
  if constexpr (!std::is_const<DataT>::value) {
    CheckHostAccessor<DataT, Dims, sycl::access::mode::read_write>();
    CheckHostAccessor<DataT, Dims, sycl::access::mode::write>();
  }
}

template <typename DataT> void CheckHostAccessorForAllDimsAndModes() {
  CheckHostAccessorForModes<DataT, 1>();
  CheckHostAccessorForModes<DataT, 2>();
  CheckHostAccessorForModes<DataT, 3>();
}

int main() {
  CheckHostAccessorForAllDimsAndModes<int>();
  CheckHostAccessorForAllDimsAndModes<const int>();
  return 0;
}
