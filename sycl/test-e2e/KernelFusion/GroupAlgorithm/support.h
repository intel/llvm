#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;

bool isSupportedDevice(device D) {
  std::string PlatformName =
      D.get_platform().get_info<sycl::info::platform::name>();
  if (PlatformName.find("CUDA") != std::string::npos)
    return true;

  if (PlatformName.find("Level-Zero") != std::string::npos)
    return true;

  if (PlatformName.find("OpenCL") != std::string::npos) {
    std::string Version = D.get_info<sycl::info::device::version>();

    // Group collectives are mandatory in OpenCL 2.0 but optional in 3.0.
    Version = Version.substr(7, 3);
    if (Version >= "2.0" && Version < "3.0")
      return true;
  }

  return false;
}

template <typename T, typename S> bool equal(const T &x, const S &y) {
  // vec equal returns a vector of which components were equal
  if constexpr (sycl::detail::is_vec<T>::value) {
    for (int i = 0; i < x.size(); ++i)
      if (x[i] != y[i])
        return false;
    return true;
  } else
    return x == y;
}

template <typename T1, typename T2>
bool ranges_equal(T1 begin1, T1 end1, T2 begin2) {
  for (; begin1 != end1; ++begin1, ++begin2)
    if (!equal(*begin1, *begin2))
      return false;
  return true;
}
