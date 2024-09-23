#pragma once

#include <sycl/detail/core.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

// Property list that contains device_image_scope if USE_DEVICE_IMAGE_SCOPE is
// defined.
#ifdef USE_DEVICE_IMAGE_SCOPE
using TestProperties = decltype(properties{device_image_scope});
#else
using TestProperties = decltype(properties{});
#endif
