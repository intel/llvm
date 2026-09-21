// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics
//
// Tests the sycl_khr_properties image properties (use_host_ptr, use_mutex,
// context_bound): that unsampled_image and sampled_image can be constructed
// with them, and that they are registered for both. (The property types
// themselves are covered in buffer_properties.cpp.)

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/image.hpp>

#include <mutex>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

using uimg = sycl::unsampled_image<1>;
using simg = sycl::sampled_image<1>;

// Registered for both image classes (including a multi-dimensional instance).
static_assert(is_property_for_v<kp::use_host_ptr, uimg> &&
              is_property_for_v<kp::use_mutex, uimg> &&
              is_property_for_v<kp::context_bound, uimg>);
static_assert(is_property_for_v<kp::use_host_ptr, simg> &&
              is_property_for_v<kp::use_mutex, simg> &&
              is_property_for_v<kp::context_bound, simg>);
static_assert(is_property_for_v<kp::use_host_ptr, sycl::unsampled_image<2>> &&
              is_property_for_v<kp::use_mutex, sycl::unsampled_image<2>> &&
              is_property_for_v<kp::context_bound, sycl::unsampled_image<2>>);

void ctors(std::mutex &m, sycl::context ctx) {
  sycl::range<1> r{4};
  auto fmt = sycl::image_format::r8g8b8a8_unorm;
  sycl::image_sampler samp{sycl::addressing_mode::none,
                           sycl::coordinate_normalization_mode::unnormalized,
                           sycl::filtering_mode::nearest};
  const void *hp = nullptr;

  sycl::unsampled_image<1> u1{fmt, r, kp::context_bound{ctx}};
  sycl::unsampled_image<1> u2{fmt, r,
                              properties{kp::use_host_ptr{}, kp::use_mutex{m}}};
  sycl::sampled_image<1> s1{hp, fmt, samp, r, kp::use_mutex{m}};

  // Old-style construction must still resolve unambiguously.
  sycl::unsampled_image<1> o1{fmt, r};
  sycl::sampled_image<1> o2{hp, fmt, samp, r};
  (void)u1;
  (void)u2;
  (void)s1;
  (void)o1;
  (void)o2;
}
