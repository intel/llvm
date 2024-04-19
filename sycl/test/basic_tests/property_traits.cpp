// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics

#include <cassert>
#include <sycl/sycl.hpp>

#define CHECK_IS_PROPERTY(PROP)                                                \
  static_assert(is_property<PROP>::value,                                      \
                "no specialization of is_property for " #PROP)

#define CHECK_IS_PROPERTY_V(PROP)                                              \
  static_assert(is_property_v<PROP>,                                           \
                "no specialization of is_property_v for " #PROP)

#define CHECK_IS_PROPERTY_OF(PROP, ...)                                        \
  static_assert(is_property_of<PROP, __VA_ARGS__>::value,                      \
                "no specialization of is_property_of for " #PROP               \
                " and " #__VA_ARGS__)

#define CHECK_IS_PROPERTY_OF_V(PROP, ...)                                      \
  static_assert(is_property_of_v<PROP, __VA_ARGS__>,                           \
                "no specialization of is_property_of_v for " #PROP             \
                " and " #__VA_ARGS__)

#define CHECK_IS_NOT_PROPERTY(PROP)                                            \
  static_assert(!is_property<PROP>::value,                                     \
                "no specialization of is_property for " #PROP)

#define CHECK_IS_NOT_PROPERTY_V(PROP)                                          \
  static_assert(!is_property_v<PROP>,                                          \
                "no specialization of is_property_v for " #PROP)

#define CHECK_IS_NOT_PROPERTY_OF(PROP, ...)                                    \
  static_assert(!is_property_of<PROP, __VA_ARGS__>::value,                     \
                "no specialization of is_property_of for " #PROP               \
                " and " #__VA_ARGS__)

#define CHECK_IS_NOT_PROPERTY_OF_V(PROP, ...)                                  \
  static_assert(!is_property_of_v<PROP, __VA_ARGS__>,                          \
                "no specialization of is_property_of_v for " #PROP             \
                " and " #__VA_ARGS__)

class NotAProperty {};
class NotASYCLObject {};

using namespace sycl;

int main() {
  //----------------------------------------------------------------------------
  // is_property positive tests
  //----------------------------------------------------------------------------

  // Accessor is_property
  CHECK_IS_PROPERTY(property::no_init);
  CHECK_IS_PROPERTY(ext::oneapi::property::no_offset);
  CHECK_IS_PROPERTY(ext::oneapi::property::no_alias);
  CHECK_IS_PROPERTY(ext::intel::property::buffer_location);

  // Buffer is_property
  CHECK_IS_PROPERTY(property::buffer::use_host_ptr);
  CHECK_IS_PROPERTY(property::buffer::use_mutex);
  CHECK_IS_PROPERTY(property::buffer::context_bound);
  CHECK_IS_PROPERTY(property::buffer::mem_channel);
  CHECK_IS_PROPERTY(ext::oneapi::property::buffer::use_pinned_host_memory);

  // Image is_property
  CHECK_IS_PROPERTY(property::image::use_host_ptr);
  CHECK_IS_PROPERTY(property::image::use_mutex);
  CHECK_IS_PROPERTY(property::image::context_bound);

  // Queue is_property
  CHECK_IS_PROPERTY(property::queue::in_order);
  CHECK_IS_PROPERTY(property::queue::enable_profiling);
  CHECK_IS_PROPERTY(property::queue::cuda::use_default_stream);
  CHECK_IS_PROPERTY(ext::oneapi::cuda::property::queue::use_default_stream);

  // Reduction is_property
  CHECK_IS_PROPERTY(property::reduction::initialize_to_identity);

  //----------------------------------------------------------------------------
  // is_property negative tests
  //----------------------------------------------------------------------------

  CHECK_IS_NOT_PROPERTY(NotAProperty);

  //----------------------------------------------------------------------------
  // is_property_of positive tests
  //----------------------------------------------------------------------------

  // Accessor is_property_of
  CHECK_IS_PROPERTY_OF(
      property::no_init,
      accessor<unsigned long, 2, access_mode::read, target::host_buffer,
               access::placeholder::true_t>);
  CHECK_IS_PROPERTY_OF(ext::oneapi::property::no_offset,
                       accessor<long, 3, access_mode::read_write, target::local,
                                access::placeholder::false_t>);
  CHECK_IS_PROPERTY_OF(ext::oneapi::property::no_alias,
                       accessor<bool, 1, access_mode::read, target::device,
                                access::placeholder::true_t>);
  CHECK_IS_PROPERTY_OF(
      ext::intel::property::buffer_location,
      accessor<sycl::half, 2, access_mode::write, target::host_buffer,
               access::placeholder::false_t>);

  // Host-accessor is_property_of
  CHECK_IS_PROPERTY_OF(property::no_init,
                       host_accessor<unsigned long, 2, access_mode::read>);

  // Buffer is_property_of
  CHECK_IS_PROPERTY_OF(property::buffer::use_host_ptr, buffer<int, 2>);
  CHECK_IS_PROPERTY_OF(property::buffer::use_mutex, buffer<char, 1>);
  CHECK_IS_PROPERTY_OF(property::buffer::context_bound, buffer<float, 1>);
  CHECK_IS_PROPERTY_OF(property::buffer::mem_channel, buffer<double, 3>);
  CHECK_IS_PROPERTY_OF(ext::oneapi::property::buffer::use_pinned_host_memory,
                       buffer<unsigned int, 2>);

  // Image is_property_of
  CHECK_IS_PROPERTY_OF(property::image::use_host_ptr, image<1>);
  CHECK_IS_PROPERTY_OF(property::image::use_mutex, image<2>);
  CHECK_IS_PROPERTY_OF(property::image::context_bound, image<3>);

  // Queue is_property_of
  CHECK_IS_PROPERTY_OF(property::queue::in_order, queue);
  CHECK_IS_PROPERTY_OF(property::queue::enable_profiling, queue);
  CHECK_IS_PROPERTY_OF(property::queue::cuda::use_default_stream, queue);
  CHECK_IS_PROPERTY_OF(ext::oneapi::cuda::property::queue::use_default_stream,
                       queue);

  //----------------------------------------------------------------------------
  // is_property_of positive tests
  //----------------------------------------------------------------------------

  // Valid properties with invalid object type
  CHECK_IS_NOT_PROPERTY_OF(property::no_init, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(ext::oneapi::property::no_offset, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(ext::oneapi::property::no_alias, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(ext::intel::property::buffer_location,
                           NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::no_init, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::no_init, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::buffer::use_host_ptr, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::buffer::use_mutex, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::buffer::context_bound, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::buffer::mem_channel, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(
      ext::oneapi::property::buffer::use_pinned_host_memory, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::image::use_host_ptr, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::image::use_mutex, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::image::context_bound, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::queue::in_order, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::queue::enable_profiling, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(property::queue::cuda::use_default_stream,
                           NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF(
      ext::oneapi::cuda::property::queue::use_default_stream, NotASYCLObject);

  // Invalid properties with valid object type
  CHECK_IS_NOT_PROPERTY_OF(NotAProperty, accessor<int, 1>);
  CHECK_IS_NOT_PROPERTY_OF(NotAProperty, host_accessor<char, 2>);
  CHECK_IS_NOT_PROPERTY_OF(NotAProperty, buffer<int, 2>);
  CHECK_IS_NOT_PROPERTY_OF(NotAProperty, context);
  CHECK_IS_NOT_PROPERTY_OF(NotAProperty, image<1>);
  CHECK_IS_NOT_PROPERTY_OF(NotAProperty, queue);

  // Invalid properties with invalid object type
  CHECK_IS_NOT_PROPERTY_OF(NotAProperty, NotASYCLObject);

  //----------------------------------------------------------------------------
  // is_property_v positive tests
  //----------------------------------------------------------------------------

  // Accessor is_property_v
  CHECK_IS_PROPERTY_V(property::no_init);
  CHECK_IS_PROPERTY_V(ext::oneapi::property::no_offset);
  CHECK_IS_PROPERTY_V(ext::oneapi::property::no_alias);
  CHECK_IS_PROPERTY_V(ext::intel::property::buffer_location);

  // Buffer is_property_v
  CHECK_IS_PROPERTY_V(property::buffer::use_host_ptr);
  CHECK_IS_PROPERTY_V(property::buffer::use_mutex);
  CHECK_IS_PROPERTY_V(property::buffer::context_bound);
  CHECK_IS_PROPERTY_V(property::buffer::mem_channel);
  CHECK_IS_PROPERTY_V(ext::oneapi::property::buffer::use_pinned_host_memory);

  // Image is_property_v
  CHECK_IS_PROPERTY_V(property::image::use_host_ptr);
  CHECK_IS_PROPERTY_V(property::image::use_mutex);
  CHECK_IS_PROPERTY_V(property::image::context_bound);

  // Queue is_property_v
  CHECK_IS_PROPERTY_V(property::queue::in_order);
  CHECK_IS_PROPERTY_V(property::queue::enable_profiling);
  CHECK_IS_PROPERTY_V(property::queue::cuda::use_default_stream);
  CHECK_IS_PROPERTY_V(ext::oneapi::cuda::property::queue::use_default_stream);

  //----------------------------------------------------------------------------
  // is_property_v negative tests
  //----------------------------------------------------------------------------

  CHECK_IS_NOT_PROPERTY_V(NotAProperty);

  //----------------------------------------------------------------------------
  // is_property_of_v positive tests
  //----------------------------------------------------------------------------

  // Accessor is_property_of_v
  CHECK_IS_PROPERTY_OF_V(
      property::no_init,
      accessor<unsigned long, 2, access_mode::read, target::host_buffer,
               access::placeholder::true_t>);
  CHECK_IS_PROPERTY_OF_V(ext::oneapi::property::no_offset,
                         accessor<long, 3, access_mode::read_write,
                                  target::local, access::placeholder::false_t>);
  CHECK_IS_PROPERTY_OF_V(ext::oneapi::property::no_alias,
                         accessor<bool, 1, access_mode::read, target::device,
                                  access::placeholder::true_t>);
  CHECK_IS_PROPERTY_OF_V(
      ext::intel::property::buffer_location,
      accessor<sycl::half, 2, access_mode::write, target::host_buffer,
               access::placeholder::false_t>);

  // Host-accessor is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::no_init,
                         host_accessor<unsigned long, 2, access_mode::read>);

  // Image-accessor (SYCL 2020) is_property_of_v
  CHECK_IS_PROPERTY_OF_V(
      property::no_init,
      unsampled_image_accessor<sycl::int4, 3, access_mode::read_write>);
  CHECK_IS_PROPERTY_OF_V(property::no_init,
                         sampled_image_accessor<sycl::float4, 1>);
  CHECK_IS_PROPERTY_OF_V(
      property::no_init,
      host_unsampled_image_accessor<sycl::int4, 2, access_mode::read>);
  CHECK_IS_PROPERTY_OF_V(property::no_init,
                         host_sampled_image_accessor<sycl::float4, 1>);

  // Buffer is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::buffer::use_host_ptr, buffer<int, 2>);
  CHECK_IS_PROPERTY_OF_V(property::buffer::use_mutex, buffer<char, 1>);
  CHECK_IS_PROPERTY_OF_V(property::buffer::context_bound, buffer<float, 1>);
  CHECK_IS_PROPERTY_OF_V(property::buffer::mem_channel, buffer<double, 3>);
  CHECK_IS_PROPERTY_OF_V(ext::oneapi::property::buffer::use_pinned_host_memory,
                         buffer<unsigned int, 2>);

  // Image (SYCL 1.2.1) is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::image::use_host_ptr, image<1>);
  CHECK_IS_PROPERTY_OF_V(property::image::use_mutex, image<2>);
  CHECK_IS_PROPERTY_OF_V(property::image::context_bound, image<3>);

  // Image (SYCL 2020) is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::image::use_host_ptr, sampled_image<1>);
  CHECK_IS_PROPERTY_OF_V(property::image::use_mutex, sampled_image<2>);
  CHECK_IS_PROPERTY_OF_V(property::image::context_bound, sampled_image<3>);
  CHECK_IS_PROPERTY_OF_V(property::image::use_host_ptr, unsampled_image<1>);
  CHECK_IS_PROPERTY_OF_V(property::image::use_mutex, unsampled_image<2>);
  CHECK_IS_PROPERTY_OF_V(property::image::context_bound, unsampled_image<3>);

  // Queue is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::queue::in_order, queue);
  CHECK_IS_PROPERTY_OF_V(property::queue::enable_profiling, queue);
  CHECK_IS_PROPERTY_OF_V(property::queue::cuda::use_default_stream, queue);
  CHECK_IS_PROPERTY_OF_V(ext::oneapi::cuda::property::queue::use_default_stream,
                         queue);

  // Reduction is_property_v
  CHECK_IS_PROPERTY_V(property::reduction::initialize_to_identity);

  //----------------------------------------------------------------------------
  // is_property_of positive tests
  //----------------------------------------------------------------------------

  // Valid properties with invalid object type
  CHECK_IS_NOT_PROPERTY_OF_V(property::no_init, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(ext::oneapi::property::no_offset, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(ext::oneapi::property::no_alias, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(ext::intel::property::buffer_location,
                             NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::no_init, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::no_init, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::buffer::use_host_ptr, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::buffer::use_mutex, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::buffer::context_bound, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::buffer::mem_channel, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(
      ext::oneapi::property::buffer::use_pinned_host_memory, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::image::use_host_ptr, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::image::use_mutex, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::image::context_bound, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::queue::in_order, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::queue::enable_profiling, NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(property::queue::cuda::use_default_stream,
                             NotASYCLObject);
  CHECK_IS_NOT_PROPERTY_OF_V(
      ext::oneapi::cuda::property::queue::use_default_stream, NotASYCLObject);

  // Invalid properties with valid object type
  CHECK_IS_NOT_PROPERTY_OF_V(NotAProperty, accessor<int, 1>);
  CHECK_IS_NOT_PROPERTY_OF_V(NotAProperty, host_accessor<char, 2>);
  CHECK_IS_NOT_PROPERTY_OF_V(NotAProperty, buffer<int, 2>);
  CHECK_IS_NOT_PROPERTY_OF_V(NotAProperty, context);
  CHECK_IS_NOT_PROPERTY_OF_V(NotAProperty, image<1>);
  CHECK_IS_NOT_PROPERTY_OF_V(NotAProperty, queue);

  // Invalid properties with invalid object type
  CHECK_IS_NOT_PROPERTY_OF_V(NotAProperty, NotASYCLObject);
}
