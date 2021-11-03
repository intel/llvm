// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -c

#include <CL/sycl.hpp>
#include <cassert>

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

using namespace cl::sycl;

int main() {
  // Accessor is_property
  CHECK_IS_PROPERTY(property::noinit);
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

  // Context is_property
  CHECK_IS_PROPERTY(property::context::cuda::use_primary_context);

  // Image is_property
  CHECK_IS_PROPERTY(property::image::use_host_ptr);
  CHECK_IS_PROPERTY(property::image::use_mutex);
  CHECK_IS_PROPERTY(property::image::context_bound);

  // Queue is_property
  CHECK_IS_PROPERTY(property::queue::in_order);
  CHECK_IS_PROPERTY(property::queue::enable_profiling);
  CHECK_IS_PROPERTY(property::queue::cuda::use_default_stream);

  // Reduction is_property
  CHECK_IS_PROPERTY(property::reduction::initialize_to_identity);

  // Accessor is_property_of
  CHECK_IS_PROPERTY_OF(property::noinit,
                       accessor<float, 1, access_mode::write, target::device,
                                access::placeholder::true_t>);
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
  CHECK_IS_PROPERTY_OF(property::noinit,
                       host_accessor<float, 1, access_mode::write>);
  CHECK_IS_PROPERTY_OF(property::no_init,
                       accessor<unsigned long, 2, access_mode::read>);

  // Buffer is_property_of
  CHECK_IS_PROPERTY_OF(property::buffer::use_host_ptr, buffer<int, 2>);
  CHECK_IS_PROPERTY_OF(property::buffer::use_mutex, buffer<char, 1>);
  CHECK_IS_PROPERTY_OF(property::buffer::context_bound, buffer<float, 1>);
  CHECK_IS_PROPERTY_OF(property::buffer::mem_channel, buffer<double, 3>);
  CHECK_IS_PROPERTY_OF(ext::oneapi::property::buffer::use_pinned_host_memory,
                       buffer<unsigned int, 2>);

  // Context is_property_of
  CHECK_IS_PROPERTY_OF(property::context::cuda::use_primary_context, context);

  // Image is_property_of
  CHECK_IS_PROPERTY_OF(property::image::use_host_ptr, image<1>);
  CHECK_IS_PROPERTY_OF(property::image::use_mutex, image<2>);
  CHECK_IS_PROPERTY_OF(property::image::context_bound, image<3>);

  // Queue is_property_of
  CHECK_IS_PROPERTY_OF(property::queue::in_order, queue);
  CHECK_IS_PROPERTY_OF(property::queue::enable_profiling, queue);
  CHECK_IS_PROPERTY_OF(property::queue::cuda::use_default_stream, queue);

#if __cplusplus > 201402L
  // Accessor is_property_v
  CHECK_IS_PROPERTY_V(property::noinit);
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

  // Context is_property_v
  CHECK_IS_PROPERTY_V(property::context::cuda::use_primary_context);

  // Image is_property_v
  CHECK_IS_PROPERTY_V(property::image::use_host_ptr);
  CHECK_IS_PROPERTY_V(property::image::use_mutex);
  CHECK_IS_PROPERTY_V(property::image::context_bound);

  // Queue is_property_v
  CHECK_IS_PROPERTY_V(property::queue::in_order);
  CHECK_IS_PROPERTY_V(property::queue::enable_profiling);
  CHECK_IS_PROPERTY_V(property::queue::cuda::use_default_stream);

  // Accessor is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::noinit,
                         accessor<float, 1, access_mode::write, target::device,
                                  access::placeholder::true_t>);
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

  // Host-ccessor is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::noinit,
                         host_accessor<float, 1, access_mode::write>);
  CHECK_IS_PROPERTY_OF_V(property::no_init,
                         accessor<unsigned long, 2, access_mode::read>);

  // Buffer is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::buffer::use_host_ptr, buffer<int, 2>);
  CHECK_IS_PROPERTY_OF_V(property::buffer::use_mutex, buffer<char, 1>);
  CHECK_IS_PROPERTY_OF_V(property::buffer::context_bound, buffer<float, 1>);
  CHECK_IS_PROPERTY_OF_V(property::buffer::mem_channel, buffer<double, 3>);
  CHECK_IS_PROPERTY_OF_V(ext::oneapi::property::buffer::use_pinned_host_memory,
                         buffer<unsigned int, 2>);

  // Context is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::context::cuda::use_primary_context, context);

  // Image is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::image::use_host_ptr, image<1>);
  CHECK_IS_PROPERTY_OF_V(property::image::use_mutex, image<2>);
  CHECK_IS_PROPERTY_OF_V(property::image::context_bound, image<3>);

  // Queue is_property_of_v
  CHECK_IS_PROPERTY_OF_V(property::queue::in_order, queue);
  CHECK_IS_PROPERTY_OF_V(property::queue::enable_profiling, queue);
  CHECK_IS_PROPERTY_OF_V(property::queue::cuda::use_default_stream, queue);

  // Reduction is_property_v
  CHECK_IS_PROPERTY_V(property::reduction::initialize_to_identity);
#endif
}
