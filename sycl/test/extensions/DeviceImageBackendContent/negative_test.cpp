// RUN: %clang -fsycl -fsyntax-only  -std=c++20 -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

#include <sycl/sycl.hpp>

class kernel;

sycl::device d;
sycl::queue q{d};
sycl::context ctxt = q.get_context();
sycl::kernel_id id = sycl::get_kernel_id<kernel>();

int main() {
  // Check that the ext_oneapi_get_backend_content and the
  // ext_oneapi_get_backend_content_view of the content functions are not
  // available
  // when the image is not in the executable state.

  auto input_bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(ctxt, {id});
  // expected-error@+1 {{no matching member function for call to 'ext_oneapi_get_backend_content'}}
  (*input_bundle.begin()).ext_oneapi_get_backend_content();
#ifdef __cpp_lib_span
  // expected-error@+1 {{no matching member function for call to 'ext_oneapi_get_backend_content_view'}}
  (*input_bundle.begin()).ext_oneapi_get_backend_content_view();
#endif

  auto object_bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::object>(ctxt, {id});
  // expected-error@+1 {{no matching member function for call to 'ext_oneapi_get_backend_content'}}
  (*input_bundle.begin()).ext_oneapi_get_backend_content();
#ifdef __cpp_lib_span
  // expected-error@+1 {{no matching member function for call to 'ext_oneapi_get_backend_content_view'}}
  (*input_bundle.begin()).ext_oneapi_get_backend_content_view();
#endif

  auto source_bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::ext_oneapi_source>(ctxt,
                                                                     {id});
  // expected-error@+1 {{no matching member function for call to 'ext_oneapi_get_backend_content'}}
  (*input_bundle.begin()).ext_oneapi_get_backend_content();
#ifdef __cpp_lib_span
  // expected-error@+1 {{no matching member function for call to 'ext_oneapi_get_backend_content_view'}}
  (*input_bundle.begin()).ext_oneapi_get_backend_content_view();
#endif

  return 0;
}
