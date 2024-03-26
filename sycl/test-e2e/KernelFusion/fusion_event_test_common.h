#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

static bool isEventComplete(sycl::event &ev) {
  return ev.get_info<info::event::command_execution_status>() ==
         info::event_command_status::complete;
}
