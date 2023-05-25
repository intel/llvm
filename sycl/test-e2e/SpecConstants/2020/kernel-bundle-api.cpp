// This test is intended to check basic operations with SYCL 2020 specialization
// constants using sycl::kernel_bundle and sycl::kernel_handler APIs:
// - test that specialization constants can be accessed in kernel and they
//   have their default values if `set_specialization_constants` wasn't called
// - test that specialization constant values can be set and retrieved through
//   kernel_bundle APIs on host
// - test that specialization constant values can be set through kernel_bundle
//   API and correctly retrieved within a kernel
//
// RUN: %{build} -o %t.out -fsycl-dead-args-optimization
// RUN: %{run} %t.out
// FIXME: ACC devices use emulation path, which is not yet supported
// UNSUPPORTED: accelerator
// UNSUPPORTED: hip

#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>

#include "common.hpp"

constexpr sycl::specialization_id<int> int_id;
constexpr sycl::specialization_id<float> float_id(3.14f);
constexpr sycl::specialization_id<custom_type> custom_type_id;

class TestDefaultValuesKernel;
class EmptyKernel;
class TestSetAndGetOnDevice;

bool test_default_values(sycl::queue q);
bool test_set_and_get_on_host(sycl::queue q);
bool test_set_and_get_on_device(sycl::queue q);
bool test_native_specialization_constant(sycl::queue q);

int main() {
  auto exception_handler = [&](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "An async SYCL exception was caught: " << e.what()
                  << std::endl;
        std::exit(1);
      }
    }
  };

  sycl::queue q(exception_handler);

  if (!test_default_values(q)) {
    std::cout << "Test for default values of specialization constants failed!"
              << std::endl;
    return 1;
  }

  if (!test_set_and_get_on_host(q)) {
    std::cout << "Test for set and get API on host failed!" << std::endl;
    return 1;
  }

  if (!test_set_and_get_on_device(q)) {
    std::cout << "Test for set and get API on device failed!" << std::endl;
    return 1;
  }

  if (!test_native_specialization_constant(q)) {
    std::cout << "Test for native specialization constants failed!"
              << std::endl;
    return 1;
  }

  return 0;
};

bool test_default_values(sycl::queue q) {
  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(q.get_context())) {
    std::cout << "Cannot obtain kernel_bundle in input state, skipping default "
                 "values test"
              << std::endl;
    // TODO: check that online_compielr aspec is not available
    return true;
  }

  sycl::buffer<int> int_buffer(1);
  sycl::buffer<float> float_buffer(1);
  sycl::buffer<custom_type> custom_type_buffer(1);

  auto input_bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(q.get_context());
  auto exec_bundle = sycl::build(input_bundle);

  q.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(exec_bundle);
    auto int_acc = int_buffer.get_access<sycl::access::mode::write>(cgh);
    auto float_acc = float_buffer.get_access<sycl::access::mode::write>(cgh);
    auto custom_type_acc =
        custom_type_buffer.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<TestDefaultValuesKernel>([=](sycl::kernel_handler kh) {
      int_acc[0] = kh.get_specialization_constant<int_id>();
      float_acc[0] = kh.get_specialization_constant<float_id>();
      custom_type_acc[0] = kh.get_specialization_constant<custom_type_id>();
    });
  });

  sycl::host_accessor int_acc(int_buffer, sycl::read_only);
  if (!check_value(
          0, int_acc[0],
          "integer specialization constant (defined without default value)"))
    return false;

  sycl::host_accessor float_acc(float_buffer, sycl::read_only);
  if (!check_value(3.14f, float_acc[0], "float specialization constant"))
    return false;

  sycl::host_accessor custom_type_acc(custom_type_buffer, sycl::read_only);
  const custom_type custom_type_ref;
  if (!check_value(custom_type_ref, custom_type_acc[0],
                   "custom_type specialization constant"))
    return false;

  return true;
}

bool test_set_and_get_on_host(sycl::queue q) {
  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(q.get_context())) {
    std::cout << "Cannot obtain kernel_bundle in input state, skipping default "
                 "values test"
              << std::endl;
    // TODO: check that online_compielr aspec is not available
    return true;
  }

  unsigned errors = 0;

  try {
    auto input_bundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(q.get_context());

    if (!input_bundle.contains_specialization_constants()) {
      std::cout
          << "Obtained kernel_bundle is expected to contain specialization "
             "constants, but it doesn't!"
          << std::endl;
      return false;
    }

    // Check default values
    if (!check_value(
            0, input_bundle.get_specialization_constant<int_id>(),
            "integer specializaiton constant before setting any value"))
      ++errors;

    if (!check_value(3.14f,
                     input_bundle.get_specialization_constant<float_id>(),
                     "float specializaiton constant before setting any value"))
      ++errors;

    custom_type custom_type_ref;
    if (!check_value(
            custom_type_ref,
            input_bundle.get_specialization_constant<custom_type_id>(),
            "custom_type specializaiton constant before setting any value"))
      ++errors;

    // Update values
    int new_int_value = 42;
    float new_float_value = 3.0f;
    custom_type new_custom_type_value('b', 1.0f, 12);

    input_bundle.set_specialization_constant<int_id>(new_int_value);
    input_bundle.set_specialization_constant<float_id>(new_float_value);
    input_bundle.set_specialization_constant<custom_type_id>(
        new_custom_type_value);

    // And re-check them again
    if (!check_value(
            new_int_value, input_bundle.get_specialization_constant<int_id>(),
            "integer specializaiton constant after setting a new value"))
      ++errors;

    if (!check_value(new_float_value,
                     input_bundle.get_specialization_constant<float_id>(),
                     "float specializaiton constant after setting a value"))
      ++errors;

    if (!check_value(
            new_custom_type_value,
            input_bundle.get_specialization_constant<custom_type_id>(),
            "custom_type specializaiton constant after setting a new value"))
      ++errors;

    // Let's try to build the bundle
    auto exec_bundle = sycl::build(input_bundle);

    // And ensure that updated spec constant values are still there
    if (!check_value(new_int_value,
                     exec_bundle.get_specialization_constant<int_id>(),
                     "integer specializaiton constant after build"))
      ++errors;

    if (!check_value(new_float_value,
                     exec_bundle.get_specialization_constant<float_id>(),
                     "float specializaiton constant after build"))
      ++errors;

    if (!check_value(new_custom_type_value,
                     exec_bundle.get_specialization_constant<custom_type_id>(),
                     "custom_type specializaiton constant after build"))
      ++errors;
  } catch (sycl::exception &e) {
  }

  return 0 == errors;
}

bool test_set_and_get_on_device(sycl::queue q) {
  sycl::buffer<int> int_buffer(1);
  sycl::buffer<float> float_buffer(1);
  sycl::buffer<custom_type> custom_type_buffer(1);

  int new_int_value = 42;
  float new_float_value = 3.0f;
  custom_type new_custom_type_value('b', 1.0f, 12);

  auto input_bundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(q.get_context());
  input_bundle.set_specialization_constant<int_id>(new_int_value);
  input_bundle.set_specialization_constant<float_id>(new_float_value);
  input_bundle.set_specialization_constant<custom_type_id>(
      new_custom_type_value);
  auto exec_bundle = sycl::build(input_bundle);

  q.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(exec_bundle);
    auto int_acc = int_buffer.get_access<sycl::access::mode::write>(cgh);
    auto float_acc = float_buffer.get_access<sycl::access::mode::write>(cgh);
    auto custom_type_acc =
        custom_type_buffer.get_access<sycl::access::mode::write>(cgh);

    cgh.single_task<TestSetAndGetOnDevice>([=](sycl::kernel_handler kh) {
      int_acc[0] = kh.get_specialization_constant<int_id>();
      float_acc[0] = kh.get_specialization_constant<float_id>();
      custom_type_acc[0] = kh.get_specialization_constant<custom_type_id>();
    });
  });

  sycl::host_accessor int_acc(int_buffer, sycl::read_only);
  if (!check_value(new_int_value, int_acc[0],
                   "integer specialization constant"))
    return false;

  sycl::host_accessor float_acc(float_buffer, sycl::read_only);
  if (!check_value(new_float_value, float_acc[0],
                   "float specialization constant"))
    return false;

  sycl::host_accessor custom_type_acc(custom_type_buffer, sycl::read_only);
  if (!check_value(new_custom_type_value, custom_type_acc[0],
                   "custom_type specialization constant"))
    return false;

  return true;
}

bool test_native_specialization_constant(sycl::queue q) {
  {
    q.submit([&](sycl::handler &cgh) {
      cgh.single_task<class Kernel>([=](sycl::kernel_handler h) {
        h.get_specialization_constant<int_id>();
      });
    });

    auto inputBundle =
        sycl::get_kernel_bundle<class Kernel, sycl::bundle_state::input>(
            q.get_context(), {q.get_device()});
    auto objectBundle = sycl::compile(inputBundle);
    auto execBundleViaLink = sycl::link(objectBundle);
    auto BE = q.get_backend();
    bool expected = (BE == sycl::backend::opencl ||
                     BE == sycl::backend::ext_oneapi_level_zero)
                        ? true
                        : false;
    if (!check_value(expected,
                     execBundleViaLink.native_specialization_constant(),
                     "linked bundle native specialization constant"))
      return false;
  }

  const auto always_false_selector = [](auto device_image) { return false; };
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q.get_context(), always_false_selector);
  return check_value(false, bundle.native_specialization_constant(),
                     "empty bundle native specialization constant");
}
