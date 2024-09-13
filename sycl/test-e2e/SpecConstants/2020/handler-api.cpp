// This test is intended to check basic operations with SYCL 2020 specialization
// constants using sycl::handler and sycl::kernel_handler APIs:
// - test that specialization constants can be accessed in kernel and they
//   have their default values if `set_specialization_constants` wasn't called
// - test that specialization constant values can be set and retrieved within
//   command group scope
// - test that specialization constant values can be set within command group
//   scope and correctly retrieved within a kernel

// RUN: %{build} -Wno-error=unused-command-line-argument -o %t.out -fsycl-dead-args-optimization
// RUN: %{run} %t.out

#include <cstdlib>
#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>

#include "common.hpp"

constexpr sycl::specialization_id<int> int_id;
constexpr sycl::specialization_id<int> int_id2(2);
constexpr sycl::specialization_id<float> float_id(3.14);
constexpr sycl::specialization_id<custom_type> custom_type_id;

class TestDefaultValuesKernel;
class EmptyKernel;
class TestSetAndGetOnDevice;

bool test_default_values(sycl::queue q);
bool test_set_and_get_on_host(sycl::queue q);
bool test_set_and_get_on_device(sycl::queue q);

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

  return 0;
};

bool test_default_values(sycl::queue q) {
  sycl::buffer<int> int_buffer(1);
  sycl::buffer<int> int_buffer2(1);
  sycl::buffer<float> float_buffer(1);
  sycl::buffer<custom_type> custom_type_buffer(1);

  q.submit([&](sycl::handler &cgh) {
    auto int_acc = int_buffer.get_access<sycl::access::mode::write>(cgh);
    auto int_acc2 = int_buffer2.get_access<sycl::access::mode::write>(cgh);
    auto float_acc = float_buffer.get_access<sycl::access::mode::write>(cgh);
    auto custom_type_acc =
        custom_type_buffer.get_access<sycl::access::mode::write>(cgh);
    cgh.single_task<TestDefaultValuesKernel>([=](sycl::kernel_handler kh) {
      int_acc[0] = kh.get_specialization_constant<int_id>();
      int_acc2[0] = kh.get_specialization_constant<int_id2>();
      float_acc[0] = kh.get_specialization_constant<float_id>();
      custom_type_acc[0] = kh.get_specialization_constant<custom_type_id>();
    });
  });

  sycl::host_accessor int_acc(int_buffer, sycl::read_only);
  if (!check_value(
          0, int_acc[0],
          "integer specialization constant (defined without default value)"))
    return false;

  sycl::host_accessor int_acc2(int_buffer2, sycl::read_only);
  if (!check_value(2, int_acc2[0], "integer specialization constant"))
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
  unsigned errors = 0;
  q.submit([&](sycl::handler &cgh) {
    if (!check_value(
            0, cgh.get_specialization_constant<int_id>(),
            "integer specializaiton constant before setting any value"))
      ++errors;

    if (!check_value(3.14f, cgh.get_specialization_constant<float_id>(),
                     "float specializaiton constant before setting any value"))
      ++errors;

    custom_type custom_type_ref;
    if (!check_value(
            custom_type_ref, cgh.get_specialization_constant<custom_type_id>(),
            "custom_type specializaiton constant before setting any value"))
      ++errors;

    int new_int_value = 8;
    float new_float_value = 3.0f;
    custom_type new_custom_type_value('b', 1.0f, 12);
    cgh.set_specialization_constant<int_id>(new_int_value);
    cgh.set_specialization_constant<float_id>(new_float_value);
    cgh.set_specialization_constant<custom_type_id>(new_custom_type_value);

    if (!check_value(
            new_int_value, cgh.get_specialization_constant<int_id>(),
            "integer specializaiton constant after setting a new value"))
      ++errors;

    if (!check_value(new_float_value,
                     cgh.get_specialization_constant<float_id>(),
                     "float specializaiton constant after setting a new value"))
      ++errors;

    if (!check_value(
            new_custom_type_value,
            cgh.get_specialization_constant<custom_type_id>(),
            "custom_type specializaiton constant after setting a new value"))
      ++errors;

    cgh.single_task<EmptyKernel>([=]() {});
  });

  return errors == 0;
}

bool test_set_and_get_on_device(sycl::queue q) {
  sycl::buffer<int> int_buffer(1);
  sycl::buffer<int> int_buffer2(1);
  sycl::buffer<float> float_buffer(1);
  sycl::buffer<custom_type> custom_type_buffer(1);

  int new_int_value = 8;
  int new_int_value2 = 0;
  float new_float_value = 3.0f;
  custom_type new_custom_type_value('b', 1.0f, 12);

  q.submit([&](sycl::handler &cgh) {
    auto int_acc = int_buffer.get_access<sycl::access::mode::write>(cgh);
    auto int_acc2 = int_buffer2.get_access<sycl::access::mode::write>(cgh);
    auto float_acc = float_buffer.get_access<sycl::access::mode::write>(cgh);
    auto custom_type_acc =
        custom_type_buffer.get_access<sycl::access::mode::write>(cgh);

    cgh.set_specialization_constant<int_id>(new_int_value);
    cgh.set_specialization_constant<int_id2>(new_int_value2);
    cgh.set_specialization_constant<float_id>(new_float_value);
    cgh.set_specialization_constant<custom_type_id>(new_custom_type_value);

    cgh.single_task<TestSetAndGetOnDevice>([=](sycl::kernel_handler kh) {
      int_acc[0] = kh.get_specialization_constant<int_id>();
      int_acc2[0] = kh.get_specialization_constant<int_id2>();
      float_acc[0] = kh.get_specialization_constant<float_id>();
      custom_type_acc[0] = kh.get_specialization_constant<custom_type_id>();
    });
  });

  sycl::host_accessor int_acc(int_buffer, sycl::read_only);
  if (!check_value(new_int_value, int_acc[0],
                   "integer specialization constant"))
    return false;

  sycl::host_accessor int_acc2(int_buffer2, sycl::read_only);
  if (!check_value(new_int_value2, int_acc2[0],
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
