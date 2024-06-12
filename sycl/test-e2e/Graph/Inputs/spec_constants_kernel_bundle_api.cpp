// This test is intended to check basic operations with SYCL 2020 specialization
// constants using Graph and sycl::kernel_bundle and sycl::kernel_handler APIs
// This test was taken from `SpecConstants/2020/kernel-bundle-api.cpp`.
// Variable names have been changed to meet PascalCase naming convention
// requirements and native constants test was removed.

#include "../graph_common.hpp"

#include <sycl/specialization_id.hpp>

constexpr sycl::specialization_id<int> IntId(2);
constexpr sycl::specialization_id<float> FloatId(3.14f);

class TestDefaultValuesKernel;
class EmptyKernel;
class TestSetAndGetOnDevice;

bool test_default_values(sycl::queue Queue);
bool test_set_and_get_on_host(sycl::queue Queue);
bool test_set_and_get_on_device(sycl::queue Queue);

int main() {
  auto ExceptionHandler = [&](sycl::exception_list Exceptions) {
    for (std::exception_ptr const &E : Exceptions) {
      try {
        std::rethrow_exception(E);
      } catch (sycl::exception const &E) {
        std::cout << "An async SYCL exception was caught: " << E.what()
                  << std::endl;
        std::exit(1);
      }
    }
  };

  queue Queue{ExceptionHandler};

  unsigned Errors = 0;
  if (!test_default_values(Queue)) {
    std::cout << "Test for default values of specialization constants failed!"
              << std::endl;
    Errors++;
  }

  if (!test_set_and_get_on_host(Queue)) {
    std::cout << "Test for set and get API on host failed!" << std::endl;
    Errors++;
  }

  if (!test_set_and_get_on_device(Queue)) {
    std::cout << "Test for set and get API on device failed!" << std::endl;
    Errors++;
  }

  return (Errors == 0) ? 0 : 1;
};

bool test_default_values(sycl::queue Queue) {
  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(
          Queue.get_context())) {
    std::cout << "Cannot obtain kernel_bundle in input state, skipping default "
                 "values test"
              << std::endl;
    return true;
  }

  sycl::buffer<int> IntBuffer(1);
  IntBuffer.set_write_back(false);
  sycl::buffer<float> FloatBuffer(1);
  FloatBuffer.set_write_back(false);

  auto InputBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Queue.get_context());
  auto ExecBundle = sycl::build(InputBundle);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(Graph, Queue, ([&](sycl::handler &CGH) {
               CGH.use_kernel_bundle(ExecBundle);
               auto IntAcc =
                   IntBuffer.get_access<sycl::access::mode::write>(CGH);
               auto FloatAcc =
                   FloatBuffer.get_access<sycl::access::mode::write>(CGH);

               CGH.single_task<TestDefaultValuesKernel>(
                   [=](sycl::kernel_handler KH) {
                     IntAcc[0] = KH.get_specialization_constant<IntId>();
                     FloatAcc[0] = KH.get_specialization_constant<FloatId>();
                   });
             }));

    auto GraphExec = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    Queue.wait_and_throw();
  }
  unsigned Errors = 0;

  sycl::host_accessor IntAcc(IntBuffer, sycl::read_only);
  if (!check_value(2, IntAcc[0], "integer specialization constant"))
    Errors++;

  sycl::host_accessor FloatAcc(FloatBuffer, sycl::read_only);
  if (!check_value(3.14f, FloatAcc[0], "float specialization constant"))
    Errors++;

  return Errors == 0;
}

bool test_set_and_get_on_host(sycl::queue Queue) {
  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(
          Queue.get_context())) {
    std::cout << "Cannot obtain kernel_bundle in input state, skipping default "
                 "values test"
              << std::endl;
    return true;
  }

  unsigned Errors = 0;

  try {
    auto InputBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Queue.get_context());

    if (!InputBundle.contains_specialization_constants()) {
      std::cout
          << "Obtained kernel_bundle is expected to contain specialization "
             "constants, but it doesn't!"
          << std::endl;
      return false;
    }

    // Check default values
    if (!check_value(
            2, InputBundle.get_specialization_constant<IntId>(),
            "integer specialization constant before setting any value"))
      ++Errors;

    if (!check_value(3.14f, InputBundle.get_specialization_constant<FloatId>(),
                     "float specialization constant before setting any value"))
      ++Errors;

    // Update values
    int NewIntValue = 42;
    float NewFloatValue = 3.0f;

    InputBundle.set_specialization_constant<IntId>(NewIntValue);
    InputBundle.set_specialization_constant<FloatId>(NewFloatValue);

    // And re-check them again
    if (!check_value(
            NewIntValue, InputBundle.get_specialization_constant<IntId>(),
            "integer specialization constant after setting a new value"))
      ++Errors;

    if (!check_value(NewFloatValue,
                     InputBundle.get_specialization_constant<FloatId>(),
                     "float specialization constant after setting a value"))
      ++Errors;

    // Let's try to build the bundle
    auto ExecBundle = sycl::build(InputBundle);

    // And ensure that updated spec constant values are still there
    if (!check_value(NewIntValue,
                     ExecBundle.get_specialization_constant<IntId>(),
                     "integer specialization constant after build"))
      ++Errors;

    if (!check_value(NewFloatValue,
                     ExecBundle.get_specialization_constant<FloatId>(),
                     "float specialization constant after build"))
      ++Errors;
  } catch (sycl::exception &e) {
  }

  return Errors == 0;
}

bool test_set_and_get_on_device(sycl::queue Queue) {
  sycl::buffer<int> IntBuffer(1);
  IntBuffer.set_write_back(false);
  sycl::buffer<float> FloatBuffer(1);
  FloatBuffer.set_write_back(false);

  int NewIntValue = 42;
  float NewFloatValue = 2.0f;

  auto InputBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Queue.get_context());
  InputBundle.set_specialization_constant<IntId>(NewIntValue);
  InputBundle.set_specialization_constant<FloatId>(NewFloatValue);
  auto ExecBundle = sycl::build(InputBundle);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(
        Graph, Queue, ([&](sycl::handler &CGH) {
          CGH.use_kernel_bundle(ExecBundle);
          auto IntAcc = IntBuffer.get_access<sycl::access::mode::write>(CGH);
          auto FloatAcc =
              FloatBuffer.get_access<sycl::access::mode::write>(CGH);

          CGH.single_task<TestSetAndGetOnDevice>([=](sycl::kernel_handler KH) {
            IntAcc[0] = KH.get_specialization_constant<IntId>();
            FloatAcc[0] = KH.get_specialization_constant<FloatId>();
          });
        }));

    auto GraphExec = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    Queue.wait_and_throw();
  }

  unsigned Errors = 0;
  sycl::host_accessor IntAcc(IntBuffer, sycl::read_only);
  if (!check_value(NewIntValue, IntAcc[0], "integer specialization constant"))
    Errors++;

  sycl::host_accessor FloatAcc(FloatBuffer, sycl::read_only);
  if (!check_value(NewFloatValue, FloatAcc[0], "float specialization constant"))
    Errors++;

  return Errors == 0;
}
