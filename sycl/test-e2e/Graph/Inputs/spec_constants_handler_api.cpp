// This test is intended to check basic operations with SYCL 2020 specialization
// constants using Graph and sycl::handler and sycl::kernel_handler APIs
// This test was taken from `SpecConstants/2020/handler-api.cpp`.
// Variable names have been changed to meet PascalCase naming convention
// requirements.

#include "../graph_common.hpp"

#include <sycl/specialization_id.hpp>

constexpr sycl::specialization_id<int> IntId;
constexpr sycl::specialization_id<int> IntId2(2);
constexpr sycl::specialization_id<float> FloatId(3.14);

class TestDefaultValuesKernel;
class EmptyKernel;
class TestSetAndGetOnDevice;

bool test_default_values(sycl::queue Queue);
bool test_set_and_get_on_host(sycl::queue Queue);
bool test_set_and_get_on_device(sycl::queue Queue);

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
  sycl::buffer<int> IntBuffer(1);
  IntBuffer.set_write_back(false);
  sycl::buffer<int> IntBuffer2(1);
  IntBuffer2.set_write_back(false);
  sycl::buffer<float> FloatBuffer(1);
  FloatBuffer.set_write_back(false);

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(Graph, Queue, ([&](sycl::handler &CGH) {
               auto IntAcc =
                   IntBuffer.get_access<sycl::access::mode::write>(CGH);
               auto IntAcc2 =
                   IntBuffer2.get_access<sycl::access::mode::write>(CGH);
               auto FloatAcc =
                   FloatBuffer.get_access<sycl::access::mode::write>(CGH);

               CGH.single_task<TestDefaultValuesKernel>(
                   [=](sycl::kernel_handler KH) {
                     IntAcc[0] = KH.get_specialization_constant<IntId>();
                     IntAcc2[0] = KH.get_specialization_constant<IntId2>();
                     FloatAcc[0] = KH.get_specialization_constant<FloatId>();
                   });
             }));

    auto GraphExec = Graph.finalize();

    Queue.submit([&](handler &CGH) { CGH.ext_oneapi_graph(GraphExec); });
    Queue.wait_and_throw();
  }

  unsigned Errors = 0;
  sycl::host_accessor IntAcc(IntBuffer, sycl::read_only);
  if (!check_value(
          0, IntAcc[0],
          "integer specialization constant (defined without default value)"))
    Errors++;

  sycl::host_accessor IntAcc2(IntBuffer2, sycl::read_only);
  if (!check_value(2, IntAcc2[0], "integer specialization constant"))
    Errors++;

  sycl::host_accessor FloatAcc(FloatBuffer, sycl::read_only);
  if (!check_value(3.14f, FloatAcc[0], "float specialization constant"))
    Errors++;

  return Errors == 0;
}

bool test_set_and_get_on_host(sycl::queue Queue) {
  unsigned Errors = 0;

  exp_ext::command_graph Graph{
      Queue.get_context(),
      Queue.get_device(),
      {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

  add_node(
      Graph, Queue, ([&](sycl::handler &CGH) {
        if (!check_value(
                0, CGH.get_specialization_constant<IntId>(),
                "integer specializaiton constant before setting any value"))
          ++Errors;

        if (!check_value(
                3.14f, CGH.get_specialization_constant<FloatId>(),
                "float specializaiton constant before setting any value"))
          ++Errors;

        int NewIntValue = 8;
        float NewFloatValue = 3.0f;
        CGH.set_specialization_constant<IntId>(NewIntValue);
        CGH.set_specialization_constant<FloatId>(NewFloatValue);

        if (!check_value(
                NewIntValue, CGH.get_specialization_constant<IntId>(),
                "integer specializaiton constant after setting a new value"))
          ++Errors;

        if (!check_value(
                NewFloatValue, CGH.get_specialization_constant<FloatId>(),
                "float specializaiton constant after setting a new value"))
          ++Errors;

        CGH.single_task<EmptyKernel>([=]() {});
      }));

  return Errors == 0;
}

bool test_set_and_get_on_device(sycl::queue Queue) {
  sycl::buffer<int> IntBuffer(1);
  IntBuffer.set_write_back(false);
  sycl::buffer<int> IntBuffer2(1);
  IntBuffer2.set_write_back(false);
  sycl::buffer<float> FloatBuffer(1);
  FloatBuffer.set_write_back(false);

  int NewIntValue = 8;
  int NewIntValue2 = 0;
  float NewFloatValue = 3.0f;

  {
    exp_ext::command_graph Graph{
        Queue.get_context(),
        Queue.get_device(),
        {exp_ext::property::graph::assume_buffer_outlives_graph{}}};

    add_node(
        Graph, Queue, ([&](sycl::handler &CGH) {
          auto IntAcc = IntBuffer.get_access<sycl::access::mode::write>(CGH);
          auto IntAcc2 = IntBuffer2.get_access<sycl::access::mode::write>(CGH);
          auto FloatAcc =
              FloatBuffer.get_access<sycl::access::mode::write>(CGH);

          CGH.set_specialization_constant<IntId>(NewIntValue);
          CGH.set_specialization_constant<IntId2>(NewIntValue2);
          CGH.set_specialization_constant<FloatId>(NewFloatValue);

          CGH.single_task<TestSetAndGetOnDevice>([=](sycl::kernel_handler KH) {
            IntAcc[0] = KH.get_specialization_constant<IntId>();
            IntAcc2[0] = KH.get_specialization_constant<IntId2>();
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

  sycl::host_accessor IntAcc2(IntBuffer2, sycl::read_only);
  if (!check_value(NewIntValue2, IntAcc2[0], "integer specialization constant"))
    Errors++;

  sycl::host_accessor FloatAcc(FloatBuffer, sycl::read_only);
  if (!check_value(NewFloatValue, FloatAcc[0], "float specialization constant"))
    Errors++;

  return Errors == 0;
}
