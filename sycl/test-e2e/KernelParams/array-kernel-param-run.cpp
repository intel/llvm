// This test checks kernel execution with array kernel parameters.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

constexpr size_t c_num_items = 4;
range<1> num_items{c_num_items}; // range<1>(num_items)

// Change if tests are added/removed
static int testCount = 4;
static int passCount;

template <typename T>
static bool verify_1D(const char *name, int X, T A, T A_ref) {
  int ErrCnt = 0;

  for (int i = 0; i < X; i++) {
    if (A_ref[i] != A[i]) {
      if (++ErrCnt < 10) {
        std::cout << name << " mismatch at " << i << ". Expected " << A_ref[i]
                  << " result is " << A[i] << "\n";
      }
    }
  }

  if (ErrCnt == 0) {
    return true;
  }
  std::cout << "  Failed. Failure rate: " << ErrCnt << "/" << X << "("
            << ErrCnt / (float)X * 100.f << "%)\n";
  return false;
}

template <typename T> void init(T &A, int value, int increment) {
  for (int i = 0; i < c_num_items; i++) {
    A[i] = value;
    value += increment;
  }
}

bool test_one_array(queue &myQueue) {
  int input1[c_num_items][c_num_items];
  int input2[c_num_items][c_num_items][c_num_items];
  int output[c_num_items];
  int ref[c_num_items];
  int value1 = 0;
  int value2 = 0;
  int increment = 1;
  for (int i = 0; i < c_num_items; i++) {
    for (int j = 0; j < c_num_items; j++) {
      for (int k = 0; k < c_num_items; k++) {
        input2[i][j][k] = value1;
        value1 += increment;
      }
      input1[i][j] = value2;
      value2 += increment;
    }
  }
  init(output, 511, 1);
  init(ref, 37, 2);

  auto out_buffer = buffer<int, 1>(&output[0], num_items);

  myQueue.submit([&](handler &cgh) {
    auto output_accessor = out_buffer.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class one_array>(num_items, [=](sycl::id<1> index) {
      output_accessor[index] = input1[0][index] + input2[2][1][index] + 1;
    });
  });
  const auto HostAccessor = out_buffer.get_host_access();

  return verify_1D<int *>("One array", c_num_items, output, ref);
}

bool test_two_arrays(queue &myQueue) {
  int input1[c_num_items];
  int input2[c_num_items];
  int output[c_num_items];
  int ref[c_num_items];
  init(input1, 1, 1);
  init(input2, 22, 1);
  init(ref, 23, 2);

  auto out_buffer = buffer<int, 1>(&output[0], num_items);

  myQueue.submit([&](handler &cgh) {
    auto output_accessor = out_buffer.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class two_arrays>(num_items, [=](sycl::id<1> index) {
      output_accessor[index] = input1[index] + input2[index];
    });
  });
  const auto HostAccessor = out_buffer.get_host_access();

  return verify_1D<int *>("Two arrays", c_num_items, output, ref);
}

bool test_accessor_arrays_1(queue &myQueue) {
  std::array<int, c_num_items> input1;
  std::array<int, c_num_items> input2;
  int input3[c_num_items];
  int input4[c_num_items];
  std::array<int, c_num_items> ref;
  init(input1, 1, 1);
  init(input2, 22, 1);
  init(input3, 5, 1);
  init(input4, -7, 1);
  init(ref, 22, 3);

  auto in_buffer1 = buffer<int, 1>(input1.data(), num_items);
  auto in_buffer2 = buffer<int, 1>(input2.data(), num_items);

  myQueue.submit([&](handler &cgh) {
    using Accessor =
        accessor<int, 1, access::mode::read_write, access::target::device>;
    Accessor a[2] = {
        in_buffer1.get_access<access::mode::read_write>(cgh),
        in_buffer2.get_access<access::mode::read_write>(cgh),
    };

    cgh.parallel_for<class accessor_arrays_1>(
        num_items, [=](sycl::id<1> index) {
          a[0][index] = a[1][index] + input3[index] + input4[index] + 2;
        });
  });
  const auto HostAccessor = in_buffer1.get_host_access();

  return verify_1D<std::array<int, c_num_items>>("Accessor arrays 1",
                                                 c_num_items, input1, ref);
}

bool test_accessor_arrays_2(queue &myQueue) {
  std::array<int, c_num_items> input1;
  std::array<int, c_num_items> input2;
  std::array<int, c_num_items> output;
  std::array<int, c_num_items> ref;
  init(input1, 1, 1);
  init(input2, 22, 1);
  init(ref, 23, 2);

  auto in_buffer1 = buffer<int, 1>(input1.data(), num_items);
  auto in_buffer2 = buffer<int, 1>(input2.data(), num_items);
  auto out_buffer = buffer<int, 1>(output.data(), num_items);

  myQueue.submit([&](handler &cgh) {
    using Accessor =
        accessor<int, 1, access::mode::read_write, access::target::device>;
    Accessor a[4] = {in_buffer1.get_access<access::mode::read_write>(cgh),
                     in_buffer2.get_access<access::mode::read_write>(cgh),
                     in_buffer1.get_access<access::mode::read_write>(cgh),
                     in_buffer2.get_access<access::mode::read_write>(cgh)};
    auto output_accessor = out_buffer.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class accessor_arrays_2>(
        num_items, [=](sycl::id<1> index) {
          output_accessor[index] = a[0][index] + a[3][index];
        });
  });
  const auto HostAccessor = out_buffer.get_host_access();

  return verify_1D<std::array<int, c_num_items>>("Accessor arrays 2",
                                                 c_num_items, output, ref);
}

bool run_tests() {
  queue Q([](exception_list L) {
    for (auto ep : L) {
      try {
        std::rethrow_exception(ep);
      } catch (std::exception &E) {
        std::cout << "*** std exception caught:\n";
        std::cout << E.what();
      } catch (sycl::exception const &E1) {
        std::cout << "*** SYCL exception caught:\n";
        std::cout << E1.what();
      }
    }
  });

  passCount = 0;
  if (test_one_array(Q)) {
    ++passCount;
  }
  if (test_two_arrays(Q)) {
    ++passCount;
  }
  if (test_accessor_arrays_1(Q)) {
    ++passCount;
  }
  if (test_accessor_arrays_2(Q)) {
    ++passCount;
  }

  auto D = Q.get_device();
  const char *devType = D.is_cpu() ? "CPU" : "GPU";
  std::cout << passCount << " of " << testCount << " tests passed on "
            << devType << "\n";

  return (testCount == passCount);
}

int main(int argc, char *argv[]) {
  bool passed = true;
  default_selector selector{};
  auto D = selector.select_device();
  const char *devType = D.is_cpu() ? "CPU" : "GPU";
  std::cout << "Running on device " << devType << " ("
            << D.get_info<sycl::info::device::name>() << ")\n";
  try {
    passed &= run_tests();
  } catch (exception e) {
    std::cout << e.what();
  }

  if (!passed) {
    std::cout << "FAILED\n";
    return 1;
  }
  std::cout << "PASSED\n";
  return 0;
}
