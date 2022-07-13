// This test checks kernel execution with array parameters inside structs.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

constexpr size_t c_num_items = 10;
range<1> num_items{c_num_items}; // range<1>(num_items)

// Change if tests are added/removed
static int testCount = 2;
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

bool test_accessor_array_in_struct(queue &myQueue) {
  std::array<int, c_num_items> input1;
  std::array<int, c_num_items> input2;
  std::array<int, c_num_items> output;
  std::array<int, c_num_items> ref;
  init(input1, 1, 1);
  init(input2, 22, 1);
  init(ref, 35, 2);

  auto in_buffer1 = buffer<int, 1>(input1.data(), num_items);
  auto in_buffer2 = buffer<int, 1>(input2.data(), num_items);
  auto out_buffer = buffer<int, 1>(output.data(), num_items);

  myQueue.submit([&](handler &cgh) {
    using Accessor =
        accessor<int, 1, access::mode::read_write, access::target::device>;

    struct S {
      int w;
      int x;
      Accessor a[2];
      int y;
      int z;
    } S = {3,
           3,
           {in_buffer1.get_access<access::mode::read_write>(cgh),
            in_buffer2.get_access<access::mode::read_write>(cgh)},
           7,
           7};
    auto output_accessor = out_buffer.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class accessor_array_in_struct>(
        num_items, [=](cl::sycl::id<1> index) {
          S.a[0][index]++;
          S.a[1][index]++;
          output_accessor[index] = S.a[0][index] + S.a[1][index] + S.x + S.y;
        });
  });
  const auto HostAccessor =
      out_buffer.get_access<cl::sycl::access::mode::read>();

  return verify_1D("Accessor array in struct", c_num_items, output, ref);
}

template <typename T> struct S { T a[c_num_items]; };
bool test_templated_array_in_struct(queue &myQueue) {
  std::array<int, c_num_items> output;
  std::array<int, c_num_items> ref;
  init(ref, 3, 3);

  auto out_buffer = buffer<int, 1>(output.data(), num_items);

  S<int> sint;
  S<long long> sll;
  init(sint.a, 1, 1);
  init(sll.a, 2, 2);

  myQueue.submit([&](handler &cgh) {
    using Accessor =
        accessor<int, 1, access::mode::read_write, access::target::device>;
    auto output_accessor = out_buffer.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class templated_array_in_struct>(
        num_items, [=](cl::sycl::id<1> index) {
          output_accessor[index] = sint.a[index] + sll.a[index];
        });
  });
  const auto HostAccessor =
      out_buffer.get_access<cl::sycl::access::mode::read>();

  return verify_1D("Templated array in struct", c_num_items, output, ref);
}

bool run_tests() {
  queue Q([](exception_list L) {
    for (auto ep : L) {
      try {
        std::rethrow_exception(ep);
      } catch (std::exception &E) {
        std::cout << "*** std exception caught:\n";
        std::cout << E.what();
      } catch (cl::sycl::exception const &E1) {
        std::cout << "*** SYCL exception caught:\n";
        std::cout << E1.what();
      }
    }
  });

  passCount = 0;
  if (test_accessor_array_in_struct(Q)) {
    ++passCount;
  }
  if (test_templated_array_in_struct(Q)) {
    ++passCount;
  }

  auto D = Q.get_device();
  const char *devType = D.is_host() ? "Host" : D.is_cpu() ? "CPU" : "GPU";
  std::cout << passCount << " of " << testCount << " tests passed on "
            << devType << "\n";

  return (testCount == passCount);
}

int main(int argc, char *argv[]) {
  bool passed = true;
  default_selector selector{};
  auto D = selector.select_device();
  const char *devType = D.is_host() ? "Host" : D.is_cpu() ? "CPU" : "GPU";
  std::cout << "Running on device " << devType << " ("
            << D.get_info<cl::sycl::info::device::name>() << ")\n";
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
