// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -I . -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "support.h"
#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace oneapi_exp = sycl::ext::oneapi::experimental;

auto async_handler_ = [](sycl::exception_list ex_list) {
  for (auto &ex : ex_list) {
    try {
      std::rethrow_exception(ex);
    } catch (sycl::exception &ex) {
      std::cerr << ex.what() << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
};

constexpr uint32_t items_per_work_item = 4;

struct CustomType {
  int x;
};

struct CustomFunctor {
  bool operator()(const CustomType &lhs, const CustomType &rhs) const {
    return lhs.x < rhs.x;
  }
};

template <typename T> bool check(T lhs, T rhs, float epsilon) {
  return sycl::abs(lhs - rhs) > epsilon;
}
bool check(CustomType lhs, CustomType rhs, float epsilon) {
  return sycl::abs(lhs.x - rhs.x) > epsilon;
}

template <typename T>
bool verify(T *expected, T *got, std::size_t n, float epsilon) {
  for (std::size_t i = 0; i < n; ++i) {
    if (check(expected[i], got[i], epsilon)) {
      return false;
    }
  }
  return true;
}

// forward declared classes to name kernels
template <typename... Args> class sort_over_group_kernel_name;
template <typename... Args> class joint_sort_kernel_name;
template <typename... Args> class custom_sorter_kernel_name;

// this class is needed to pass dimension value to aforementioned classes
template <int dim> class int_wrapper;

// custom sorter
template <typename Compare> struct bubble_sorter {
  Compare comp;
  size_t idx;

  template <typename Group, typename Ptr>
  void operator()(Group g, Ptr begin, Ptr end) {
    size_t n = end - begin;
    if (idx == 0)
      for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
          if (comp(begin[j], begin[i]))
            std::swap(begin[i], begin[j]);
  }
};

template <int dim> sycl::range<dim> get_range(const std::size_t local);

template <> sycl::range<1> get_range<1>(const std::size_t local) {
  return sycl::range<1>(local);
}

template <> sycl::range<2> get_range<2>(const std::size_t local) {
  return sycl::range<2>(local, 1);
}

template <> sycl::range<3> get_range<3>(const std::size_t local) {
  return sycl::range<3>(local, 1, 1);
}

template <int dim, typename T, typename Compare>
int test_sort_over_group(sycl::queue &q, std::size_t local,
                         sycl::buffer<T> &bufI1, Compare comp, int test_case) {
  auto n = bufI1.size();
  if (n > local)
    return -1;

  sycl::range<dim> local_range = get_range<dim>(local);

  std::size_t local_memory_size =
      oneapi_exp::default_sorter<>::memory_required<T>(
          sycl::memory_scope::work_group, local_range);

  if (local_memory_size >
      q.get_device().template get_info<sycl::info::device::local_mem_size>())
    std::cout << "local_memory_size = " << local_memory_size << ", available = "
              << q.get_device()
                     .template get_info<sycl::info::device::local_mem_size>()
              << std::endl;
  q.submit([&](sycl::handler &h) {
     auto aI1 = sycl::accessor(bufI1, h);
     sycl::accessor<std::byte, 1, sycl::access_mode::read_write,
                    sycl::access::target::local>
         scratch({local_memory_size}, h);

     h.parallel_for<sort_over_group_kernel_name<int_wrapper<dim>, T, Compare>>(
         sycl::nd_range<dim>(local_range, local_range),
         [=](sycl::nd_item<dim> id) {
           scratch[0] = std::byte{};
           auto local_id = id.get_local_linear_id();
           switch (test_case) {
           case 0:
             if constexpr (std::is_same_v<Compare, std::less<T>> &&
                           !std::is_same_v<T, CustomType>)
               aI1[local_id] = oneapi_exp::sort_over_group(
                   oneapi_exp::group_with_scratchpad(
                       id.get_group(),
                       sycl::span{&scratch[0], local_memory_size}),
                   aI1[local_id]);
             break;
           case 1:
             aI1[local_id] = oneapi_exp::sort_over_group(
                 oneapi_exp::group_with_scratchpad(
                     id.get_group(),
                     sycl::span{&scratch[0], local_memory_size}),
                 aI1[local_id], comp);
             break;
           case 2:
             aI1[local_id] = oneapi_exp::sort_over_group(
                 id.get_group(), aI1[local_id],
                 oneapi_exp::default_sorter<Compare>(
                     sycl::span{&scratch[0], local_memory_size}));
             break;
           }
         });
   }).wait_and_throw();
  return 1;
}

template <typename T, typename Compare>
int test_joint_sort(sycl::queue &q, std::size_t n_items, std::size_t local,
                    sycl::buffer<T> &bufI1, Compare comp, int test_case) {
  auto n = bufI1.size();
  auto n_groups = (n - 1) / n_items + 1;

  std::size_t local_memory_size =
      oneapi_exp::default_sorter<>::memory_required<T>(
          sycl::memory_scope::work_group, n);
  if (local_memory_size >
      q.get_device().template get_info<sycl::info::device::local_mem_size>())
    std::cout << "local_memory_size = " << local_memory_size << ", available = "
              << q.get_device()
                     .template get_info<sycl::info::device::local_mem_size>()
              << std::endl;
  q.submit([&](sycl::handler &h) {
     auto aI1 = sycl::accessor(bufI1, h);
     sycl::accessor<std::byte, 1, sycl::access_mode::read_write,
                    sycl::access::target::local>
         scratch({local_memory_size}, h);

     h.parallel_for<joint_sort_kernel_name<T, Compare>>(
         sycl::nd_range<1>{{n_groups * local}, {local}},
         [=](sycl::nd_item<1> id) {
           auto group_id = id.get_group(0);
           auto ptr_keys = &aI1[group_id * n_items];
           //  Replacing the line above with the line below also works
           //  auto ptr_keys = aI1.get_pointer() + group_id * n_items;

           scratch[0] = std::byte{};
           switch (test_case) {
           case 0:
             if constexpr (std::is_same_v<Compare, std::less<T>> &&
                           !std::is_same_v<T, CustomType>)
               oneapi_exp::joint_sort(
                   oneapi_exp::group_with_scratchpad(
                       id.get_group(),
                       sycl::span{&scratch[0], local_memory_size}),
                   ptr_keys,
                   ptr_keys + sycl::min(n_items, n - group_id * n_items));
             break;
           case 1:
             oneapi_exp::joint_sort(
                 oneapi_exp::group_with_scratchpad(
                     id.get_group(),
                     sycl::span{&scratch[0], local_memory_size}),
                 ptr_keys,
                 ptr_keys + sycl::min(n_items, n - group_id * n_items), comp);
             break;
           case 2:
             oneapi_exp::joint_sort(
                 id.get_group(), ptr_keys,
                 ptr_keys + sycl::min(n_items, n - group_id * n_items),
                 oneapi_exp::default_sorter<Compare>(
                     sycl::span{&scratch[0], local_memory_size}));
             break;
           }
         });
   }).wait_and_throw();
  return n_groups;
}

template <typename T, typename Compare>
int test_custom_sorter(sycl::queue &q, sycl::buffer<T> &bufI1, Compare comp) {
  std::size_t local = 4;
  auto n = bufI1.size();
  if (n > local)
    return -1;
  local = std::min(local, n);

  q.submit([&](sycl::handler &h) {
     auto aI1 = sycl::accessor(bufI1, h);

     h.parallel_for<custom_sorter_kernel_name<T, Compare>>(
         sycl::nd_range<2>({local, 1}, {local, 1}), [=](sycl::nd_item<2> id) {
           auto ptr = aI1.get_pointer();

           oneapi_exp::joint_sort(
               id.get_group(), ptr, ptr + n,
               bubble_sorter<Compare>{comp, id.get_local_linear_id()});
         });
   }).wait_and_throw();
  return 1;
}

template <typename T, typename Compare>
void run_sort(sycl::queue &q, std::vector<T> &in, std::size_t size,
              Compare comp, int test_case, int sort_case) {
  std::vector<T> in2(in.begin(), in.begin() + size);
  std::vector<T> expected(in.begin(), in.begin() + size);
  constexpr size_t work_size_limit = 4;
  std::size_t local = std::min(
      work_size_limit,
      q.get_device()
          .template get_info<sycl::info::device::max_work_group_size>());
  local = std::min(local, size);
  auto n_items = items_per_work_item * local;

  int n_groups = 1;
  { // scope to destruct buffers
    sycl::buffer<T> bufKeys(in2.data(), size);
    {
      switch (sort_case) {
      case 0:
        // this case is just to check the compilation
        n_groups = test_sort_over_group<1>(q, local, bufKeys, comp, test_case);

        n_groups = test_sort_over_group<2>(q, local, bufKeys, comp, test_case);
        break;
      case 1:
        n_groups = test_joint_sort(q, n_items, local, bufKeys, comp, test_case);
        break;
      case 2:
        n_groups = test_custom_sorter(q, bufKeys, comp);
        break;
      }
    }
  }

  // check results
  for (int i_group = 0; i_group < n_groups; ++i_group) {
    std::sort(expected.begin() + i_group * n_items,
              expected.begin() + std::min((i_group + 1) * n_items, size), comp);
  }
  if (n_groups != -1 &&
      (test_case != 0 ||
       test_case == 0 && std::is_same_v<Compare, std::less<T>> &&
           !std::is_same_v<T, CustomType>)&&!verify(expected.data(), in2.data(),
                                                    size, 0.001f)) {
    std::cerr << "Verification failed \n";
    exit(1);
  }
}

template <typename T> struct test_sort_cases {
  template <typename Generator, typename Compare>
  void operator()(sycl::queue &q, std::size_t dataSize, Compare comp,
                  Generator generate) {
    std::vector<T> stationaryData(dataSize);
    // fill data
    for (std::size_t i = 0; i < dataSize; ++i)
      stationaryData[i] = generate(i);

    // run test
    for (int test_case = 0; test_case < 3; ++test_case) {
      for (int sort_case = 0; sort_case < 3; ++sort_case) {
        run_sort(q, stationaryData, dataSize, comp, test_case, sort_case);
      }
    }
  }
};

void test_custom_type(sycl::queue &q, std::size_t dataSize) {
  std::vector<CustomType> stationaryData(dataSize, CustomType{0});
  // fill data
  for (std::size_t i = 0; i < dataSize; ++i)
    stationaryData[i] = CustomType{int(i)};

  // run test
  for (int test_case = 0; test_case < 1; ++test_case) {
    for (int sort_case = 0; sort_case < 3; ++sort_case) {
      run_sort(q, stationaryData, dataSize, CustomFunctor{}, test_case,
               sort_case);
    }
  }
}

template <typename T, typename Compare>
void test_sort_by_comp(sycl::queue &q, std::size_t dataSize) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution((10.0), (2.0));

  T max_size = std::numeric_limits<T>::max();
  std::size_t to_fill = dataSize;
  if (dataSize > max_size)
    to_fill = max_size;

  // reversed order
  test_sort_cases<T>()(q, to_fill, Compare{},
                       [to_fill](std::size_t i) { return T(to_fill - i - 1); });
  // filled by 1
  test_sort_cases<T>()(q, dataSize, Compare{},
                       [](std::size_t) { return T(1); });
  // random distribution
  test_sort_cases<T>()(q, dataSize, Compare{},
                       [&distribution, &generator](std::size_t) {
                         return T(distribution(generator));
                       });
}

template <typename T>
void test_sort_by_type(sycl::queue &q, std::size_t dataSize) {
  test_sort_by_comp<T, std::less<T>>(q, dataSize);
  test_sort_by_comp<T, std::greater<T>>(q, dataSize);
}

int main(int argc, char *argv[]) {
  sycl::queue q(sycl::default_selector{}, async_handler_);
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  std::vector<int> sizes{1, 12, 32};

  for (int i = 0; i < sizes.size(); ++i) {
    test_sort_by_type<std::int8_t>(q, sizes[i]);
    test_sort_by_type<std::uint16_t>(q, sizes[i]);
    test_sort_by_type<std::int32_t>(q, sizes[i]);
    test_sort_by_type<std::uint32_t>(q, sizes[i]);
    test_sort_by_type<float>(q, sizes[i]);
    test_sort_by_type<sycl::half>(q, sizes[i]);
    test_sort_by_type<double>(q, sizes[i]);
    test_sort_by_type<std::size_t>(q, sizes[i]);

    test_custom_type(q, sizes[i]);
  }
  std::cout << "Test passed." << std::endl;
}
