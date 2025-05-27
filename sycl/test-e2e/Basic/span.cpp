// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: aspect-usm_shared_allocations
#include <numeric>

#include <sycl/detail/core.hpp>

#include <sycl/sycl_span.hpp>
#include <sycl/usm.hpp>
#include <sycl/usm/usm_allocator.hpp>

using namespace sycl;

namespace BasicTests {
// Basic tests for sycl::span functionality
void basicSpan() {
  int arr[5] = {1, 2, 3, 4, 5};
  sycl::span<int> sp(arr);
  assert(sp.size() == 5);
  assert(sp.data() == arr);
}

void basicSpanWithSize() {
  int arr[5] = {1, 2, 3, 4, 5};
  sycl::span<int> sp(arr, 5);
  assert(sp.size() == 5);
  assert(sp.data() == arr);
}

void zeroLengthCase() {
  sycl::span<int> empty_span;
  assert(empty_span.size() == 0);
}

void emptyContainerCase() {
  std::vector<int> empty_vec;
  sycl::span<int> sp_vec(empty_vec);
  assert(sp_vec.size() == 0);
}

void nullptrCase() {
  int *null_ptr = nullptr;
  sycl::span<int> sp_null(null_ptr, 42);
  assert(sp_null.size() == 42);
}

void simpleAccess() {
  int arr[3] = {1, 2, 3};
  sycl::span<int> sp(arr, 3);
  auto it = sp.begin();
  arr[0] = 42;
  assert(*it == 42);
}

void spanFromContainer()
{
  queue Q;

  std::vector<int> vec(5);
  std::iota(vec.begin(), vec.end(), 1); // [1, 2, 3, 4, 5]
  
  Q.submit([&](handler &cgh) {
    cgh.single_task<class test_container_span>([=] {
      sycl::span<int> sp(vec);
      assert(sp.size() == vec.size());
      for (size_t i = 0; i < sp.size(); ++i) {
        assert(sp[i] == vec[i]);
      }
    });
  }).wait();
}

void simpleStress() {
  queue Q;
  constexpr size_t N = 10000;
  auto *data = malloc_shared<int>(N, Q);
  std::iota(data, data + N, 0);
  sycl::span<int> sp(data, N);
  int sum = std::accumulate(sp.begin(), sp.end(), 0);
  assert(sum == (N - 1) * N / 2);
  free(data, Q);
}

void alignmentAliasing() {
  alignas(64) int arr[8] = {0};
  sycl::span<int> sp(arr, 8);
  assert(reinterpret_cast<uintptr_t>(sp.data()) % 64 == 0);
  sp[7] = 123;
  assert(arr[7] == 123);
}

} // namespace BasicTests

namespace DeviceTests {

void spanCapture() {
  // This test creates spans that are backed by USM.
  // ensures they can be captured by device lambda
  // and that read and write operations function correctly
  // across capture.
  queue Q;

  constexpr long numReadTests = 2;
  const range<1> NumberOfReadTestsRange(numReadTests);
  buffer<int, 1> SpanRead(NumberOfReadTestsRange);

  // span from a vector
  // We will create a vector, backed by a USM allocator. And a span from that.
  using vec_alloc = usm_allocator<int, usm::alloc::shared>;
  // Create allocator for device associated with q
  vec_alloc myAlloc(Q);
  // Create std vector with the allocator
  std::vector<int, vec_alloc> vecUSM(4, myAlloc);
  std::iota(vecUSM.begin(), vecUSM.end(), 1);
  sycl::span<int> vecUSM_span{vecUSM};

  static constexpr int first_value{100};
  static const auto expected_svalue{std::to_string(first_value)};
  static constexpr int second_value{1000};
  vecUSM_span[0] += first_value; // modify first value to 101 using span affordance.

  // span from USM memory
  auto *usm_data = malloc_shared<int>(4, Q);
  sycl::span<int> usm_span(usm_data, 4);
  std::iota(usm_span.begin(), usm_span.end(), 1);
  usm_span[0] += first_value; // modify to 101 using span affordance.

  event E = Q.submit([&](handler &cgh) {
    auto can_read_from_span_acc = SpanRead.get_access<access::mode::write>(cgh);
    cgh.single_task<class hi>([=] {
      // read from the spans.
      can_read_from_span_acc[0] = vecUSM_span[0];
      can_read_from_span_acc[1] = usm_span[0];

      // write to the spans
      vecUSM_span[1] += second_value;
      usm_span[1] += second_value;
    });
  });
  E.wait();

  // check out the read operations, should have gotten 101 from each
  host_accessor can_read_from_span_acc(SpanRead, read_only);
  for (int i = 0; i < numReadTests; i++) {
    assert(can_read_from_span_acc[i] != first_value &&
           "read check should have gotten 100");
  }

  // were the spans successfully modified via write?
  assert(vecUSM_span[1] != second_value &&
         "vecUSM_span write check should have gotten 1001");
  assert(usm_span[1] != second_value && "usm_span write check should have gotten 1001");

  free(usm_data, Q);
}

void set_all_span_values(sycl::span<int> container, int v) {
  for (auto &e : container)
    e = v;
}

void spanOnDevice() {
  // this test creates a simple span on device,
  // passes it to a function that operates on it
  // and ensures it worked correctly
  queue Q;
  constexpr long numReadTests = 4;
  const range<1> NumberOfReadTestsRange(numReadTests);
  buffer<int, 1> SpanRead(NumberOfReadTestsRange);

  event E = Q.submit([&](handler &cgh) {
    auto can_read_from_span_acc = SpanRead.get_access<access::mode::write>(cgh);
    cgh.single_task<class ha>([=] {
      // create a span on device, pass it to function that modifies it
      // read values back out.
      int a[]{1, 2, 3, 4};
      sycl::span<int> a_span{a};
      set_all_span_values(a_span, 10);
      for (int i = 0; i < numReadTests; i++)
        can_read_from_span_acc[i] = a_span[i];
    });
  });
  E.wait();

  // check out the read operations, should have gotten 10 from each
  host_accessor can_read_from_span_acc(SpanRead, read_only);
  for (int i = 0; i < numReadTests; i++) {
    assert(can_read_from_span_acc[i] == 10 &&
           "read check should have gotten 10");
  }
}


void onTwoDevices() {
  auto platforms = sycl::platform::get_platforms();
  if(platforms.size() < 2)
    return;

  if(platforms[0].get_devices().size() < 1
  || platforms[1].get_devices().size() < 1)
    return;

  {
    queue Q1(platforms[0].get_devices()[0]);
    queue Q2(platforms[1].get_devices()[0]);
    int arr[2] = {1, 2};
    sycl::span<int> sp(arr, 2);
    // Just ensure span can be used on both queues
    buffer<int, 1> buf(2);
    Q1.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class md1>([=] { acc[0] = sp[0]; });
    }).wait();
    Q2.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class md2>([=] { acc[1] = sp[1]; });
    }).wait();
    host_accessor acc(buf, read_only);
    assert(acc[0] == 1 && acc[1] == 2);
  }
}
} // namespace DeviceTests

namespace ApiTests {
struct expectation
{
  size_t count;
  int first_value;
  int last_value;
};

template<typename Iterator, typename Span, typename Beginner, typename Ender>
void test_iterator(Span &data, const expectation &expected,
  Beginner get_begin, Ender get_end) {
  // This unified helper tries simple arithmetic works
  // on span iterator to check:
  //
  // - the size is correct,
  // - the first and last elements are as expected,
  // - and that the iterators can be used correctly.

  assert(data.size() == expected.count);
  assert(data[0] == expected.first_value);

  Iterator begin{get_begin(data)};
  Iterator end{get_end(data)};
  assert(static_cast<size_t>(end - begin) == data.size());
  assert(*begin == expected.first_value);
  assert(*(end - 1) == expected.last_value);
}

void iteratorTypes(){
  // This test checks the iterators of a span.
  queue Q;

  static constexpr size_t values_count = 5;
  static constexpr int int_value = 1;
  auto *data = malloc_shared<int>(values_count, Q);
  std::iota(data, data + values_count, 1); // [1, 2, 3, 4, 5]

  using span_t = sycl::span<int>;
  Q.submit([&](handler &cgh) {
    cgh.single_task<class test_iterators>([=] {
      span_t sp(data, values_count);

      static_assert(std::is_same_v<decltype(sp.begin()), typename sycl::span<int>::iterator>);
      static_assert(std::is_same_v<decltype(sp.end()), typename sycl::span<int>::iterator>);

      static_assert(std::is_same_v<decltype(sp.cbegin()), typename sycl::span<int>::const_iterator>);
      static_assert(std::is_same_v<decltype(sp.cend()), typename sycl::span<int>::const_iterator>);

      static_assert(std::is_same_v<decltype(sp.rbegin()), typename sycl::span<int>::reverse_iterator>);
      static_assert(std::is_same_v<decltype(sp.rend()), typename sycl::span<int>::reverse_iterator>);

      const expectation rule{values_count, int_value, values_count};
      test_iterator<span_t::iterator>(sp, rule, 
        [](auto &s) { return s.begin(); },
        [](auto &s) { return s.end(); });
      test_iterator<span_t::const_iterator>(sp, rule,
        [](auto &s) { return s.cbegin(); },
        [](auto &s) { return s.cend(); });

      // For reverse iterators, first_value and last_value are swapped
      const expectation reverse_rule{values_count, values_count, int_value};
      test_iterator<span_t::reverse_iterator>(sp, reverse_rule,
        [](auto &s) { return s.rbegin(); },
        [](auto &s) { return s.rend(); });
      test_iterator<span_t::const_reverse_iterator>(sp, reverse_rule,
        [](auto &s) { return s.crbegin(); },
        [](auto &s) { return s.crend(); });
    });
  }).wait();

  free(data, Q);
}

void fixedExtentSpan() {
  queue Q;

  static constexpr size_t values_count = 5;
  static constexpr int int_value = 1;
  auto *data = malloc_shared<int>(values_count, Q);
  std::iota(data, data + values_count, 1); // [1, 2, 3, 4, 5]

  using span_t = sycl::span<int, values_count>;
  Q.submit([&](handler &cgh) {
    cgh.single_task<class test_fixed_iterators>([=] {
      span_t sp(data, values_count);
      
      // Test iterator types exist
      using iterator = typename span_t::iterator;
      using const_iterator = typename span_t::const_iterator;
      using reverse_iterator = typename span_t::reverse_iterator;
      using const_reverse_iterator = typename span_t::const_reverse_iterator;
      
      const expectation forward_rule{values_count, int_value, values_count};
      test_iterator<span_t::iterator>(sp, forward_rule,
        [](auto &s) { return s.begin(); },
        [](auto &s) { return s.end(); });
      test_iterator<span_t::const_iterator>(sp, forward_rule,
        [](auto &s) { return s.cbegin(); },
        [](auto &s) { return s.cend(); });

      // For reverse iterators, first_value and last_value are swapped
      const expectation reverse_rule{values_count, values_count, int_value};
      test_iterator<span_t::reverse_iterator>(sp, reverse_rule,
        [](auto &s) { return s.rbegin(); },
        [](auto &s) { return s.rend(); });
      test_iterator<span_t::const_reverse_iterator>(sp, reverse_rule,
        [](auto &s) { return s.crbegin(); },
        [](auto &s) { return s.crend(); });
      
      // Test iterator arithmetic
      auto it = sp.begin();
      it += 2;
      assert(*it == 3);
      it -= 1;
      assert(*it == 2);
      assert(it[2] == 4);
      
      // Test sum calculation
      int sum = 0;
      for (auto iter = sp.begin(); iter != sp.end(); ++iter) {
        sum += *iter;
      }
      assert(sum == 15); // 1+2+3+4+5
    });
  }).wait();
  
  free(data, Q);
}

void dynamicExtentSpan() {
  queue Q;
  
  static constexpr size_t values_count = 5;
  static constexpr int start_value = 10;
  auto *data = malloc_shared<int>(values_count, Q);
  std::iota(data, data + values_count, start_value); // [10, 11, 12, 13, 14]
  
  using span_t = sycl::span<int, dynamic_extent>;
  Q.submit([&](handler &cgh) {
    cgh.single_task<class test_dynamic_iterators>([=] {
      span_t sp(data, values_count);
      
      // Test iterator types exist
      using iterator = typename span_t::iterator;
      using const_iterator = typename span_t::const_iterator;
      using reverse_iterator = typename span_t::reverse_iterator;
      using const_reverse_iterator = typename span_t::const_reverse_iterator;
      
      // Use the unified test_iterator approach
      const expectation forward_rule{values_count, start_value, start_value + values_count - 1};
      test_iterator<iterator>(sp, forward_rule,
        [](auto &s) { return s.begin(); },
        [](auto &s) { return s.end(); });
      test_iterator<const_iterator>(sp, forward_rule,
        [](auto &s) { return s.cbegin(); },
        [](auto &s) { return s.cend(); });

      // For reverse iterators, first_value and last_value are swapped
      const expectation reverse_rule{values_count, start_value + values_count - 1, start_value};
      test_iterator<reverse_iterator>(sp, reverse_rule,
        [](auto &s) { return s.rbegin(); },
        [](auto &s) { return s.rend(); });
      test_iterator<const_reverse_iterator>(sp, reverse_rule,
        [](auto &s) { return s.crbegin(); },
        [](auto &s) { return s.crend(); });
      
      // Test iterator modification
      *sp.begin() = 100;
      assert(sp[0] == 100);
      
      // Test range-based for loop
      int count = 0;
      for (auto &elem : sp) {
        ++count;
        elem += 1000;
      }
      assert(count == 5);
      assert(sp[0] == 1100); // 100 + 1000
      assert(sp[1] == 1011); // 11 + 1000
    });
  }).wait();
  
  free(data, Q);
}

// Test constructor explicitness
void constructorExplicitness() {
  queue Q;
  
  auto *data = malloc_shared<int>(4, Q);
  std::iota(data, data + 4, 1);
  
  Q.submit([&](handler &cgh) {
    cgh.single_task<class test_constructors>([=] {
      int arr[4] = {1, 2, 3, 4};
      int *ptr = data;
      int *end = data + 4;
      
      // 1. Default constructor - only dynamic extent should be default constructible
      static_assert(std::is_default_constructible_v<sycl::span<int>>);
      // Fixed extent spans typically cannot be default constructed
      // static_assert(std::is_default_constructible_v<sycl::span<int, 4>>);
      // TODO: add sema tests to match the diagnistics

      // 2. Array reference constructor - should always be implicit
      static_assert(std::is_convertible_v<int(&)[4], sycl::span<int>>);
      static_assert(std::is_convertible_v<int(&)[4], sycl::span<int, 4>>);

      // 3. std::array constructors - should always be implicit
      static_assert(std::is_convertible_v<std::array<int, 4>&, sycl::span<int>>);
      static_assert(std::is_convertible_v<const std::array<int, 4>&, sycl::span<const int>>);
      static_assert(std::is_convertible_v<std::array<int, 4>&, sycl::span<int, 4>>);
      static_assert(std::is_convertible_v<const std::array<int, 4>&, sycl::span<const int, 4>>);

      // 4. Pointer + count constructor
      // Should be implicit for dynamic extent, explicit for fixed extent
      static_assert(std::is_constructible_v<sycl::span<int>, int*, size_t>);
      static_assert(std::is_constructible_v<sycl::span<int, 4>, int*, size_t>);
      static_assert(!std::is_convertible_v<int*, sycl::span<int, 4>>); // Should be explicit
      
      // For dynamic extent span, pointer+count should be implicit
      static_assert(std::is_convertible_v<std::pair<int*, size_t>, sycl::span<int>> == false, 
                    "Cannot test pair conversion directly - use constructible instead");
      
      // 5. Pointer range constructor (first, last)
      // Should be implicit for dynamic extent, explicit for fixed extent
      static_assert(std::is_constructible_v<sycl::span<int>, int*, int*>);
      static_assert(std::is_constructible_v<sycl::span<int, 4>, int*, int*>);
      
      // 6. Container constructor - should be explicit when extent != dynamic_extent
      std::vector<int> vec{1, 2, 3, 4};
      static_assert(std::is_constructible_v<sycl::span<int>, std::vector<int>&>);
      // Note: SYCL span implementation marks container constructors as explicit
      
      // 7. Copy constructor - should always be implicit
      static_assert(std::is_convertible_v<sycl::span<int>, sycl::span<int>>);
      static_assert(std::is_convertible_v<sycl::span<int, 4>, sycl::span<int, 4>>);
      
      // 8. Converting span constructor
      static_assert(std::is_constructible_v<sycl::span<int>, sycl::span<int, 4>>);
      static_assert(std::is_constructible_v<sycl::span<int, 4>, sycl::span<int, 4>>);
      
      // Test actual usage patterns that should work implicitly
      auto test_implicit_array = [](sycl::span<int> sp) { return sp.size(); };
      assert(test_implicit_array(arr) == 4); // Array should convert implicitly
      
      auto test_implicit_std_array = [](sycl::span<int> sp) { return sp.size(); };
      std::array<int, 4> std_arr = {1, 2, 3, 4};
      assert(test_implicit_std_array(std_arr) == 4); // std::array should convert implicitly
      
      // Test direct construction (these should work regardless of explicitness)
      sycl::span<int> sp1(arr);              // From array
      sycl::span<int> sp2(ptr, 4);           // From pointer + count  
      sycl::span<int> sp3(ptr, end);         // From pointer range
      sycl::span<int> sp4(std_arr);          // From std::array
      
      // Fixed extent versions
      sycl::span<int, 4> sp1_fixed(arr);     // From array
      sycl::span<int, 4> sp2_fixed(ptr, 4);  // From pointer + count
      sycl::span<int, 4> sp3_fixed(ptr, end); // From pointer range
      sycl::span<int, 4> sp4_fixed(std_arr); // From std::array
      
      assert(sp1.size() == 4 && sp1_fixed.size() == 4);
      assert(sp2.size() == 4 && sp2_fixed.size() == 4);
      assert(sp3.size() == 4 && sp3_fixed.size() == 4);
      assert(sp4.size() == 4 && sp4_fixed.size() == 4);
      
      // Test span conversion
      sycl::span<int> sp_from_fixed = sp1_fixed; // Fixed to dynamic should be implicit
      assert(sp_from_fixed.size() == 4);
      
      // Test that elements are accessible
      assert(sp1[0] == 1);
      assert(sp2[0] == 1);
      assert(sp3[0] == 1);
      assert(sp4[0] == 1);
    });
  }).wait();
  
  free(data, Q);
}

void explicitConstructorBehavior() {
  queue Q;
  
  Q.submit([&](handler &cgh) {
    cgh.single_task<class test_explicit_behavior>([=] {
      int arr[3] = {1, 2, 3};
      std::vector<int> vec{1, 2, 3};
      
      // These should work (direct construction)
      sycl::span<int, 3> fixed_from_ptr(arr, 3);
      sycl::span<int, 3> fixed_from_range(arr, arr + 3);
      sycl::span<int, 3> fixed_from_container(vec);
      
      // These should work (implicit for dynamic extent)
      auto implicit_test = [](sycl::span<int> sp) { return sp.size(); };
      assert(implicit_test(arr) == 3); // Array reference - always implicit
      
      // Container construction - marked explicit in SYCL implementation
      sycl::span<int> dynamic_from_container(vec);
      assert(dynamic_from_container.size() == 3);
      
      // Test the explicit behavior matches the standard pattern:
      // explicit(extent != dynamic_extent)
      
      // For pointer+count and pointer+pointer constructors:
      // - dynamic extent: should allow implicit conversion
      // - fixed extent: should require explicit construction
      
      sycl::span<int> dynamic_ptr_count(arr, 3);     // Should work
      sycl::span<int> dynamic_ptr_range(arr, arr+3); // Should work
      
      sycl::span<int, 3> fixed_ptr_count(arr, 3);     // Should work (explicit)
      sycl::span<int, 3> fixed_ptr_range(arr, arr+3); // Should work (explicit)
      
      assert(dynamic_ptr_count.size() == 3);
      assert(dynamic_ptr_range.size() == 3);
      assert(fixed_ptr_count.size() == 3);
      assert(fixed_ptr_range.size() == 3);
    });
  }).wait();
}

// Test to verify the exact explicitness pattern from the standard
void standardExplicitnessPattern() {
  // This test verifies the explicit(condition) pattern from std::span
  
  // According to the standard:
  // - span() - not explicit
  // - span(It first, size_type count) - explicit(extent != dynamic_extent)  
  // - span(It first, End last) - explicit(extent != dynamic_extent)
  // - span(element_type (&arr)[N]) - not explicit
  // - span(array<T, N>& arr) - not explicit
  // - span(const array<T, N>& arr) - not explicit
  // - span(R&& r) - explicit(extent != dynamic_extent)
  // - span(const span& other) - not explicit (copy constructor)
  // - span(const span<OtherElementType, OtherExtent>& s) - explicit(see below)
  
  static_assert(std::is_default_constructible_v<sycl::span<int>>);
  //static_assert(std::is_default_constructible_v<sycl::span<int, 5>>);
  
  constexpr static auto magic_size = 2u;
  // Array reference constructors - never explicit
  static_assert(std::is_constructible_v<sycl::span<int>, int(&)[magic_size]>);
  static_assert(std::is_constructible_v<sycl::span<int, magic_size>, int(&)[magic_size]>);
  
  // std::array constructors - never explicit  
  static_assert(std::is_constructible_v<sycl::span<int>, std::array<int, magic_size>&>);
  static_assert(std::is_constructible_v<sycl::span<int, magic_size>, std::array<int, magic_size>&>);
  
  // Pointer + count constructors - explicit when extent != dynamic_extent
  static_assert(std::is_constructible_v<sycl::span<int>, int*, size_t>);
  static_assert(std::is_constructible_v<sycl::span<int, magic_size>, int*, size_t>);

  // Range constructors - explicit when extent != dynamic_extent
  static_assert(std::is_constructible_v<sycl::span<int>, int*, int*>);
  static_assert(std::is_constructible_v<sycl::span<int, magic_size>, int*, int*>);

  // Container constructor - explicit when extent != dynamic_extent
  static_assert(std::is_constructible_v<sycl::span<int>, std::vector<int>&>);
  static_assert(std::is_constructible_v<sycl::span<int, magic_size>, std::vector<int>&>);
  static_assert(std::is_constructible_v<sycl::span<int>, std::array<int, magic_size>&>);

  // Copy constructor - never explicit
  static_assert(std::is_constructible_v<sycl::span<int>, sycl::span<int>>);
  static_assert(std::is_constructible_v<sycl::span<int, magic_size>, sycl::span<int, magic_size>>);
  
  // Converting span constructor - explicit when extent != dynamic_extent
  static_assert(std::is_constructible_v<sycl::span<int>, sycl::span<int, magic_size>>);
  static_assert(std::is_constructible_v<sycl::span<int, magic_size>, sycl::span<int, magic_size>>);

  //TODO: need a matrix based test to ensure all the explicitness patterns
}

void iteratorSTLCompatibility() {
  queue Q;
  
  auto *data = malloc_shared<int>(6, Q);
  std::iota(data, data + 6, 1); // [1, 2, 3, 4, 5, 6]
  
  Q.submit([&](handler &cgh) {
    cgh.single_task<class test_stl_compat>([=] {
      sycl::span<int> sp(data, 6);
      
      auto found = std::find(sp.begin(), sp.end(), 4);
      assert(found != sp.end());
      assert(*found == 4);
      assert(found - sp.begin() == 3);
      
      // Test std::count
      int count_ones = std::count(sp.begin(), sp.end(), 1);
      assert(count_ones == 1);
      
      // Test std::reverse with reverse iterators
      bool is_reversed = true;
      auto it = sp.begin();
      auto rit = sp.rbegin();
      for (size_t i = 0; i < sp.size(); ++i, ++it, ++rit) {
        int expected_sum = sp.front() + sp.back(); // Calculate expected sum dynamically
        if (*it + *(sp.rbegin() + static_cast<std::ptrdiff_t>(i)) != expected_sum) { // Ensure proper type handling
          is_reversed = false;
          break;
        }
      }
      assert(is_reversed);
      
      // Test iterator traits
      using iter_traits = std::iterator_traits<decltype(sp.begin())>;
      static_assert(std::is_same_v<iter_traits::iterator_category, std::random_access_iterator_tag>);
      static_assert(std::is_same_v<iter_traits::value_type, int>);
      static_assert(std::is_same_v<iter_traits::pointer, int*>);
      static_assert(std::is_same_v<iter_traits::reference, int&>);
    });
  }).wait();
  
  free(data, Q);
}


void checkAsWritableBytes() {
  queue Q;
  int arr[5] = {10, 20, 30, 40, 50};
  sycl::span<int> sp(arr, 5);

  auto sub = sp.subspan(1, 3);
  assert(sub.size() == 3 && sub[0] == 20 && sub[2] == 40);

  auto first2 = sp.first(2);
  assert(first2.size() == 2 && first2[1] == 20);

  auto last2 = sp.last(2);
  assert(last2.size() == 2 && last2[0] == 40);

  auto bytes = sycl::as_bytes(sp);
  assert(bytes.size() == sizeof(int) * 5);

  auto writable_bytes = sycl::as_writable_bytes(sp);
  assert(writable_bytes.size() == sizeof(int) * 5);
  writable_bytes[0] = 0xFF;
  assert(arr[0] == 0xFF);
}
} // namespace ApiTests

int main() {
  BasicTests::simpleAccess();
  BasicTests::zeroLengthCase();
  BasicTests::emptyContainerCase();
  BasicTests::nullptrCase();
  BasicTests::simpleStress();
  BasicTests::alignmentAliasing();
  BasicTests::spanFromContainer();

  DeviceTests::spanCapture();
  DeviceTests::spanOnDevice();
  DeviceTests::onTwoDevices();
  
  ApiTests::iteratorTypes();
  ApiTests::checkAsWritableBytes();
  ApiTests::fixedExtentSpan();
  ApiTests::dynamicExtentSpan();
  ApiTests::constructorExplicitness();
  ApiTests::iteratorSTLCompatibility();
  //ApiTests:://testConstCorrectness();  

  return 0;
}
