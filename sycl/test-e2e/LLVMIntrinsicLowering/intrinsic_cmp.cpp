// RUN: %{build} -Wno-error=psabi -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

// Define vector types for different integer bit widths. We need these to
// trigger llvm.scmp/ucmp for vector types. std::array or sycl::vec don't
// trigger these, as they are not lowered to vector types.
typedef int8_t v4i8_t __attribute__((ext_vector_type(4)));
typedef int16_t v4i16_t __attribute__((ext_vector_type(4)));
typedef int32_t v4i32_t __attribute__((ext_vector_type(4)));
typedef int64_t v4i64_t __attribute__((ext_vector_type(4)));
typedef uint8_t v4u8_t __attribute__((ext_vector_type(4)));
typedef uint16_t v4u16_t __attribute__((ext_vector_type(4)));
typedef uint32_t v4u32_t __attribute__((ext_vector_type(4)));
typedef uint64_t v4u64_t __attribute__((ext_vector_type(4)));

// Check if a given type is a vector type or not. Used in submitAndCheck to
// branch the check: we need element-wise comparison for vector types. Default
// case: T is not a vector type.
template <typename T> struct is_vector : std::false_type {};
// Specialization for vector types. If T has
// __attribute__((ext_vector_type(N))), then it's a vector type.
template <typename T, std::size_t N>
struct is_vector<T __attribute__((ext_vector_type(N)))> : std::true_type {};
template <typename T> inline constexpr bool is_vector_v = is_vector<T>::value;

// Get the length of a vector type. Used in submitAndCheck to iterate over the
// elements of the vector type. Default case: length is 1.
template <typename T> struct vector_length {
  static constexpr std::size_t value = 1;
};
// Specialization for vector types. If T has
// __attribute__((ext_vector_type(N))), then the length is N.
template <typename T, std::size_t N>
struct vector_length<T __attribute__((ext_vector_type(N)))> {
  static constexpr std::size_t value = N;
};
template <typename T>
inline constexpr std::size_t vector_length_v = vector_length<T>::value;

// Get the element type of a vector type. Used in submitVecCombinations to
// convert unsigned vector types to signed vector types for return type. Primary
// template for element_type.
template <typename T> struct element_type;
// Specialization for vector types. If T has
// __attribute__((ext_vector_type(N))), return T.
template <typename T, int N>
struct element_type<T __attribute__((ext_vector_type(N)))> {
  using type = T;
};
// Helper alias template.
template <typename T> using element_type_t = typename element_type<T>::type;

// TypeList for packing the types that we want to test.
// Base case for variadic template recursion.
template <typename...> struct TypeList {};

// Function to trigger llvm.scmp/ucmp.
template <typename RetTy, typename ArgTy>
void compare(RetTy &res, ArgTy x, ArgTy y) {
  auto lessOrEq = (x <= y);
  auto lessThan = (x < y);
  res = lessOrEq ? (lessThan ? RetTy(-1) : RetTy(0)) : RetTy(1);
}

// Function to submit kernel and check device result with host result.
template <typename RetTy, typename ArgTy>
void submitAndCheck(sycl::queue &q, ArgTy x, ArgTy y) {
  RetTy res;
  {
    sycl::buffer<RetTy, 1> res_b{&res, 1};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc{res_b, cgh, sycl::write_only};
      cgh.single_task<>([=] {
        RetTy tmp;
        compare<RetTy, ArgTy>(tmp, x, y);
        acc[0] = tmp;
      });
    });
  }
  RetTy expectedRes;
  compare<RetTy, ArgTy>(expectedRes, x, y);
  if constexpr (is_vector_v<RetTy>) {
    for (int i = 0; i < vector_length_v<RetTy>; ++i) {
      assert(res[i] == expectedRes[i]);
    }
  } else {
    assert(res == expectedRes);
  }
}

// Helper to call submitAndCheck for each combination.
template <typename RetTypes, typename ArgTypes>
void submitAndCheckCombination(sycl::queue &q, int x, int y) {
  submitAndCheck<RetTypes, ArgTypes>(q, x, y);
}

// Function to generate all the combinations possible with the two type lists.
// It implements the following pseudocode :
// foreach RetTy : RetTypes
//   foreach ArgTy : ArgTypes
//     submitAndCheck<RetTy, ArgTy>(q, x, y);

// Recursive case to generate combinations.
template <typename RetType, typename... RetTypes, typename... ArgTypes>
void submitCombinations(sycl::queue &q, int x, int y,
                        TypeList<RetType, RetTypes...>, TypeList<ArgTypes...>) {
  (submitAndCheckCombination<RetType, ArgTypes>(q, x, y), ...);
  submitCombinations(q, x, y, TypeList<RetTypes...>{}, TypeList<ArgTypes...>{});
}
// Base case to stop recursion.
template <typename... ArgTypes>
void submitCombinations(sycl::queue &, int, int, TypeList<>,
                        TypeList<ArgTypes...>) {}

// Function to generate all the combinations out of the given list.
// It implements the following pseudocode :
// foreach ArgTy : ArgTypes
//   submitAndCheck<ArgTy, ArgTy>(q, x, y);

// Recursive case to generate combinations.
template <typename ArgType, typename... ArgTypes>
void submitVecCombinations(sycl::queue &q, int x, int y,
                           TypeList<ArgType, ArgTypes...>) {
  // Use signed types for return type, as it may return -1.
  using ElemType = std::make_signed_t<element_type_t<ArgType>>;
  using RetType =
      ElemType __attribute__((ext_vector_type(vector_length_v<ArgType>)));
  submitAndCheckCombination<RetType, ArgType>(q, x, y);
  submitVecCombinations(q, x, y, TypeList<ArgTypes...>{});
}
// Base case to stop recursion.
void submitVecCombinations(sycl::queue &, int, int, TypeList<>) {}

int main(int argc, char **argv) {
  sycl::queue q;
  // RetTypes includes only signed types because it may return -1.
  using RetTypes = TypeList<int8_t, int16_t, int32_t, int64_t>;
  using ArgTypes = TypeList<int8_t, int16_t, int32_t, int64_t, uint8_t,
                            uint16_t, uint32_t, uint64_t>;
  submitCombinations(q, 50, 49, RetTypes{}, ArgTypes{});
  submitCombinations(q, 50, 50, RetTypes{}, ArgTypes{});
  submitCombinations(q, 50, 51, RetTypes{}, ArgTypes{});
  using VecTypes = TypeList<v4i8_t, v4i16_t, v4i32_t, v4i64_t, v4u8_t, v4u16_t,
                            v4u32_t, v4u64_t>;
  submitVecCombinations(q, 50, 49, VecTypes{});
  submitVecCombinations(q, 50, 50, VecTypes{});
  submitVecCombinations(q, 50, 51, VecTypes{});
  return 0;
}
