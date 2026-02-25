//==--------------- poc-properties.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

// This is a POC implementation that I am using to help flesh out issues with
// the proposed sycl_khr__properties extension.  It is not part of the DPC++
// implementation for properties currently, though we might consider using
// ideas from this POC in the future.

// Compile and run tests with:
//
// $ clang++ -std=c++20 -o poc-properties poc-properties.cpp
// $ ./poc-properties

#include <cassert>
#include <tuple>
#include <type_traits>


namespace sycl {

namespace khr {
namespace detail {

// Every property and property key must inherit from one of these base classes.
// These base classes are used to define the traits for each property and key.
//
//  runtime_property_base
//    Use for a property where all values are defined at runtime.  There is no
//    separate property key in this case.
//
//  constant_property_base
//  constant_property_key_base
//    Use these for a property where all values are defined at compile-time.
//
//  hybrid_property_base
//  hybrid_property_key_base
//    Use these for a property where some values are defined at runtime and some
//    are defined at compile-time.
struct runtime_property_base{};
struct constant_property_base{};
struct constant_property_key_base{};
struct hybrid_property_base{};
struct hybrid_property_key_base{};

// Convenience base class for a property where all values are defined at runtime.
template<typename Property>
struct runtime_property : runtime_property_base {
  using __detail_key_t = Property;
};

// Convenience base class for a property / key with a single non-type value that
// is defined at compile-time.
struct constant_value_property_key : constant_property_key_base {
  template<typename PropertyKey, typename Type, Type Value>
  struct __detail_property_t : constant_property_base {
    using __detail_key_t = PropertyKey;
    static constexpr Type value = Value;
  };
};

// Convenience base class for a property / key with a single type value (that is
// defined at compile-time).
struct constant_type_property_key : constant_property_key_base {
  template<typename PropertyKey, typename Type>
  struct __detail_property_t : constant_property_base {
    using __detail_key_t = PropertyKey;
    using value_t = Type;
  };
};

// We could add more convenience base classes for other patterns that become
// common.


// Metaprogramming utility that given a property key, finds the property in a
// pack which matches that key.
template<typename PropertyKey, typename... Properties>
struct find_property {
  using type = void;
};

template<typename PropertyKey, typename Property, typename... Rest>
struct find_property<PropertyKey, Property, Rest...> {
  using type = std::conditional_t<
    std::is_same_v<PropertyKey, typename Property::__detail_key_t>,
    Property,
    typename find_property<PropertyKey, Rest...>::type>;
};


// Trait which is true if property Property has values that are all known at
// compile-time.
template<typename Property>
struct is_property_compile_time: std::conditional_t<
  std::is_base_of_v<detail::constant_property_base, Property>,
  std::true_type,
  std::false_type> {};

template<typename Property>
inline constexpr bool is_property_compile_time_v = is_property_compile_time<Property>::value;


// Metaprogramming utility that takes a type T and a tuple, and constructs a new
// tuple where T is the first element and the remaining elements are those from
// the input tuple.
template <typename, typename> struct tuple_concat;
template <typename T, typename... Args> using tuple_concat_t = tuple_concat<T, Args...>::type;

template <typename T, typename... Args>
struct tuple_concat<T, std::tuple<Args...>> {
  using type = std::tuple<T, Args...>;
};


// Metaprogramming utility that takes a pack of properties and constructs a
// tuple whose elements contain only those properties that have runtime values.
template<typename... Properties> struct filter_runtime_properties {
  using type = std::tuple<>;
};

template<typename... Properties> using filter_runtime_properties_t =
  typename filter_runtime_properties<Properties...>::type;

template<typename Property, typename... Rest>
struct filter_runtime_properties<Property, Rest...> {
  using type = std::conditional_t<
    !is_property_compile_time_v<Property>,
    tuple_concat_t<Property, filter_runtime_properties_t<Rest...>>,
    filter_runtime_properties_t<Rest...>
  >;
};

} // namespace detail


// Trait which is true if T is a property.
template<typename T>
struct is_property : std::conditional_t<
  (std::is_base_of_v<detail::runtime_property_base, T> ||
   std::is_base_of_v<detail::constant_property_base, T> ||
   std::is_base_of_v<detail::hybrid_property_base, T>),
  std::true_type,
  std::false_type> {};

template<typename T>
inline constexpr bool is_property_v = is_property<T>::value;


// Trait which is true if T is a property key.
template<typename T>
struct is_property_key : std::conditional_t<
  (std::is_base_of_v<detail::runtime_property_base, T> ||
   std::is_base_of_v<detail::constant_property_key_base, T> ||
   std::is_base_of_v<detail::hybrid_property_key_base, T>),
  std::true_type,
  std::false_type> {};

template<typename T>
inline constexpr bool is_property_key_v = is_property_key<T>::value;


// Trait which is true if T is a property key for a property whose
// values are all known at compile-time.
template<typename T>
struct is_property_key_compile_time : std::conditional_t<
  std::is_base_of_v<detail::constant_property_key_base, T>,
  std::true_type,
  std::false_type> {};

template<typename T>
inline constexpr bool is_property_key_compile_time_v =
  is_property_key_compile_time<T>::value;


// Trait which is true if T is a property list that contains only properties
// that are for class Class.
template<typename T, typename Class>
struct is_property_list_for : std::false_type {};

template<typename T, typename Class>
inline constexpr bool is_property_list_for_v = is_property_list_for<T, Class>::value;


// Trait which is true if T is a property key that is for class Class.
template<typename T, typename Class>
struct is_property_key_for : std::false_type {};

template<typename T, typename Class>
inline constexpr bool is_property_key_for_v = is_property_key_for<T, Class>::value;


// Trait which is true if T is a property that is for class Class.
template<typename T, typename Class>
struct is_property_for : std::false_type {};

template<typename T, typename Class>
requires(is_property_v<T>)
struct is_property_for<T, Class> : is_property_key_for<typename T::__detail_key_t, Class> {};

template<typename T, typename Class>
inline constexpr bool is_property_for_v = is_property_for<T, Class>::value;


// A list of properties
//
// EncodedProperties are the types of all the properties in the list.  This
// includes both runtime and compile-time properties.
//
// In this implementation, EncodedProperties is the same as the Properties pack
// that is passed to the constructor.  In the future, though, EncodedProperties
// could be a sorted list of the Properties, which would allow us to compare two
// properties lists for equality.
template<typename... EncodedProperties>
class properties {
 private:
  // A tuple which contains only those properties that have at least one value
  // defined at runtime.
  using stored_runtime_properties_t =
    detail::filter_runtime_properties_t<EncodedProperties...>;
  stored_runtime_properties_t stored_properties;

  // TODO: There should be a static_assert here that diagnoses an error if the
  // EncodedProperties pack contains properties with duplicate keys.

 public:
  template<typename... Properties>
  requires(is_property_v<Properties> && ...)
  properties(Properties... props)
  :
  stored_properties{
    // Find the subset of properties in `props` that have runtime values and
    // construct `stored_properties` from this subset.
    std::tuple_cat(
      std::get<detail::is_property_compile_time_v<Properties> ? 0 : 1>(
        std::make_tuple(
          [](Properties&&) { return std::tuple<>{}; },
          [](Properties&& prop) { return std::tuple<Properties&&>{std::forward<Properties>(prop)}; }
        )
      )(std::forward<Properties>(props))
      ...
    )
  } {}

  template<typename PropertyKey>
  requires(is_property_key_v<PropertyKey>)
  static constexpr bool has_property() {
    // Each property in EncodedProperties has a __detail_key_t type alias.  To see if
    // PropertyKey is in the list, we check if it is the same as any __detail_key_t.
    return (std::is_same_v<PropertyKey, typename EncodedProperties::__detail_key_t> || ...);
  }

  template<typename PropertyKey>
  requires(is_property_key_compile_time_v<PropertyKey> && has_property<PropertyKey>())
  static constexpr auto get_property() {
    // Search EncodedProperties to find the property that corresponds to
    // PropertyKey.  Since this is a compile-time property, we can default
    // construct it at compile-time.
    using Property = detail::find_property<PropertyKey, EncodedProperties...>::type;
    return Property{};
  }

  template<typename PropertyKey>
  requires(!is_property_key_compile_time_v<PropertyKey> && has_property<PropertyKey>())
  constexpr auto get_property() {
    // Search EncodedProperties to find the property that corresponds to
    // PropertyKey.  Return the runtime value stored in `stored_properties`.
    using Property = detail::find_property<PropertyKey, EncodedProperties...>::type;
    return get<Property>(stored_properties);
  }
};

template<typename... Properties>
properties(Properties...) -> properties<Properties...>;

using empty_properties_t = decltype(properties{});

template<typename... Properties, typename Class>
requires(is_property_for_v<Properties, Class> && ...)
struct is_property_list_for<properties<Properties...>, Class> : std::true_type {};

} // namespace khr


// Example usage in queue

class queue {
 public:
  template<typename PropertyOrList = khr::empty_properties_t>
  requires(khr::is_property_for_v<PropertyOrList, queue> ||
           khr::is_property_list_for_v<PropertyOrList, queue>)
  queue(PropertyOrList props = {}) {}

#if 0
  // It makes little sense to provide these because applications don't want to
  // know if the queue was constructed with a certain property.  For example,
  // it's senseless to query whether the queue was constructed with
  // `enable_profiling` because the property's value might be either true or
  // false.  Instead, applications want to know whether the queue has profiling
  // enabled.
  template<typename Property>
  requires(khr::is_property_for_v<Property, queue>)
  bool has_property() const;

  template<typename Property>
  requires(khr::is_property_for_v<Property, queue>)
  Prop get_property() const;
#endif
};

namespace khr {

// Example runtime property with a single value.
struct enable_profiling : detail::runtime_property<enable_profiling> {
  enable_profiling(bool v) : value{v} {}
  bool value;
};
template<> struct is_property_for<enable_profiling, queue> : std::true_type {};

// Example runtime property with two values.
struct twoarg : detail::runtime_property<twoarg> {
  twoarg(int x, int y) : one{x}, two{y} {}
  int one;
  int two;
};
template<> struct is_property_for<twoarg, queue> : std::true_type {};

}


// Example usage in annotated_ptr

namespace khr {

struct annotated_ptr_properties;

// The annotated_ptr class only allows compile-time properties.
// Properties is the property list with all the properties attached to this
// annotated_ptr.
template <typename T, typename Properties = empty_properties_t>
class annotated_ptr {
 public:
  annotated_ptr(T* p, const Properties& props = {})
  requires(is_property_for_v<Properties, annotated_ptr_properties> ||
           is_property_list_for_v<Properties, annotated_ptr_properties>)
  {}

  template<typename PropertyKey>
  requires(is_property_key_for_v<PropertyKey, annotated_ptr_properties>)
  static constexpr bool has_property() {
    return Properties::template has_property<PropertyKey>();
  }

  template<typename PropertyKey>
  requires(is_property_key_for_v<PropertyKey, annotated_ptr_properties>)
  static constexpr auto get_property() {
    return Properties::template get_property<PropertyKey>();
  }
};

// Deduction guides that allow annotated_ptr to be constructed from either a
// single property or a property list.
template<typename T, typename Properties>
requires(is_property_list_for_v<Properties, annotated_ptr_properties>)
annotated_ptr(T*, Properties) -> annotated_ptr<T, Properties>;

template<typename T, typename Property>
requires(is_property_for_v<Property, annotated_ptr_properties>)
annotated_ptr(T*, Property) -> annotated_ptr<T, decltype(properties{Property{}})>;


// Example compile-time property with one value.
struct alignment_key : detail::constant_value_property_key {};
template<> struct is_property_key_for<alignment_key, annotated_ptr_properties> : std::true_type {};

template<std::size_t Alignment>
inline constexpr alignment_key::__detail_property_t<alignment_key, std::size_t, Alignment> alignment;

// Example compile-time property with one type.
struct alignment_type_key : detail::constant_type_property_key {};
template<> struct is_property_key_for<alignment_type_key, annotated_ptr_properties> : std::true_type {};

template<typename T>
inline constexpr alignment_type_key::__detail_property_t<alignment_type_key, T> alignment_type;

// Example compile-time property with three values.
// This is more verbose because it doesn't use a convenience base class.
struct threearg_key : detail::constant_property_key_base {
  template<int X, bool Y, int Z>
  struct __detail_property_t : detail::constant_property_base {
    using __detail_key_t = threearg_key;
    static constexpr int x = X;
    static constexpr bool y = Y;
    static constexpr int z = Z;
  };
};
template<> struct is_property_key_for<threearg_key, annotated_ptr_properties> : std::true_type {};

template<int X, bool Y, int Z>
inline constexpr threearg_key::__detail_property_t<X, Y, Z> threearg;

} // namespace khr


namespace khr {

// Example hybrid property that contains both compile-time and runtime values.
struct hybrid_key : detail::hybrid_property_key_base {};

template<int X>
struct hybrid : detail::hybrid_property_base {
  hybrid(int y) : y{y} {}
  using __detail_key_t = hybrid_key;
  static constexpr int x = X;
  int y;
};

} // namespace khr
} // namespace sycl


int main() {
  // Check sizes of properties.  The spec does not mandate this, but we
  // expect that compile-time properties have the minimum possible size
  // (1 byte), and we expect runtime properties to have size equal to their
  // runtime values.
  static_assert(sizeof(sycl::khr::enable_profiling{true}) == sizeof(bool));
  static_assert(sizeof(sycl::khr::twoarg{1, 2}) == 2*sizeof(int));
  static_assert(sizeof(sycl::khr::alignment<16>) == 1);
  static_assert(sizeof(sycl::khr::alignment_type<int>) == 1);
  static_assert(sizeof(sycl::khr::threearg<1, false, 2>) == 1);
  static_assert(sizeof(sycl::khr::hybrid<1>{2}) == sizeof(int));

  // Check sizes of property lists.  The spec does not mandate any specific
  // size, but we expect that a property list with all compile-time properties
  // should have the minimum possible size (1 byte), and we expect that a
  // property list with a mixture of compile-time and runtime properties will
  // have a size equal to the sum of the sizes of the runtime properties.
  {
    sycl::khr::properties p;
    static_assert(sizeof(p) == 1);
  }
  {
    sycl::khr::properties p{
      sycl::khr::alignment<16>,
      sycl::khr::alignment_type<int>,
      sycl::khr::threearg<1, false, 2>
    };
    static_assert(sizeof(p) == 1);
  }
  {
    sycl::khr::properties p{
      sycl::khr::enable_profiling{true},
      sycl::khr::twoarg{1, 2}
    };
    static_assert(sizeof(p) == sizeof(int) + 2*sizeof(int));
  }
  {
    sycl::khr::properties p{
      sycl::khr::enable_profiling{true},
      sycl::khr::twoarg{1, 2}
    };
    static_assert(sizeof(p) == sizeof(int) + 2*sizeof(int));
  }
  {
    sycl::khr::properties p{
      sycl::khr::alignment_type<float>,
      sycl::khr::enable_profiling{true},
      sycl::khr::threearg<1, false, 2>,
    };
    static_assert(sizeof(p) == 1);
  }
  {
    sycl::khr::properties p{
      sycl::khr::hybrid<1>{2},
      sycl::khr::alignment<16>,
      sycl::khr::threearg<1, false, 2>,
    };
    static_assert(sizeof(p) == sizeof(int));
  }

  // Check that properties and properties lists are trivially copyable.
  {
    static_assert(std::is_trivially_copyable_v<decltype(sycl::khr::enable_profiling{true})>);
    static_assert(std::is_trivially_copyable_v<decltype(sycl::khr::twoarg{1, 2})>);
    static_assert(std::is_trivially_copyable_v<decltype(sycl::khr::alignment<16>)>);
    static_assert(std::is_trivially_copyable_v<decltype(sycl::khr::alignment_type<int>)>);
    static_assert(std::is_trivially_copyable_v<decltype(sycl::khr::threearg<1, false, 2>)>);
    static_assert(std::is_trivially_copyable_v<decltype(sycl::khr::hybrid<1>{2})>);
  }
#if 0
  // Disabled because the "properties" type is not trivially copyable in this
  // POC due to the use of "std::tuple".
  //
  // TODO: Can this POC be made to be trivially copyable when all of the
  // properties in the list are trivially copyable?  The DPC++ implementation
  // seems to do this, so I think it should be possible.
  {
    sycl::khr::properties p{
      sycl::khr::alignment_type<float>,
      sycl::khr::enable_profiling{true},
      sycl::khr::threearg<1, false, 2>
    };
    static_assert(std::is_trivially_copyable_v<decltype(p)>);
  }
#endif

  // Check has_property and get_property on a compile-time property list.
  {
    sycl::khr::properties p{
      sycl::khr::alignment<16>,
      sycl::khr::alignment_type<int>,
      sycl::khr::threearg<1, false, 2>
    };
    static_assert(p.has_property<sycl::khr::alignment_key>() == true);
    static_assert(p.has_property<sycl::khr::alignment_type_key>() == true);
    static_assert(p.has_property<sycl::khr::threearg_key>() == true);
    static_assert(p.get_property<sycl::khr::alignment_key>().value == 16);
    static_assert(std::is_same_v<decltype(p.get_property<sycl::khr::alignment_type_key>())::value_t, int> == true);
    static_assert(p.get_property<sycl::khr::threearg_key>().x == 1);
    static_assert(p.get_property<sycl::khr::threearg_key>().y == false);
    static_assert(p.get_property<sycl::khr::threearg_key>().z == 2);
  }

  // Check has_property and get_property on a runtime property list.
  {
    sycl::khr::properties p{
      sycl::khr::enable_profiling{true},
      sycl::khr::twoarg{1, 2}
    };
    static_assert(p.has_property<sycl::khr::enable_profiling>() == true);
    static_assert(p.has_property<sycl::khr::twoarg>() == true);
    assert(p.get_property<sycl::khr::enable_profiling>().value == true);
    assert(p.get_property<sycl::khr::twoarg>().one == 1);
    assert(p.get_property<sycl::khr::twoarg>().two == 2);
  }

  // Check has_property and get_property on a hybrid property list.
  {
    sycl::khr::properties p{
      sycl::khr::hybrid<1>{2}
    };
    static_assert(p.has_property<sycl::khr::hybrid_key>() == true);
    assert(p.get_property<sycl::khr::hybrid_key>().x == 1);
    assert(p.get_property<sycl::khr::hybrid_key>().y == 2);
  }

  // Check has_property and get_property on a mixed property list.
  {
    sycl::khr::properties p{
      sycl::khr::enable_profiling{false},
      sycl::khr::alignment<8>,
      sycl::khr::hybrid<3>{4}
    };
    static_assert(p.has_property<sycl::khr::enable_profiling>() == true);
    static_assert(p.has_property<sycl::khr::alignment_key>() == true);
    static_assert(p.has_property<sycl::khr::hybrid_key>() == true);
    assert(p.get_property<sycl::khr::enable_profiling>().value == false);
    static_assert(p.get_property<sycl::khr::alignment_key>().value == 8);
    assert(p.get_property<sycl::khr::hybrid_key>().x == 3);
    assert(p.get_property<sycl::khr::hybrid_key>().y == 4);
  }

  // Check negative has_property
  {
    sycl::khr::properties p{
      sycl::khr::enable_profiling{false},
      sycl::khr::alignment<8>
    };
    static_assert(p.has_property<sycl::khr::twoarg>() == false);
    static_assert(p.has_property<sycl::khr::threearg_key>() == false);
    static_assert(p.has_property<sycl::khr::hybrid_key>() == false);
  }

  // Check constructing a queue with no properties
  {
    sycl::queue q1;
    sycl::queue q2{};
    sycl::queue q3{sycl::khr::properties{}};
  }

  // Check constructing a queue with a single property
  {
    sycl::queue q1{sycl::khr::enable_profiling{true}};
    sycl::khr::enable_profiling prof{false};
    sycl::queue q2{prof};
  }

  // Check constructing a queue with a property list
  {
    sycl::queue q1{sycl::khr::properties{
      sycl::khr::enable_profiling{true},
      sycl::khr::twoarg{1, 2}
    }};
    sycl::khr::properties p{
      sycl::khr::enable_profiling{false},
      sycl::khr::twoarg{3, 4}
    };
    sycl::queue q2{p};
  }

  // Check annotated_ptr with no properties
  {
    int x;
    sycl::khr::annotated_ptr aptr{&x};
    static_assert(aptr.has_property<sycl::khr::alignment_key>() == false);
    static_assert(aptr.has_property<sycl::khr::threearg_key>() == false);
  }

  // Check annotated_ptr with one property
  {
    int x;
    sycl::khr::annotated_ptr aptr{&x, sycl::khr::alignment<16>};
    static_assert(aptr.has_property<sycl::khr::alignment_key>() == true);
    static_assert(aptr.has_property<sycl::khr::threearg_key>() == false);
    static_assert(aptr.get_property<sycl::khr::alignment_key>().value == 16);
  }

  // Check annotated_ptr with property list
  {
    int x;
    sycl::khr::annotated_ptr aptr{&x, sycl::khr::properties{
      sycl::khr::alignment_type<float>,
      sycl::khr::threearg<0, true, 3>
    }};
    static_assert(aptr.has_property<sycl::khr::alignment_type_key>() == true);
    static_assert(aptr.has_property<sycl::khr::threearg_key>() == true);
    static_assert(std::is_same_v<decltype(aptr.get_property<sycl::khr::alignment_type_key>())::value_t, float> == true);
    static_assert(aptr.get_property<sycl::khr::threearg_key>().x == 0);
    static_assert(aptr.get_property<sycl::khr::threearg_key>().y == true);
    static_assert(aptr.get_property<sycl::khr::threearg_key>().z == 3);
  }

  // Check that adding a property to a property list uses the move constructor
  // if available.
  {
    struct copy_check : sycl::khr::detail::runtime_property<copy_check> {
      copy_check() : copy_count{0} {}
      copy_check(const copy_check& other) {
        copy_count = other.copy_count+1;
      }
      copy_check(copy_check&& other) {
        copy_count = other.copy_count;
      }
      int copy_count;
    };

    // Property list constructed with a temporary property object: uses the
    // move constructor to move the property into the list.  We get a "1" count
    // because "get_property" makes a copy when returning the property by value.
    sycl::khr::properties p1{copy_check{}};
    static_assert(p1.has_property<copy_check>() == true);
    assert(p1.get_property<copy_check>().copy_count == 1);

    // Property list constructed with a non-temporary property object: uses the
    // copy constructor to copy the property into the list.
    copy_check c;
    sycl::khr::properties p2{c};
    static_assert(p1.has_property<copy_check>() == true);
    assert(p2.get_property<copy_check>().copy_count == 2);
  }
}
