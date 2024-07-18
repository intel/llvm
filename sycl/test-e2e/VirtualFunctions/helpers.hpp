#include <algorithm>
#include <type_traits>

// TODO: strictly speaking, selecting a max alignment here may not be always
// valid, but for test cases that we have now we expect alignment of all types
// to be the same.
// std::aligned_storage uses double under the hood which prevents us from
// using it on some HW. Therefore we use a custom implementation.
template <typename... T> struct aligned_storage {
  static constexpr size_t Len = std::max({sizeof(T)...});
  static constexpr size_t Align = std::max({alignof(T)...});

  struct type {
    alignas(Align) unsigned char data[Len];
  };
};

// Helper data structure that automatically creates a right (in terms of size
// and alignment) storage to accomodate a value of any of types T...
template <typename... T> struct obj_storage_t {
  static_assert(std::max({alignof(T)...}) == std::min({alignof(T)...}),
                "Unsupported alignment of input types");
  using type = typename aligned_storage<T...>::type;
  static constexpr size_t size = std::max({sizeof(T)...});

  type storage;

  template <typename RetT> RetT *construct(const unsigned int TypeIndex) {
    if (TypeIndex >= sizeof...(T)) {
#ifndef __SYCL_DEVICE_ONLY__
      assert(false && "Type index is invalid");
#endif
      return nullptr;
    }

    return constructHelper<RetT, T...>(TypeIndex, 0);
  }

private:
  template <typename RetT> RetT *constructHelper(const int, const int) {
    // Won't be ever called, but required to compile
    return nullptr;
  }

  template <typename RetT, typename Type, typename... Rest>
  RetT *constructHelper(const int TargetIndex, const int CurIndex) {
    if (TargetIndex != CurIndex)
      return constructHelper<RetT, Rest...>(TargetIndex, CurIndex + 1);

    RetT *Ptr = new (reinterpret_cast<Type *>(&storage)) Type;
    return Ptr;
  }
};
