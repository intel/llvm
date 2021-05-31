(RFC?) SYCL namespaces refactoring
===================================

# Context
Currently the implementation provides users with SYCL API available in
`::cl::sycl`. Users can also access SYCL API by using only `::sycl` namespace,
this is achieved by marking `cl` namespace with `inline` keyword.

Simplified example of `CL/sycl.hpp` looks like:

```c++

// CL/sycl.hpp :

#inlcude <sycl/context.hpp>

// CL/sycl/context.hpp :

inline namespace cl {
namespace sycl {
  class context {...};
}
}

```

The SYCL2020 specification introduces a new header `sycl/sycl.hpp` which should
provide SYCL2020 API in the `::sycl` namespace. The SYCL2020 API partially
overlaps with SYCL1.2.1 API and extensions DPCPP implementation has. So, this
means that `sycl/sycl.hpp` must expose a lot of existing APIs but in `sycl`
namespace. In other words `::cl` namespace should not be visible if a user
includes `sycl/sycl.hpp`.

# Proposal

It cannot be done by just removing `inline namespace cl` from `context.hpp`-like
files because SYCL types are part of `libsycl.so` interface and changing their
mangling would break backward compatibility.

To overcome this problem symbol aliases can be used:

```
namespace OLD_NAMESPACE {
  void foo() {}
}

namespace NEW_NAMESPACE {
  void glob_foo() __attribute__ ((alias ("_ZN13OLD_NAMESPACE3fooEv")));
}
```
In case of msvc toolchain we can use a linker option:
```
#pragma comment(linker, "/export:?glob_foo@NEW_NAMESPACE@@YAXXZ=?foo@OLD_NAMESPACE@@YAXXZ")
```

Doing this would allow us to refactor namespaces to meet SYCL2020 requirements
while maintaining backward compatibility. And since we are going to change
namespace it might be useful to make additional adjustments not directly related
to getting rid of `::cl` namespace.

One of the additional modification is to decouple public interface(one which
is available to the users) and internal details, so we will be able to change
public interfaces however we want without impact on ABI and internal details.

The second additional modification is to add versioning namespaces. It's good
thing to have to simplify maintenance of two version of the same function.

So, the complete example would be:

```
// sycl/sycl.hpp :

  #inlcude <sycl/context.hpp>

// sycl/context.hpp :

  namespace __sycl_internal {
    namespace __v1 {
      class context {...};
    }
  }

  namespace sycl {
    using __sycl_internal::__v1::context;
  }

// CL/sycl.hpp :

  #inlcude <sycl/sycl.hpp>

  inline namespace cl {
    using sycl;
  }

```

# Concerns:

1. An application, which exposes interfaces which have SYCL types, compiled
   with the version of SYCL headers before the refactoring won't be compatible
   with an application compiled with new version of headers.

   Workaround: recompile the application or use alias approach mentioned above.

2. We need to be careful to not break ADL.


