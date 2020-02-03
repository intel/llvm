// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

void defined() {
  // empty
}

void undefined();
// expected-note@-1 {{'undefined' declared here}}

SYCL_EXTERNAL void undefinedExternal();

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
  // expected-note@-1 {{called by 'kernel_single_task<CallToUndefinedFnTester,}}
}

template <typename T>
void definedTpl() {
  // empty
}

template <typename T>
void undefinedTpl();
// expected-note@-1 {{'undefinedTpl<int>' declared here}}

template <typename T>
extern SYCL_EXTERNAL void undefinedExternalTpl();

template <typename T, bool X>
void definedPartialTpl() {
  // empty
}

template<>
void definedPartialTpl<char, true>() {
  // empty
}

template <typename T, bool X>
struct Tpl {
  void defined() {
    // empty
  }
};

template <typename T>
struct Tpl<T, true> {
  void defined() {
    // empty
  }
};

template <typename T, bool X>
struct TplWithTplMethod {
  template <typename T2, bool Y>
  void defined() {
    // empty
  }
};

template <typename T>
struct TplWithTplMethod<T, true> {
  template <typename T2, bool Y>
  void defined() {
    // empty
  }
};

template <typename T, bool X>
struct TplWithTplMethod2 {
  template <typename T2, bool Y>
  void defined() {
    // empty
  }

  template <>
  void defined<char, true>() {
    // empty
  }
};

template <typename T>
struct TplWithTplMethod2<T, true> {
  template <typename T2, bool Y>
  void defined() {
    // empty
  }

  template <>
  void defined<char, true>() {
    // empty
  }
};

int main() {
  kernel_single_task<class CallToUndefinedFnTester>([]() {
    defined();
    undefinedExternal();
    undefined();
    // expected-error@-1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}

    definedTpl<int>();
    undefinedExternalTpl<int>();
    undefinedTpl<int>();
    // expected-error@-1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}

    {
      Tpl<int, false> tpl;
      tpl.defined();
    }

    {
      Tpl<int, true> tpl;
      tpl.defined();
    }

    {
      TplWithTplMethod<int, false> tpl;
      tpl.defined<char, false>();
      tpl.defined<char, true>();
      tpl.defined<int, false>();
      tpl.defined<int, true>();
    }

    {
      TplWithTplMethod<int, true> tpl;
      tpl.defined<char, false>();
      tpl.defined<char, true>();
      tpl.defined<int, false>();
      tpl.defined<int, true>();
    }

    {
      TplWithTplMethod2<int, false> tpl;
      tpl.defined<char, false>();
      tpl.defined<char, true>();
      tpl.defined<int, false>();
      tpl.defined<int, true>();
    }

    {
      TplWithTplMethod2<int, true> tpl;
      tpl.defined<char, false>();
      tpl.defined<char, true>();
      tpl.defined<int, false>();
      tpl.defined<int, true>();
    }

    definedPartialTpl<int, false>();
    definedPartialTpl<int, true>();
    definedPartialTpl<char, false>();
    definedPartialTpl<char, true>();
  });
}
