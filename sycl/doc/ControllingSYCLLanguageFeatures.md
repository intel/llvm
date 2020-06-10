# Controlling SYCL/DPC++ language features

This documents aims to describe a mechanism which is used to control (i.e.
enable or disable) different SYCL/DPC++ language and library features, like
version of SYCL/DPC++ standard to use, set of extensions to use, etc.

## Controlling SYCL/DPC++ standard version

In order to control the language standard to compile for, the following options
can be used:

`-sycl-std=<value>`, `--sycl-std=<value>`, `--sycl-std <value>`

SYCL language standard to compile for.

Possible values:

- `2017`, `121`, `1.2.1`, `sycl-1.2.1`: corresponds to SYCL 1.2.1
  specification, see [SYCL Registry] for more details and the specification
  text.

- `dpcpp-0.8`: corresponds to DPC++ version documented in oneAPI 0.8

  Basically, `-sycl-std=dpcpp-0.8` implies support for SYCL 1.2.1 specification
  and bunch of extension enabled.

  See [oneAPI Specification] for more details and the specification text.

[SYCL Registry]: https://www.khronos.org/registry/SYCL/
[oneAPI Specification]: https://spec.oneapi.com/

Note: `-sycl-std` **doesn't** imply any C++ standard version to be set,
which means that some default value will be selected. If by some reason, that
default version is not a desired one,  it is possible to change C++ version
independently of SYCL/DPC++ standard version via specifying additional option:
`-sycl-std=1.2.1 -std=c++14`, for example.

If SYCL/DPC++ standard version and C++ standard version (either default value
for the compiler or one which was explicitly set via `-std`) are incompatible,
then it is expected to see compilation errors. Incompatible means that C++
standard version is less than minimum required by SYCL/DPC++ standard.

`-std=<value>`, `--std=<value>`, `--std <value>`

One more way to specify SYCL/DPC++ standard version is to use a general clang
option, which allows to specify language standard to compile for.

Supported values (besides listed in clang documentation/help):

- `sycl-1.2.1`, `sycl-2017`: corresponds to `-sycl-std=1.2.1`
- `dpcpp-0.8`: corresponds to `-sycl-std=dpcpp-0.8`

Note: setting SYCL or DPC++ standard version via `-std` option automatically
implies some C++ standard version to be set, according to requirements of
corresponding SYCL/DPC++ specification. For example, for SYCL 1.2.1 it would be
at least C++11, while for DPC++ 0.8 it would be C++17.

Please note that if you specify `-std` flag several times, only the last
value takes effect. This means, that if you want to specify a particular C++
standard version instead of some default one implied by the SYCL/DPC++ standard,
you have to use two separate options: `-sycl-std` and `-std`.

## Controlling SYCL/DPC++ language extensions

Both SYCL and DPC++ has several extensions or proposals about how to expand
standard with new features, which might be vendor- or device-specific.

Each extension can be separately enabled or disabled depending on user
preferences. List of extensions enabled by default is controlled by SYCL/DPC++
standard version.

Enabling/disabling extensions is done via single `-sycl-ext` option. It accepts
comma-separated list of extension names prefixed with `+` or `-` to indicate
whether particular extension should be enabled or disabled.

Example: `-sycl-ext=+EXTENSION_NAME1,-EXTENSION_NAME2` - this option specifies
that `EXTENSION_NAME1` extension should be enabled and `EXTENSION_NAME2` should
be disabled.

When particular extension is enabled, the compiler automatically defines a macro
with the same name, i.e. if `-sycl-ext=EXTENSION_NAME1` command line option was
specified, then `EXTENSION_NAME1` macro will be defined.

**TODO**: update table with supported extensions with identifiers (macro or
compiler options), which should be used to enable/disable them.

### Materials for discussion

Details of controlling SYCL/DPC++ language extensions for sure are not settled
down yet and there are few questions and different ways of answering them.

Information below is provided to highlight main questions and pros and cons of
different solutions to select the best one.

#### Macro vs. compiler option for header-only extensions

There are extensions/proposals, which doesn't require any specific changes
in the compiler or underlying components, but just define some helpers or
sugar in the API and therefore, can be purely implemented in header files,
without even changing SYCL/DPC++ runtime. For example,
[SYCL_INTEL_device_specific_kernel_queries]

[SYCL_INTEL_device_specific_kernel_queries]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/DeviceSpecificKernelQueries/SYCL_INTEL_device_specific_kernel_queries.asciidoc

On the one hand, it shouldn't be necessary to modify the compiler in order to
enable/disable some new methods or types defined in some header file. It should
be enough to guard the content of a such file with some `#ifdef EXTENSION_NAME`
and user is expected to pass `-DEXTENSION_NAME` to compiler in order to use such
extension.

So, the main pros here is that no changes to the compiler is needed at all.

However, there are several downsides of that approach:

- According to [Predefined macros] section from [SYCL_INTEL_extension_api]
  proposal, if toolchain supports extension, it should _automatically_ define
  corresponding macro. So, if we require from user to specify the macro for an
  extension, it contradicts with the current proposal of extensions mechanism in
  SYCL.

- It is inconsistent for users (and they likely don't really care about how
  particular extension is designed internally) why some of them are enabled
  via some compiler flag and another ones are enabled via macro. Also,
  implementation of every extension might be changed in the future as extension
  and the whole SYCL implementation evolves over time.

- It is easy to make a typo in some extension name and instead of having one
  clear error that extension name is invalid, user will get bunch of errors that
  particular types/methods or functions are not available. For example:
  `-DSYCL_INTEL_SUBGRUOP_ALGORITHMS_ENABLE` - note the typo.


[SYCL_INTEL_extension_api]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/ExtensionMechanism/SYCL_INTEL_extension_api.asciidoc
[Predefined macros]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/ExtensionMechanism/SYCL_INTEL_extension_api.asciidoc#predefined-macros

#### -sycl-ext vs. -fsycl-extension-name/-fno-sycl-extension-name

Another one way to enable/disable extensions is to provide separate option
for each extensions.

There are pros and cons of both approaches:

##### Single flag, i.e. -sycl-ext

Pros:
- We don't have to hardcode list of known extensions and update the compiler
  for each new extension we have prepared: `-cl-std` works in this manner

  - This simplifies prototyping of new extensions, especially if they are
    header-only

- This allows to easily treat header-only extensions in the same way as
  actual compiler extension and change the implementation without any
  updates in compiler/toolchain interface (command line options)

- Enabling/disabling extension via this option could automatically define
  corresponding macro, which can be used by header-only extensions to hide
  new classes/APIs and other stuff introduced by the extension to avoid it
  being accidentally used

Cons:
- Without list of known extensions, we cannot emit proper diagnostic that some
  unknown extension was enabled just because of the typo in its spelling

  - Potentially, we could introduce one more option, which will allow to check
    that list of enabled/disabled extensions doesn't contain anything unknown,
    but this again means change of the compiler for each even header-only
    extension

- According to [Predefined macros] section from [SYCL_INTEL_extension_api]
  proposal, extension-specific macro should not only be defined, but also has a
  value in particular form. How can we automatically put any meaningful value
  in there without having predefined list of known extensions? Do we need to
  extend the format so user can specify which version of the extension is
  needed (`-sycl-ext=+EXTENSION_NAME1=123`)?

##### Separate flag for each extension

Generic form of command line option to control SYCL/DPC++ language extension
is `-f[no-]sycl-extension-name`. For example, `-fsycl-usm` flag enables support
for [USM] extension, `-fno-sycl-usm` disables it.

[USM]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc

Flags controlling the same extension might be set multiple times in the same
compiler invocation: the last one overrides previous flag.

Pros:
- We will automatically get a diagnostic if user made a typo in an option name,
  which corresponds to a particular extension

Cons:
- Seems like a significant amount of new flags coming to the compiler
- Harder to prototype and implement new extensions
- What about header-only extensions? Bunch of compiler options which are only
  used to define a macro doesn't look good
