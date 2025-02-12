# Considerations for working on KHR extensions

SYCL specification evolves through embedding extensions developed by various
vendors, including Khronos Group itself (`khr` extensions).

In order for a KHR extension to be accepted, there must be CTS tests for it and
at least one implementation which passes them.

Considering that KHR extensions are being developed in public, we can start
prototyping them as soon as corresponding PR for an extension is published at
KhronosGroup/SYCL-Docs.

However, we shouldn't be exposing those extensions to end users until the
extension if finalised, ratified and published by Khronos - due to risk of an
extension changing during that process and lack of the officially published
version of it.

So, we can have a PR but can't merge it. Keeping PRs opened for a long time is a
bad practice, because they tend to get stale: there are merge conflicts,
potential functional issues due to the codebase changes, etc.

In order for us to avoid stale PRs, all functionality which is a public
interface of an "in-progress" KHR extension, must be hidden under
`__DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS` macro. That way we can merge a PR to
avoid constantly maintaining it in a good shape, start automatically testing it
but at the same time avoid exposing incomplete and/or undocumented feature to
end users just yet.

"in-progress" KHR extension term used above is defined as:
- PR proposing a KHR extension has not been merged/cherry-picked to `sycl-2020`
  branch of KhronosGroup/SYCL-Docs.

  That only happens after all formal processes on Khronos Group side are
  completed so an extension can be considered good and stable to be released by
  us.

  Note: merge of an extension proposal PR into `main` branch of
  KhronosGroup/SYCL-Docs repo is **not** enough.
- Published (i.e. the above bullet complete) KHR extension, which hasn't been
  fully implemented by us

The macro is **not** intended to be used by end users and its purpose is to
simplify our development process by allowing us to merge implementation (full
or partial) of the aforementioned extensions earlier to simplify maintenance and
enable automated testing.

Due to this reason, we are not providing a separate macro for each "in-progress"
KHR extension we may (partially) support, but just a single guard.
