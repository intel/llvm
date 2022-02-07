# Lifetime of an Extension

This document describes the process for creating and maintaining SYCL extension
documents over their lifetime.


## Creating a new extension

Start by making a copy of the [template][1] extension specification document,
and follow the instructions in that document.  Your extension should also
follow the rules in [chapter 6][2] of the SYCL specification, including the
"Guidelines for portable extensions".  These rules require you to choose a
`<vendorstring>`.  For DPC++, we use the string "ONEAPI" unless the extension
is very specific to Intel hardware in which case we use the string "INTEL".
The template uses the string "ONEAPI", so you must change occurrences of that
string if your extension is specific to Intel hardware.

[1]: <template.asciidoc>
[2]: <https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#chapter.extensions>

Some sections in the template are optional.  Remove these sections if they are
not needed.  You should also remove any of the explanatory text (generally in
italics).

Each extension must also have a name.  The template uses "MYEXTENSION", which
must be changed to the name of your extension.  Use uppercase and separate
words with underbars, since the name is also used as a C++ macro.  The name of
the specification file should match the name of the extension, for example
"SYCL\_EXT\_ONEAPI\_MYEXTENSION.asciidoc".

Usually new extensions are first created in the "proposed" state, so the
document should be added to that directory.  However, it is also possible to
add a new extension at the same time as its implementation, in which case the
specification should be added to either the "supported" or "experimental"
directories.

While an extension is in the "proposed" state, it is perfectly OK to make
further modifications to its specification.  There is no need to change the
version of the extension's feature-test macro when this occurs.


## Implementing an extension

Often, an extension is implemented sometime after it is proposed.  When this
happens, the PR that implements the extension should also move the
specification to either the "supported" or "experimental" directory, as
appropriate.  It is common to make small change to the specification when it is
implemented, so the PR that implements the extension may also make
modifications to the specification document.

Be sure to change the text in the "Status" section when the extension is
implemented.  See the [template][1] for the proper text.

Sometimes an extension is implemented with multiple PRs.  When this happens,
the last PR that implements the extension should also move the specification
document.  We want the specification document to reflect the features that are
implemented in DPC++, so a specification should not be moved to "supported" or
"experimental" before the final PR that implements it.

Ideally, all APIs in an extension should be implemented by the time we announce
support.  If this is not possible, something must be done to ensure that the
specification is an accurate description of what is implemented.  Following are
some techniques to accomplish this.

### Split the specification into versions

This is the preferred technique if the first release of an extension implements
some APIs but not others.  In this case, the extension document should be
copied to the "supported" (or "experimental") directory, but the description of
the unimplemented APIs should be removed from this copy.  Thus, the document in
that directory is an accurate description of the implementation.

The original version of the specification in the "proposed" folder should
remain.  In addition, a new "version" row should be added to the table that
describes the feature-test macro, and all the unimplemented APIs become part of
"version 2" of the specification.  These APIs can be implemented later,
following the normal process of [adding a new version to an existing
extension][3].

[3]: <#adding-a-new-version-to-an-existing-extension>

### Add NOTEs describing what is not implemented

Sometimes all of the APIs in an extension are implemented, but they are not yet
implemented for all devices or backends.  When this happens, we prefer to add
non-normative "notes" to the extension specification indicating what is not
yet implemented.  The placement of these notes depends on the nature of the
unimplemented thing.  For example, if the entire extension is unimplemented on
a certain backend, a note should be added in the "Status" section of the
document, as demonstrated in the [template][1].  If there are restrictions with
certain APIs, a note should be added near the description of each such API.


## Adding a new version to an existing extension

It is common to add new APIs to an extension after it is first released.  When
this happens, the new APIs should be protected by a new version of the
extension's feature-test macro.  This allows an application to test the value
of the macro to know whether the implementation supports the API.

Assuming the extension document is currently in the "supported" directory, make
a copy of that document in the "proposed" directory.  Update the "Status"
section as shown in the [template][1], and add a new "version" row to the table
that describes the feature-test macro with a short summary of the new APIs
enabled in that version.  The description of each new API should contain a
statement saying which version adds the API.  For example,

> This API is available starting in version 2 of this specification.

Avoid unnecessary reformatting of the extension after it is copied.  It should
be possible to see the new APIs that are proposed in the new version by using a
command like:

```
$ git diff {supported,proposed}/SYCL_EXT_ONEAPI_MYEXTENSION.asciidoc
```

When the new version of the extension is implemented, the "proposed" version of
the specification should be moved back to the "supported" directory,
overwriting the previous version.

Note that a new version of a supported extension should never remove any
functionality from the previous version.  We expect existing code that uses the
old version to still work with the new version.


## Deprecating an extension

Occasionally, we may decide to deprecate a supported extension.  For example,
this might happen if an extension is adopted into a new version of the core
SYCL specification.  When this happens, the specification is moved from the
"supported" directory to the "deprecated" directory, and the "Status" section
is changed as shown in the [template][1].  A signpost file is also added to
the "supported" directory with the same name as the original file and content
that looks like:

```
This extension has been deprecated, but the specification is still available
link:../deprecated/SYCL_EXT_ONEAPI_MYEXTENSION.asciidoc[here].
```

The purpose of the signpost file is to ensure that external links to the
extension are not broken, while still making it obvious that the extension is
now deprecated.

Note that a deprecated extension is still supported, so the implementation is
not removed.

We usually do not deprecate experimental extensions since there is no guarantee
that these extension remain supported from one DPC++ release to the next.
Instead, these extensions can be removed without a deprecation period.


## Removing support for an extension

Eventually, we typically remove an extension some time after it is deprecated.
When this happens, we move the specification file to the "removed" directory
and update the "Status" section as shown in the [template][1].  We also remove
the signpost file.  This typically happens in the same PR that removes the
implementation of the extension.


## Experimental extensions

The process of creating and implementing an "experimental" extension has mostly
been described already, but there are some additional things to keep in mind.
Even though an extension may be experimental, we still want the specification
to accurately describe the API.  Usually, the extension document is the main
user-facing description of the API, so it must be accurate in order for
customers to use the extension.  Therefore, even an experimental extension
specification must contain [NOTEs][4] describing any APIs that are not yet
implemented.

[4]: <#add-notes-describing-what-is-not-implemented>

Since experimental extensions have no guaranteed compatibility from one DPC++
release to another, we typically do not bother to add versions to the
feature-test macro.  This is still allowed, of course, but it is also OK to
add, remove, or modify APIs without changing the version.
