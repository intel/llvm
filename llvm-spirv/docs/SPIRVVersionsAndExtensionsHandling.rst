=======================================
SPIR-V versions and extensions handling
=======================================

.. contents::
   :local:

Overview
========

This document describes how the translator makes decisions about using
instructions from different version of the SPIR-V core and extension
specifications.

Being able to control the resulting SPIR-V version is important: the target
consumer might be quite old, without support for new SPIR-V versions and there
must be the possibility to control which version of the SPIR-V specification
that will be used during translation.

SPIR-V extensions is another thing which must be controllable. Extensions
can update and re-define semantics and validation rules for existing SPIR-V
entries and it is important to ensure that the translator is able to generate
valid SPIR-V according to the core spec, without uses of any extensions if such
SPIR-V was requested by user.

For example, without such infrastructure it is impossible to disable use of
``SPV_KHR_no_integer_wrap_decoration`` - it will be always generated if
corresponding LLVM IR counterparts are encountered in input module.

It is worth mentioning that SPIR-V versions and extensions the handling of
SPIR-V versions and extension is mostly important for the SPIR-V generation
step. On the consumer side it is the responsibility of the consumer to analyze
the incoming SPIR-V file and reject it if it contains something that is not
supported by the consumer.

However, translator simplifies this step for downstream users by checking
version and extensions in SPIR-V module during ``readSpirv``/``readSpirvModule``
phases.

SPIR-V Versions
===============

SPIR-V Generation step
----------------------

By default translator selects version of generated SPIR-V file based on features
used in this file. For example, if it contains ``dereferencable`` LLVM IR
attribute, ``MaxByteOffset`` decoration will be generated and resulting SPIR-V
version will be raised to 1.1.

.. note::
   There is no documentation about which exact features from newest
   SPIR-V spec versions will be used by the translator. If you are interested
   when or why a particular SPIR-V instruction is generated, please check this
   in the source code. Consider this as an implementation detail and if you
   disagree with something, you can always open an issue or submit pull request
   - contributions are welcome!

There is one option to control the behavior of the translator with respect to
the version of the SPIR-V file which is being generated/consumed.

* ``--spirv-max-version=`` - instructs the translator to generate SPIR-V file
  corresponding to any spec version which is less than or equal to the
  specified one. Behavior of the translator is the same as by default with only
  one exception: resulting SPIR-V version cannot be raised higher than
  specified by this option.

Allowed values are ``1.0``, ``1.1``, ``1.2``, ``1.3``, and ``1.4``.

.. warning::
   These two options are mutually exclusive and cannot be specified at the
   same time.

If the translator encounters something that cannot be represented by set of
allowed SPIR-V versions (which might contain only one version), it does one of
the following things:

* ignores LLVM IR entity in the input file.

  For example, ``dereferencable`` LLVM IR attribute can be ignored if it is not
  allowed to generate SPIR-V 1.1 and higher.

* tries to represent LLVM IR entity with allowed instructions.

  For example, ``OpPtrEqual`` can be used if SPIR-V 1.4 is not allowed and can
  be emulated via ``OpConvertPtrToU`` + ``OpIEqual`` sequence.

* emits error if LLVM IR entity cannot be ignored and cannot be emulated using
  available instructions.

  For example, if global constructors/destructors
  (represented by @llvm.global_ctors/@llvm.global_dtors) are present in a module
  then the translator should emit error if it cannot use SPIR-V 1.1 and higher
  where ``Initializer`` and ``Finalizer`` execution modes are described.

SPIR-V Consumption step
-----------------------

By default, translator consumes SPIR-V of any version which is supported.

This behavior, however, can be controlled via the same switches described in
the previous section.

If one of the switches present and translator encountered SPIR-V file
corresponding to a spec version which is not included into set of allowed
SPIR-V versions, translator emits error.

SPIR-V Extensions
=================

SPIR-V Generation step
----------------------

By default, translator doesn't use any extensions. If it required to enable
certain extension, the following command line option can be used:

* ``--spirv-ext=`` - allows to control list of allowed/disallowed extensions.

Valid value for this option is comma-separated list of extension names prefixed
with ``+`` or ``-`` - plus means allow to use extension, minus means disallow
to use extension. There is one more special value which can be used as extension
name in this option: ``all`` - it affects all extension which are known to the
translator.

If ``--spirv-ext`` contains name of extension which is not know for the
translator, it will emit error.

Examples:

* ``--spirv-ext=+SPV_KHR_no_integer_wrap_decoration,+SPV_INTEL_subgroups``
* ``--spirv-ext=+all,-SPV_INTEL_fpga_loop_controls``

.. warning::
   Extension name cannot be allowed and disallowed at the same time: for inputs
   like ``--spirv-ext=+SPV_INTEL_subgroups,-SPV_INTEL_subgroups`` translator
   will emit error about invalid arguments.

.. note::
   Since by default during SPIR-V generation all extensions are disabled, this
   means that ``-all,`` is implicitly added at the beggining of the
   ``-spirv-ext`` value.

If the translator encounters something that cannot be represented by set of
allowed SPIR-V extensions (which might be empty), it does one of the following
things:

* ignores LLVM IR entity in the input file.

  For example, ``nsw``/``nuw`` LLVM IR attributes can be ignored if it is not
  allowed to generate SPIR-V 1.4 and ``SPV_KHR_no_integer_wrap_decoration``
  extension is disallowed.

* tries to represent LLVM IR entity with allowed instructions.

  Translator could translate calls to a new built-in functions defined by some
  extensions as usual call instructions without using special SPIR-V
  instructions.

  However, this could result in a strange SPIR-V and most likely will lead to
  errors during consumption. Having that, translator should emit errors if it
  encounters a call to a built-in function from an extension which must be
  represented as a special SPIR-V instruction from extension which wasn't
  allowed to be used. I.e. if translator knows that this certain LLVM IR entity
  belongs to an extension functionality and this extension is disallowed, it
  should emit error rather than emulating it.

* emits error if LLVM IR entity cannot be ignored and cannot be emulated using
  available instructions.

  For example, new built-in types defined by
  ``cl_intel_device_side_avc_motion_estimation`` cannot be represented in SPIR-V
  if ``SPV_INTEL_device_side_avc_motion_estimation`` is disallowed.

SPIR-V Consumption step
-----------------------

By default, translator consumes SPIR-V regardless of list extensions which are
used by the input file, i.e. all extensions are allowed by default during
consumption step.

.. note::
   This is opposite to the generation step and this is done on purpose: to not
   broke workflows of existing users of the translator.

.. note::
   Since by default during SPIR-V consumption all extensions are enabled, this
   means that ``+all,`` is implicitly added at the beggining of the
   ``-spirv-ext`` value.

This behavior, however, can be controlled via the same switches described in
the previous section.

If ``--spirv-ext`` switch presents, translator will emit error if it finds out
that input SPIR-V file uses disallowed extension.

.. note::
   If the translator encounters unknown extension in the input SPIR-V file, it
   will emit error regardless of ``-spirv-ext`` option value.

If one of the switches present and translator encountered SPIR-V file
corresponding to a spec version which is not included into set of allowed
SPIR-V versions, translator emits error.

How to control translator behavior when using it as library
===========================================================

When using translator as library it can be controlled via bunch of alternative
APIs that have additional argument: ``TranslatorOpts`` object which
encapsulates information about available SPIR-V versions and extensions.

List of new APIs is: ``readSpirvModule``, ``writeSpirv`` and ``readSpirv``.

.. note::
   See ``LLVMSPIRVOpts.h`` for more details.

How to get ``TranslatorOpts`` object
------------------------------------

1. Default constructor. Equal to:

   ``--spirv-max-version=MaxKnownVersion --spirv-ext=-all``

   .. note::
      There is method ``TranslatorOpts::enableAllExtensions()`` that allows you
      to quickly enable all known extensions if it is needed.

2. Constructor which accepts all parameters

   Consumes both max SPIR-V version and optional map with extensions status
   (i.e. which one is allowed and which one is disallowed)

Extensions status map
^^^^^^^^^^^^^^^^^^^^^

This map is defined as ``std::map<ExtensionID, bool>`` and it is intended to
show which extension is allowed to be used (``true`` as value) and which is not
(``false`` as value).

.. note::
   If certain ``ExtensionID`` value is missed in the map, it automatically means
   that extension is not allowed to be used.

   This implies that by default, all extensions are disallowed.
