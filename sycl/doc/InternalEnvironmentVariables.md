# Internal Environment Variables

**Warning:** the environment variables described in this document are used for
development and debugging of DPC++ compiler and runtime. Their semantics are
subject to change. Do not rely on these variables in production code.

| Environment variable | Values | Description |
| -------------------- | ------ | ----------- |
| `INTEL_ENABLE_OFFLOAD_ANNOTATIONS` | Any(\*) | Enables ITT Annotations support for SYCL runtime. This variable should only be used by tools, that support ITT Annotations. |

`(*) Note: Any means this environment variable is effective when set to any non-null value.`

