# Requirements

## API and Library Requirements

### API Namespace
* APIs will clearly define the functionality
* APIs will use a consistent style within the library
* APIs will not allow namespace collision when multiple libraries are linked in the same user application
* API will be consistent across libraries included in the product
* Will not allow namespace collision when multiple libraries are linked in the same user application
* Exceptions:
  + If two libraries are unlikely to used together, willing to sacrifice optimal performance
  + For existing libraries, favor API consistency vs. stylist name space

### Common Functional Requirements
* Libraries will use common resource allocation (memory, threads, device context) – under & above API where applicable
* Libraries will support consistent data types to avoid unnecessary data manipulation / reformatting
* Libraries will support a composable threading model and synchronous/asynchronous operations
* Libraries will utilize unified error reporting, debug information, I/O  mechanism (file, network,…), etc.

### Directory Structure and Deployment Requirements
* The Libraries can be delivered as “stand-alone” deliverables
* The Libraries can be delivered in a common product suite configuration (with other product components)
* When delivered in a product suite, the directory structure will conform to the master product directory layout requirements
* The library subcomponents will be installed in a consistent directory structure across the included libraries
* When two or more libraries are installed independently, they will conform to a common directory structure and versioning layout
* Libraries will be packaged for installation using common packaging mechanism for the operations systems that they targeted (e.g. .rpm, .deb, .msi, etc.)

### Library Language Binding Requirements
* DPC++ Language binding requirements
  + Performance Libraries that execute on multiple Intel IP’s shall support the DPCP++ language binding as the primary mechanism for programming on multiple Intel Hardware IP’s
  + If the library supports DPC++ for a only a subset of functions for offload to an accelerator (e.g. ATS), all CPU functions should all support DPC++ Language bindings so that application developers can write their entire application in DPC++
  + If a Library only supports only the CPU, but is likely to be used with another library the supports DPC++ on CPU and ATS, the library shall also support DPC++ 
* Libraries may support other language bindings (C/C++/FORTRAN/JAVA/PYTHON, etc.) to support existing user base and use cases required by the developer domain

### Library API Deprecation Management
* Library API deprecation will be managed via a change control process
* Definitions:
  + Supported – In the currently released library and will not change at short notice.
  + Deprecated – Documented to be no longer supported, and may be removed in the future at an announced date. Users are encouraged to use an alternative API or library
  + Removed – Removed from library, no longer supported or available

#### Objectives for Deprecation
* Refine and improve APIs to deliver better value for developer and Intel
  + API does not expose best performance or support important usage models
* Remove outdated or infrequently used functionality based on usage data. 

#### Multi year/release process
* Goal: Provide customers ample time to review, respond and adapt
  + Proactively communicate deprecation
  + Use warning and \#pragma per each deprecated item
  + Over the deprecation period: collect & analyze deprecation feedback & remove APIs
