<%!
import re
from templates import helper as th
%><%
%>

==============================
${groupname} API
==============================
oneAPI Runtime Specification - Version ${rev}

%for s in specs:
<%
    name = s['name']
    title = name.capitalize()
    header = s['header']
    objects = s['objects']
    needstitle = True
%>\
#################################################################
## Generate section title and links table
#################################################################
## -------------------------
## Functions
## -------------------------
 <%isempty = True%>
%for obj in objects:
%if re.match(r"function", obj['type']) and ("condition" not in obj):
%if isempty: # only display section title if there is content.
%if needstitle:
<%needstitle = False%>
${title}
============================================================
%endif
* Functions

<%isempty = False%>
%endif
    * :ref:`${th.make_func_name(n, tags, obj).replace("_", "-")}`
%endif
%endfor # obj in objects

#################################################################
## -------------------------
## Enums
## -------------------------
 <%isempty = True%>
%for obj in objects:
%if re.match(r"enum", obj['type']):
%if isempty: # only display section title if there is content.
%if needstitle:
<%needstitle = False%>
${title}
============================================================
%endif
* Enumerations

<%isempty = False%>
%endif
    * :ref:`${th.make_type_name(n, tags, obj).replace("_", "-")}`
%endif
%endfor # obj in objects

#################################################################
## -------------------------
## Structs/Unions
## -------------------------
 <%isempty = True%>
%for obj in objects:
%if re.match(r"struct|union", obj['type']):
%if isempty: # only display section title if there is content.
%if needstitle:
<%needstitle = False%>
${title}
============================================================
%endif
* Structures

<%isempty = False%>
%endif
    * :ref:`${th.make_type_name(n, tags, obj).replace("_", "-")}`
%endif
%endfor # obj in objects

#################################################################
## Generate API documentation
#################################################################
## -------------------------
## Functions
## -------------------------
<%isempty = True%>
%for obj in objects:
%if re.match(r"function", obj['type']) and ("condition" not in obj):
%if isempty: # only display section title if there is content.
${title} Functions
------------------------------------------------------------------------------
<%isempty = False%>
%endif

.. _${th.make_func_name(n, tags, obj).replace("_", "-")}:

${th.make_func_name(n, tags, obj)}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: ${th.make_func_name(n, tags, obj)}
    :project: UnifiedRuntime

%endif
%endfor # obj in objects

#################################################################
## -------------------------
## Enums
## -------------------------
<%isempty = True%>
%for obj in objects:
%if re.match(r"enum", obj['type']):
%if isempty: # only display section title if there is content.
${title} Enums
------------------------------------------------------------------------------
<%isempty = False%>
%endif

.. _${th.make_type_name(n, tags, obj).replace("_", "-")}:

${th.make_type_name(n, tags, obj)}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: ${th.make_enum_name(n, tags, obj)}
    :project: UnifiedRuntime

%endif
%endfor # obj in objects
#################################################################
## -------------------------
## Structs/Unions
## -------------------------
 <%isempty = True%>
%for obj in objects:
%if re.match(r"struct|union", obj['type']):
%if isempty: # only display section title if there is content.
${title} Structures
------------------------------------------------------------------------------
<%isempty = False%>
%endif
.. _${th.make_type_name(n, tags, obj).replace("_", "-")}:

${th.make_type_name(n, tags, obj)}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

%if re.match(r"struct", obj['type']):
.. doxygenstruct:: ${th.make_type_name(n, tags, obj)}
    :project: UnifiedRuntime
    :members:
    :undoc-members:
%endif
%if re.match(r"union", obj['type']):
.. doxygenunion:: ${th.make_type_name(n, tags, obj)}
    :project: UnifiedRuntime
%endif

%endif
%endfor # obj in objects

%endfor # s in specs
