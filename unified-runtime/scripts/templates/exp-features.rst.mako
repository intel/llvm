<%
import os
import glob

expdocs = []
for section in sections:
    foundDocs = glob.glob(section + "/EXP-*.rst")
    for found in sorted(foundDocs):
        expdocs.append(found)
%>
=====================
Experimental Features
=====================

.. warning::

    Experimental features:

    *   May be replaced, updated, or removed at any time.
    *   Do not require maintaining API/ABI stability of their own additions
        over time.
    *   Do not require conformance testing of their own additions.

More information about experimental features can be found in :ref:`core/CONTRIB:Experimental Features`.

.. toctree::

%for expdoc in expdocs:
%if os.path.exists(os.path.join(sourcepath, expdoc)):
    ${expdoc}
%endif
%endfor
