.. Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
.. 
.. Licensed under the Apache License, Version 2.0 (the "License").
.. You may not use this file except in compliance with the License.
.. A copy of the License is located at
.. 
..     http://www.apache.org/licenses/LICENSE-2.0
.. 
.. or in the "license" file accompanying this file. This file is distributed 
.. on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
.. express or implied. See the License for the specific language governing 
.. permissions and limitations under the License.

{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
