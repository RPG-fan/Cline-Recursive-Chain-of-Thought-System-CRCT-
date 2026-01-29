"""
IO module for mini tracker specific data.
"""

from typing import Any, Dict


def get_mini_tracker_data() -> Dict[str, Any]:
    """Returns the data structure for mini tracker."""
    return {
        "template": """# Module: {module_name}

## Purpose & Responsibility
{{1-2 paragraphs on module purpose & responsibility}}

## Interfaces
* `{{InterfaceName}}`: {{purpose}}
* `{{Method1}}`: {{description}}
* `{{Method2}}`: {{description}}
* Input: [Data received]
* Output: [Data provided]
...

## Implementation Details
* Files: [List with 1-line descriptions]
* Important algorithms: [List with 1-line descriptions]
* Data Models
    * `{{Model1}}`: {{description}}
    * `{{Model2}}`: {{description}}

## Current Implementation Status
* Completed: [List of completed items]
* In Progress: [Current work]
* Pending: [Future work]

## Implementation Plans & Tasks
* `implementation_plan_{{filename1}}.md`
* [Task1]: {{brief description}}
* [Task2]: {{brief description}}
* `implementation_plan_{{filename2}}.md`
* [Task1]: {{brief description}}
* [Task2]: {{brief description}} 
...

## Mini Dependency Tracker
---mini_tracker_start---
""",
        "markers": ("---mini_tracker_start---", "---mini_tracker_end---"),
    }
