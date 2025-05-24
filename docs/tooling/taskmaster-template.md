# Task Master Configuration Template

This template shows how to configure Task Master for the AHGD ETL project. Copy this to `.taskmasterconfig` in the project root and customize as needed.

```json
{
    "api_key": "<YOUR_API_KEY>",
    "model": "gpt-4-turbo",
    "project_name": "AHGD-ETL",
    "project_root": "/path/to/AHGD",
    "tasks_directory": "tasks",
    "prd_file": "scripts/prd.txt",
    "output_format": "markdown",
    "complexity_threshold": 5,
    "dependencies": {
        "validate": true,
        "auto_fix": false
    },
    "task_types": [
        "feature",
        "bugfix",
        "refactor",
        "documentation",
        "test"
    ],
    "generate_subtasks": true
}
```

## Important Configuration Notes

1. Do not commit your actual `.taskmasterconfig` file, especially with real API keys
2. Update `project_root` to match your local development environment
3. Customize `task_types` to match your project's task categorization needs
4. Set `complexity_threshold` between 1-10 (higher means more subtasks will be generated)
5. The `prd_file` should point to your Project Requirements Document if available