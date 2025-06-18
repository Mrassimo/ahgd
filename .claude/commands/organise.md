# Repository Refinement Command: Organize and Standardize Repository

## 1. Repository Analysis

1.  **Analyze Current Structure:**
    * Map out the existing directory and file structure.
    * Identify the current organizational paradigm (e.g., by file type like `controllers`, `services`, `models` or by feature).
    * Determine the extent of deviation from a strict vertical slice architecture.

2.  **Establish Naming Conventions:**
    * Review existing file and directory names for inconsistencies.
    * Define a clear and consistent naming convention for all files and directories based on their vertical slice and role (e.g., `feature/user-authentication/user-authentication.service.ts`, `feature/order-processing/order.model.ts`).
    * Document these conventions in `CONTRIBUTING.md`.

3.  **Audit Documentation:**
    * Scan all Markdown (`.md`), AsciiDoc, or other documentation files for "stodge"—prose that is overly verbose, unclear, or offers little value.
    * Identify documentation that is out-of-date, referring to deprecated features, old workflows, or previous architectural designs.
    * Locate project setup, architecture, and decision logs.

## 2. Refactoring and Organization

1.  **Implement Vertical Slice Structure:**
    * Create a primary source directory (e.g., `src/features/` or `src/slices/`).
    * For each distinct feature or domain, create a dedicated subdirectory.
    * Move all related files (e.g., models, services, controllers, components, tests) for a specific feature into its corresponding vertical slice directory.

2.  **Standardize File and Directory Naming:**
    * Systematically rename all files and directories to conform to the conventions defined in the analysis phase.
    * Use a script or manual process to apply changes, ensuring consistency. For example:
        * `userController.js` -> `user-authentication.controller.js`
        * `views/login.html` -> `user-authentication/ui/login.view.html`

3.  **Update Internal Code References:**
    * Conduct a project-wide search for all import/export statements, file paths, and resource links.
    * Update all references to reflect the new file locations and names.
    * Run a static analysis tool or compiler/linter to catch broken references automatically.

## 3. Documentation Stewardship

1.  **Reduce Documentation Stodge:**
    * Refactor identified documentation to be concise and actionable.
    * Replace long paragraphs with bullet points, diagrams, or code examples where appropriate.
    * Ensure the purpose and context of each document are clear from the start.

2.  **Archive Outdated Documentation:**
    * Create a top-level directory named `documentation_archive/`.
    * Move all identified out-of-date documentation into this directory.
    * Add a `README.md` inside `documentation_archive/` explaining that the contents are for historical reference only and are not actively maintained.

3.  **Update Core Documentation:**
    * Modify the `README.md` to reflect the new repository structure and point to key architectural documents.
    * Update any `ARCHITECTURE.md` or similar high-level documents to describe the new vertical slice paradigm.

## 4. Final Reporting and Verification

1.  **Generate Impact Report:**
    * Create a new file named `REPO_REFINEMENT_LOG.md` in the root directory.
    * In this file, document all significant changes made:
        * **Structural Changes:** Provide a `before` and `after` tree view of the repository structure.
        * **File Migrations:** List all files that were moved or renamed, showing their old and new paths.
        * **Documentation Changes:** Summarize the documents that were refactored or archived.
        * **Potential Impact:** Detail any potential breaking changes, especially those that might affect build scripts, CI/CD pipelines, or external systems that depend on the old structure.

2.  **Validate Repository Integrity:**
    * Execute the full suite of automated tests (`unit`, `integration`, `e2e`) to confirm that no functionality has been broken.
    * Perform a clean build of the project to ensure all dependencies and paths are correctly resolved.
    * Manually review the application's critical paths if automated test coverage is insufficient.

## Guidelines

-   **DO NOT** mix organizational paradigms; commit fully to the vertical slice structure.
-   **DO** perform refactoring in a separate branch.
-   **ENSURE** all code references, import paths, and module declarations are updated.
-   **VERIFY** that CI/CD pipelines and build scripts are updated to accommodate the new structure.
-   **MAINTAIN** a clear record of all changes in the final `REPO_REFINEMENT_LOG.md`.
-   **PRESERVE** Git history when moving files by using `git mv` where possible.

### Summary of Deliverables After Completion:

1.  **Updated Files:** All source code files reorganized into vertical slices with standardized names. All documentation files refactored or archived.
2.  **New Files:** A `REPO_REFINEMENT_LOG.md` detailing all changes and their impact. A `README.md` within the `documentation_archive/` directory.
3.  **Architectural Changes:** The repository is now organized by feature-centric vertical slices.
4.  **Status:** The repository is fully functional, with a more maintainable and intuitive structure. All tests are passing.