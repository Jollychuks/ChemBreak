# GitHub Setup for ChemBreak Colab Generator V3

The V3 notebook uses GitHub as the source for the project files.

Default repository:

`https://github.com/Jollychuks/ChemBreak.git`

Default project folder inside the repository:

`ChemBreak_Colab_Generator_v3`

Colab clones the repository to:

`/content/chembreak_repo`

The active V3 project directory becomes:

`/content/chembreak_repo/ChemBreak_Colab_Generator_v3`

## Important

Cloning does not create another GitHub repository.

It creates a temporary working copy inside the Colab runtime.

When the runtime is deleted, `/content/chembreak_repo` disappears.

## Generated datasets

V3 initially writes generated files into the temporary cloned folder.

For example:

`/content/chembreak_repo/ChemBreak_Colab_Generator_v3/candidate_tasks.csv`

To keep generated files permanently:
- push them back to GitHub using the optional checkpoint cell, or
- download them, or
- use another persistent storage location.

A GitHub token is needed only if you choose to push from Colab. It is not an LLM API key and does not create LLM usage charges.

Never hard-code a GitHub token in the repository.
