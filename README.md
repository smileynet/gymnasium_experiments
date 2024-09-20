
# Gymnasium Experiments

A collection of gym-type environment related scripts.

## Installation and Setup

This project uses `uv` for dependency management and `ruff` for linting. Here's how to set up your development environment:

### 1. Install uv

First, install `uv` using the official installation script:

```bash
`curl -LsSf https://astral.sh/uv/install.sh | sh`
```

For alternative installation methods, visit the [uv documentation](https://github.com/astral-sh/uv).

### 2. Set up the project

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/gymnasium-experiments.git
cd gymnasium-experiments
```

### 3. Create a virtual environment and install dependencies

Use `uv` to create a virtual environment and install the project dependencies:

```bash
uv venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
uv sync
```

This will install all the dependencies specified in the `pyproject.toml` file.

### 4. Install development dependencies

Install the development dependencies (including `ruff`) using:

```bash
uv pip install --dev
```

## Using ruff

`ruff` is a fast Python linter. To run `ruff` on your project:

```bash
ruff check .
```

To automatically fix issues:

```bash
ruff check --fix .
```

## pyproject.toml

The `pyproject.toml` file in this project defines the project metadata, dependencies, and tool configurations. Here's a brief overview:

- `[project]`: Defines project metadata and dependencies.
- `[tool.uv]`: Specifies development dependencies managed by `uv`.

To add new dependencies, simply edit the `pyproject.toml` file and run `uv pip install -e .` again.

## Running the project

_TODO:_ Add specific script instructions