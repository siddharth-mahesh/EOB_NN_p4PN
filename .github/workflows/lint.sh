#!/bin/bash
# Comprehensive documentation, type, and style checks for the project.

set -e

# Get the directory where the script is located
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
VENV_BIN="$SCRIPTPATH/.venv/bin"

# Target file to check
TARGET="EOB_NN_p4PN/dho_example/damped_oscillators_symbolic_regression.py"

# Function to run a check and record its status
run_check() {
    local name=$1
    local cmd=$2
    echo "=== Running $name ==="
    # Execute the command
    if $cmd; then
        echo -e "\033[32m✔ $name passed\033[0m"
        return 0
    else
        echo -e "\033[31m✘ $name failed\033[0m"
        return 1
    fi
}

# Check if the virtual environment exists
if [ ! -d "$VENV_BIN" ]; then
    echo "Virtual environment not found at $VENV_BIN. Defaulting to system-wide tools..."
    VENV_BIN=""
fi

# Determine the absolute paths to the check binaries
PYLINT="${VENV_BIN:+$VENV_BIN/}pylint"
MYPY="${VENV_BIN:+$VENV_BIN/}mypy"
PYDOCSTYLE="${VENV_BIN:+$VENV_BIN/}pydocstyle"
DARGLINT="${VENV_BIN:+$VENV_BIN/}darglint"

FAILED=0

# Run each check sequentially
run_check "Pylint" "$PYLINT $TARGET" || FAILED=1
echo ""

run_check "Mypy" "$MYPY --no-incremental --config-file .mypy.ini $TARGET" || FAILED=1
echo ""

run_check "Pydocstyle" "$PYDOCSTYLE $TARGET" || FAILED=1
echo ""

run_check "Darglint" "$DARGLINT $TARGET" || FAILED=1
echo ""

# Summary report
if [ $FAILED -eq 0 ]; then
    echo -e "\n\033[32m All document and style checks passed successfully!\033[0m"
    exit 0
else
    echo -e "\n\033[31m Some checks failed. Please review the reports above for specific errors.\033[0m"
    exit 1
fi
