#!/usr/bin/env bash
# Runs the test suite. PYTEST_DISABLE_PLUGIN_AUTOLOAD keeps pytest plugins that ROS puts on
# PYTHONPATH (launch_testing) from being auto-loaded into this unrelated project.
cd "$(dirname "$0")"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q "$@"
