"""
Tests for CI workflow configuration.

These tests verify that the GitHub Actions CI workflow is properly configured
following TDD principles - tests are written first, then the workflow is implemented.
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def ci_workflow_path():
    """Path to CI workflow file."""
    return Path(".github/workflows/ci.yml")


@pytest.fixture
def ci_workflow_content(ci_workflow_path):
    """Load CI workflow content."""
    if not ci_workflow_path.exists():
        pytest.skip(f"CI workflow not yet created: {ci_workflow_path}")
    
    with open(ci_workflow_path, 'r') as f:
        return yaml.safe_load(f)


def test_ci_workflow_file_exists(ci_workflow_path):
    """Test: CI workflow file exists."""
    assert ci_workflow_path.exists(), \
        f"CI workflow file should exist at {ci_workflow_path}"


def test_workflow_has_name(ci_workflow_content):
    """Test: Workflow has a descriptive name."""
    assert 'name' in ci_workflow_content, \
        "Workflow should have a name"
    assert len(ci_workflow_content['name']) > 0, \
        "Workflow name should not be empty"


def test_workflow_triggers_on_push(ci_workflow_content):
    """Test: Workflow triggers on push to any branch."""
    assert 'on' in ci_workflow_content or True in ci_workflow_content, \
        "Workflow should have trigger configuration"
    
    triggers = ci_workflow_content.get('on', ci_workflow_content.get(True, {}))
    assert 'push' in triggers, \
        "Workflow should trigger on push"


def test_workflow_triggers_on_pull_request(ci_workflow_content):
    """Test: Workflow triggers on pull requests to main."""
    triggers = ci_workflow_content.get('on', ci_workflow_content.get(True, {}))
    assert 'pull_request' in triggers, \
        "Workflow should trigger on pull requests"
    
    pr_config = triggers['pull_request']
    if isinstance(pr_config, dict):
        assert 'branches' in pr_config, \
            "Pull request trigger should specify branches"
        assert 'main' in pr_config['branches'], \
            "Pull request trigger should include main branch"


def test_workflow_has_lint_job(ci_workflow_content):
    """Test: Workflow has lint job."""
    assert 'jobs' in ci_workflow_content, \
        "Workflow should have jobs"
    assert 'lint' in ci_workflow_content['jobs'], \
        "Workflow should have a lint job"


def test_workflow_has_test_job(ci_workflow_content):
    """Test: Workflow has test job."""
    assert 'test' in ci_workflow_content['jobs'], \
        "Workflow should have a test job"


def test_lint_job_uses_python_311(ci_workflow_content):
    """Test: Lint job uses Python 3.11."""
    lint_job = ci_workflow_content['jobs']['lint']
    
    # Find setup-python step
    steps = lint_job.get('steps', [])
    python_steps = [s for s in steps if s.get('uses', '').startswith('actions/setup-python')]
    
    assert len(python_steps) > 0, \
        "Lint job should use actions/setup-python"
    
    python_step = python_steps[0]
    python_version = python_step.get('with', {}).get('python-version', '')
    
    assert '3.11' in str(python_version), \
        f"Lint job should use Python 3.11, got {python_version}"


def test_test_job_uses_python_311(ci_workflow_content):
    """Test: Test job uses Python 3.11."""
    test_job = ci_workflow_content['jobs']['test']
    
    # Find setup-python step
    steps = test_job.get('steps', [])
    python_steps = [s for s in steps if s.get('uses', '').startswith('actions/setup-python')]
    
    assert len(python_steps) > 0, \
        "Test job should use actions/setup-python"
    
    python_step = python_steps[0]
    python_version = python_step.get('with', {}).get('python-version', '')
    
    assert '3.11' in str(python_version), \
        f"Test job should use Python 3.11, got {python_version}"


def test_lint_job_runs_ruff(ci_workflow_content):
    """Test: Lint job runs ruff."""
    lint_job = ci_workflow_content['jobs']['lint']
    steps = lint_job.get('steps', [])
    
    # Check for ruff in run commands
    run_commands = [s.get('run', '') for s in steps if 'run' in s]
    ruff_commands = [cmd for cmd in run_commands if 'ruff' in cmd.lower()]
    
    assert len(ruff_commands) > 0, \
        "Lint job should run ruff"


def test_test_job_runs_pytest(ci_workflow_content):
    """Test: Test job runs pytest."""
    test_job = ci_workflow_content['jobs']['test']
    steps = test_job.get('steps', [])
    
    # Check for pytest in run commands
    run_commands = [s.get('run', '') for s in steps if 'run' in s]
    pytest_commands = [cmd for cmd in run_commands if 'pytest' in cmd.lower()]
    
    assert len(pytest_commands) > 0, \
        "Test job should run pytest"


def test_workflow_caches_dependencies(ci_workflow_content):
    """Test: Workflow caches pip dependencies."""
    jobs = ci_workflow_content['jobs']
    
    # Check if any job uses caching
    has_cache = False
    for _job_name, job_config in jobs.items():
        steps = job_config.get('steps', [])
        cache_steps = [s for s in steps if s.get('uses', '').startswith('actions/cache')]
        if cache_steps:
            has_cache = True
            break
    
    assert has_cache, \
        "Workflow should cache pip dependencies for faster builds"


def test_test_job_has_coverage_reporting(ci_workflow_content):
    """Test: Test job includes coverage reporting."""
    test_job = ci_workflow_content['jobs']['test']
    steps = test_job.get('steps', [])
    
    # Check for coverage in run commands
    run_commands = [s.get('run', '') for s in steps if 'run' in s]
    coverage_commands = [cmd for cmd in run_commands if '--cov' in cmd or 'coverage' in cmd.lower()]
    
    assert len(coverage_commands) > 0, \
        "Test job should include coverage reporting"


def test_workflow_sets_pythonpath(ci_workflow_content):
    """Test: Workflow sets PYTHONPATH for tests."""
    test_job = ci_workflow_content['jobs']['test']
    
    # Check for PYTHONPATH in env or run commands
    env = test_job.get('env', {})
    steps = test_job.get('steps', [])
    
    has_pythonpath = 'PYTHONPATH' in env
    
    if not has_pythonpath:
        # Check in run commands
        run_commands = [s.get('run', '') for s in steps if 'run' in s]
        has_pythonpath = any('PYTHONPATH' in cmd for cmd in run_commands)
    
    assert has_pythonpath, \
        "Test job should set PYTHONPATH=. for imports to work"


def test_workflow_installs_dependencies(ci_workflow_content):
    """Test: Workflow installs dependencies from requirements.txt."""
    jobs = ci_workflow_content['jobs']
    
    # Check if any job installs requirements
    has_install = False
    for _job_name, job_config in jobs.items():
        steps = job_config.get('steps', [])
        run_commands = [s.get('run', '') for s in steps if 'run' in s]
        install_commands = [
            cmd for cmd in run_commands
            if 'pip install' in cmd and 'requirements.txt' in cmd
        ]
        if install_commands:
            has_install = True
            break
    
    assert has_install, \
        "Workflow should install dependencies from requirements.txt"


def test_workflow_downloads_nltk_data(ci_workflow_content):
    """Test: Workflow downloads NLTK data."""
    test_job = ci_workflow_content['jobs']['test']
    steps = test_job.get('steps', [])
    
    # Check for NLTK download in run commands
    run_commands = [s.get('run', '') for s in steps if 'run' in s]
    nltk_commands = [
        cmd for cmd in run_commands
        if 'nltk' in cmd.lower() and 'download' in cmd.lower()
    ]
    
    assert len(nltk_commands) > 0, \
        "Test job should download NLTK vader_lexicon data"


def test_lint_job_runs_on_ubuntu(ci_workflow_content):
    """Test: Lint job runs on Ubuntu."""
    lint_job = ci_workflow_content['jobs']['lint']
    runs_on = lint_job.get('runs-on', '')
    
    assert 'ubuntu' in runs_on.lower(), \
        f"Lint job should run on Ubuntu, got {runs_on}"


def test_test_job_runs_on_ubuntu(ci_workflow_content):
    """Test: Test job runs on Ubuntu."""
    test_job = ci_workflow_content['jobs']['test']
    runs_on = test_job.get('runs-on', '')
    
    assert 'ubuntu' in runs_on.lower(), \
        f"Test job should run on Ubuntu, got {runs_on}"
