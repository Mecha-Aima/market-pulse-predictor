"""
Tests for CD workflow configuration.

These tests verify that the GitHub Actions CD workflow is properly configured
following TDD principles - tests are written first, then the workflow is implemented.
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def cd_workflow_path():
    """Path to CD workflow file."""
    return Path(".github/workflows/cd.yml")


@pytest.fixture
def cd_workflow_content(cd_workflow_path):
    """Load CD workflow content."""
    if not cd_workflow_path.exists():
        pytest.skip(f"CD workflow not yet created: {cd_workflow_path}")

    with open(cd_workflow_path, "r") as f:
        return yaml.safe_load(f)


def test_cd_workflow_file_exists(cd_workflow_path):
    """Test: CD workflow file exists."""
    assert cd_workflow_path.exists(), f"CD workflow file should exist at {cd_workflow_path}"


def test_workflow_only_triggers_on_main(cd_workflow_content):
    """Test: Workflow only triggers on push to main branch."""
    assert "on" in cd_workflow_content or True in cd_workflow_content, (
        "Workflow should have trigger configuration"
    )

    triggers = cd_workflow_content.get("on", cd_workflow_content.get(True, {}))
    assert "push" in triggers, "Workflow should trigger on push"

    push_config = triggers["push"]
    if isinstance(push_config, dict):
        assert "branches" in push_config, "Push trigger should specify branches"
        branches = push_config["branches"]
        assert "main" in branches, "Push trigger should include main branch"
        assert len(branches) == 1, (
            "Push trigger should ONLY include main branch (no other branches)"
        )


def test_workflow_has_build_and_push_job(cd_workflow_content):
    """Test: Workflow has build-and-push job."""
    assert "jobs" in cd_workflow_content, "Workflow should have jobs"
    assert "build-and-push" in cd_workflow_content["jobs"], (
        "Workflow should have a build-and-push job"
    )


def test_workflow_has_deploy_job(cd_workflow_content):
    """Test: Workflow has deploy job."""
    assert "deploy" in cd_workflow_content["jobs"], "Workflow should have a deploy job"


def test_deploy_job_depends_on_build(cd_workflow_content):
    """Test: Deploy job depends on build-and-push job."""
    deploy_job = cd_workflow_content["jobs"]["deploy"]

    assert "needs" in deploy_job, "Deploy job should have dependencies"

    needs = deploy_job["needs"]
    if isinstance(needs, str):
        needs = [needs]

    assert "build-and-push" in needs, "Deploy job should depend on build-and-push job"


def test_workflow_uses_aws_credentials_secret(cd_workflow_content):
    """Test: Workflow uses AWS_ACCESS_KEY_ID secret."""
    jobs = cd_workflow_content["jobs"]

    # Check if any job references AWS credentials
    has_aws_key = False
    for _job_name, job_config in jobs.items():
        steps = job_config.get("steps", [])
        env = job_config.get("env", {})

        # Check job-level env
        if "AWS_ACCESS_KEY_ID" in str(env):
            has_aws_key = True
            break

        # Check step-level env
        for step in steps:
            step_env = step.get("env", {})
            if "AWS_ACCESS_KEY_ID" in str(step_env):
                has_aws_key = True
                break

        if has_aws_key:
            break

    assert has_aws_key, "Workflow should use AWS_ACCESS_KEY_ID secret"


def test_workflow_uses_aws_secret_access_key_secret(cd_workflow_content):
    """Test: Workflow uses AWS_SECRET_ACCESS_KEY secret."""
    jobs = cd_workflow_content["jobs"]

    # Check if any job references AWS secret key
    has_aws_secret = False
    for _job_name, job_config in jobs.items():
        steps = job_config.get("steps", [])
        env = job_config.get("env", {})

        # Check job-level env
        if "AWS_SECRET_ACCESS_KEY" in str(env):
            has_aws_secret = True
            break

        # Check step-level env
        for step in steps:
            step_env = step.get("env", {})
            if "AWS_SECRET_ACCESS_KEY" in str(step_env):
                has_aws_secret = True
                break

        if has_aws_secret:
            break

    assert has_aws_secret, "Workflow should use AWS_SECRET_ACCESS_KEY secret"


def test_workflow_uses_ecr_repository_secret(cd_workflow_content):
    """Test: Workflow uses ECR_REPOSITORY secret or configuration."""
    jobs = cd_workflow_content["jobs"]

    # Check if any job references ECR repository
    has_ecr = False
    for _job_name, job_config in jobs.items():
        steps = job_config.get("steps", [])
        env = job_config.get("env", {})

        # Check job-level env
        if "ECR" in str(env).upper():
            has_ecr = True
            break

        # Check step-level env and run commands
        for step in steps:
            step_env = step.get("env", {})
            step_run = step.get("run", "")
            if "ECR" in str(step_env).upper() or "ECR" in step_run.upper():
                has_ecr = True
                break

        if has_ecr:
            break

    assert has_ecr, "Workflow should reference ECR repository for Docker image push"


def test_workflow_uses_ec2_ssh_key_secret(cd_workflow_content):
    """Test: Workflow uses EC2_SSH_KEY secret for deployment."""
    deploy_job = cd_workflow_content["jobs"]["deploy"]
    steps = deploy_job.get("steps", [])

    # Check if any step references SSH key
    has_ssh_key = False
    for step in steps:
        step_env = step.get("env", {})
        step_run = step.get("run", "")
        step_with = step.get("with", {})

        ssh_found = (
            "SSH" in str(step_env).upper()
            or "SSH" in step_run.upper()
            or "SSH" in str(step_with).upper()
        )
        if ssh_found:
            has_ssh_key = True
            break

    assert has_ssh_key, "Deploy job should use EC2_SSH_KEY secret for SSH access"


def test_build_job_builds_api_docker_image(cd_workflow_content):
    """Test: Build job builds API Docker image."""
    build_job = cd_workflow_content["jobs"]["build-and-push"]
    steps = build_job.get("steps", [])

    # Check for Docker build commands
    run_commands = [s.get("run", "") for s in steps if "run" in s]
    docker_build_commands = [
        cmd
        for cmd in run_commands
        if "docker build" in cmd.lower() or "docker buildx" in cmd.lower()
    ]

    # Check if API Dockerfile is referenced
    has_api_build = any(
        "Dockerfile.api" in cmd or "api" in cmd.lower() for cmd in docker_build_commands
    )

    assert has_api_build, "Build job should build API Docker image from docker/Dockerfile.api"


def test_build_job_builds_dashboard_docker_image(cd_workflow_content):
    """Test: Build job builds Dashboard Docker image."""
    build_job = cd_workflow_content["jobs"]["build-and-push"]
    steps = build_job.get("steps", [])

    # Check for Docker build commands
    run_commands = [s.get("run", "") for s in steps if "run" in s]
    docker_build_commands = [
        cmd
        for cmd in run_commands
        if "docker build" in cmd.lower() or "docker buildx" in cmd.lower()
    ]

    # Check if Dashboard Dockerfile is referenced
    has_dashboard_build = any(
        "Dockerfile.dashboard" in cmd or "dashboard" in cmd.lower() for cmd in docker_build_commands
    )

    assert has_dashboard_build, (
        "Build job should build Dashboard Docker image from docker/Dockerfile.dashboard"
    )


def test_build_job_pushes_to_ecr(cd_workflow_content):
    """Test: Build job pushes images to ECR."""
    build_job = cd_workflow_content["jobs"]["build-and-push"]
    steps = build_job.get("steps", [])

    # Check for Docker push commands or ECR login
    run_commands = [s.get("run", "") for s in steps if "run" in s]
    uses_commands = [s.get("uses", "") for s in steps if "uses" in s]

    has_ecr_login = any("ecr" in cmd.lower() for cmd in uses_commands) or any(
        "ecr" in cmd.lower() and "login" in cmd.lower() for cmd in run_commands
    )

    has_docker_push = any("docker push" in cmd.lower() for cmd in run_commands)

    assert has_ecr_login or has_docker_push, (
        "Build job should push images to ECR (requires ECR login and docker push)"
    )


def test_deploy_job_ssh_to_ec2(cd_workflow_content):
    """Test: Deploy job SSHs into EC2 instance."""
    deploy_job = cd_workflow_content["jobs"]["deploy"]
    steps = deploy_job.get("steps", [])

    # Check for SSH commands
    run_commands = [s.get("run", "") for s in steps if "run" in s]
    uses_commands = [s.get("uses", "") for s in steps if "uses" in s]

    has_ssh = any("ssh" in cmd.lower() for cmd in run_commands) or any(
        "ssh" in cmd.lower() for cmd in uses_commands
    )

    assert has_ssh, "Deploy job should SSH into EC2 instance"


def test_deploy_job_pulls_docker_images(cd_workflow_content):
    """Test: Deploy job pulls Docker images on EC2."""
    deploy_job = cd_workflow_content["jobs"]["deploy"]
    steps = deploy_job.get("steps", [])

    # Check for docker pull commands
    run_commands = [s.get("run", "") for s in steps if "run" in s]

    has_docker_pull = any("docker pull" in cmd.lower() for cmd in run_commands)

    assert has_docker_pull, "Deploy job should pull Docker images on EC2"


def test_deploy_job_runs_docker_compose(cd_workflow_content):
    """Test: Deploy job runs docker-compose to update services."""
    deploy_job = cd_workflow_content["jobs"]["deploy"]
    steps = deploy_job.get("steps", [])

    # Check for docker-compose commands
    run_commands = [s.get("run", "") for s in steps if "run" in s]

    has_compose = any(
        "docker-compose" in cmd.lower() or "docker compose" in cmd.lower() for cmd in run_commands
    )

    assert has_compose, "Deploy job should run docker-compose to update services"


def test_deploy_job_has_health_check(cd_workflow_content):
    """Test: Deploy job verifies health check after deployment."""
    deploy_job = cd_workflow_content["jobs"]["deploy"]
    steps = deploy_job.get("steps", [])

    # Check for health check commands
    run_commands = [s.get("run", "") for s in steps if "run" in s]

    has_health_check = any("health" in cmd.lower() or "curl" in cmd.lower() for cmd in run_commands)

    assert has_health_check, "Deploy job should verify health check after deployment"


def test_deploy_job_has_rollback_logic(cd_workflow_content):
    """Test: Deploy job has rollback logic on failure."""
    deploy_job = cd_workflow_content["jobs"]["deploy"]
    steps = deploy_job.get("steps", [])

    # Check for rollback-related commands or conditional steps
    has_rollback = False
    for step in steps:
        step_name = step.get("name", "").lower()
        step_run = step.get("run", "").lower()
        step_if = step.get("if", "").lower()

        if "rollback" in step_name or "rollback" in step_run or "failure" in step_if:
            has_rollback = True
            break

    assert has_rollback, "Deploy job should have rollback logic on failure"
