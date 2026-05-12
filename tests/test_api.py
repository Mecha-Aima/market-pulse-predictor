import shutil
import subprocess

from fastapi.testclient import TestClient

from tests.conftest import REQUIRED_ENV_KEYS


def test_health_endpoint_returns_200(api_app) -> None:
    client = TestClient(api_app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_env_example_has_all_required_keys(project_root) -> None:
    env_path = project_root / ".env.example"
    keys = {
        line.split("=", maxsplit=1)[0]
        for line in env_path.read_text().splitlines()
        if line and not line.startswith("#") and "=" in line
    }

    assert keys == REQUIRED_ENV_KEYS


def test_docker_compose_valid(project_root) -> None:
    docker_compose = shutil.which("docker-compose")
    command = (
        [docker_compose, "config", "--quiet"]
        if docker_compose
        else ["docker", "compose", "config", "--quiet"]
    )

    result = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
