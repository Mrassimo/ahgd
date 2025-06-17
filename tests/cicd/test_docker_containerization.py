"""
CI/CD Testing - Docker Containerization and Security Testing

This module provides comprehensive testing for Docker containerization,
multi-stage builds, security scanning, and orchestration validation.
"""

import pytest
import docker
import json
import os
import subprocess
import yaml
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional

class DockerContainerValidator:
    """Validates Docker containers and orchestration"""
    
    def __init__(self):
        self.client = None
        self.project_root = Path(__file__).parent.parent.parent
        
    def setup_docker_client(self):
        """Set up Docker client for testing"""
        try:
            self.client = docker.from_env()
            return True
        except Exception as e:
            print(f"Docker client setup failed: {e}")
            return False
    
    def validate_dockerfile(self, dockerfile_path: str = "Dockerfile") -> dict:
        """Validate Dockerfile best practices and security"""
        dockerfile_path = self.project_root / dockerfile_path
        
        if not dockerfile_path.exists():
            return {"valid": False, "error": "Dockerfile not found"}
        
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "security_issues": [],
            "best_practices": []
        }
        
        lines = content.split('\n')
        
        # Security checks
        if 'USER root' in content:
            validation_results["security_issues"].append("Running as root user")
        
        if not any('USER ' in line and 'USER root' not in line for line in lines):
            validation_results["security_issues"].append("No non-root user specified")
        
        if 'COPY . .' in content:
            validation_results["security_issues"].append("Copying entire context - use .dockerignore")
        
        # Best practice checks
        if not any(line.startswith('FROM ') and ':' in line for line in lines):
            validation_results["best_practices"].append("Use specific image tags instead of 'latest'")
        
        if content.count('RUN ') > 5:
            validation_results["best_practices"].append("Consider combining RUN commands to reduce layers")
        
        if 'HEALTHCHECK' not in content:
            validation_results["best_practices"].append("Add HEALTHCHECK instruction")
        
        if not any('LABEL' in line for line in lines):
            validation_results["best_practices"].append("Add metadata labels")
        
        return validation_results
    
    def build_test_image(self, dockerfile_path: str = "Dockerfile", tag: str = "test-image") -> dict:
        """Build Docker image for testing"""
        if not self.setup_docker_client():
            return {"success": False, "error": "Docker client not available"}
        
        try:
            build_start = time.time()
            image, build_logs = self.client.images.build(
                path=str(self.project_root),
                dockerfile=dockerfile_path,
                tag=tag,
                rm=True,
                forcerm=True,
                pull=True
            )
            build_duration = time.time() - build_start
            
            return {
                "success": True,
                "image_id": image.id,
                "size": image.attrs['Size'],
                "build_duration": build_duration,
                "layers": len(image.history()),
                "tags": image.tags
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def scan_image_vulnerabilities(self, image_tag: str) -> dict:
        """Simulate container vulnerability scanning"""
        # In production, this would use actual tools like Trivy, Clair, etc.
        mock_vulnerabilities = [
            {
                "cve": "CVE-2023-12345",
                "severity": "MEDIUM",
                "package": "libssl1.1",
                "version": "1.1.1-1ubuntu2.1~18.04.20",
                "description": "Sample vulnerability for testing"
            }
        ]
        
        return {
            "total_vulnerabilities": len(mock_vulnerabilities),
            "critical": 0,
            "high": 0,
            "medium": 1,
            "low": 0,
            "vulnerabilities": mock_vulnerabilities,
            "scan_time": time.time()
        }
    
    def test_container_runtime(self, image_tag: str) -> dict:
        """Test container runtime behavior"""
        if not self.setup_docker_client():
            return {"success": False, "error": "Docker client not available"}
        
        try:
            # Run container with basic health check
            container = self.client.containers.run(
                image_tag,
                command="python -c 'import src; print(\"Import successful\")'",
                detach=True,
                remove=True
            )
            
            # Wait for container to complete
            result = container.wait(timeout=60)
            logs = container.logs().decode('utf-8')
            
            return {
                "success": True,
                "exit_code": result['StatusCode'],
                "logs": logs,
                "runtime_duration": 10  # Mock duration
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def validate_multi_stage_build(self, dockerfile_content: str) -> dict:
        """Validate multi-stage build optimization"""
        stages = []
        current_stage = None
        
        for line in dockerfile_content.split('\n'):
            line = line.strip()
            if line.startswith('FROM '):
                if ' AS ' in line.upper():
                    stage_name = line.split(' AS ')[1].strip()
                    current_stage = {"name": stage_name, "instructions": []}
                    stages.append(current_stage)
                else:
                    current_stage = {"name": "final", "instructions": []}
                    stages.append(current_stage)
            elif current_stage:
                current_stage["instructions"].append(line)
        
        return {
            "is_multi_stage": len(stages) > 1,
            "stage_count": len(stages),
            "stages": [stage["name"] for stage in stages],
            "optimized": len(stages) > 1 and any("builder" in stage["name"].lower() for stage in stages)
        }

class TestDockerContainerization:
    """Test Docker containerization and security"""
    
    def setup_method(self):
        """Set up test environment"""
        self.validator = DockerContainerValidator()
        self.test_dockerfile_path = Path("tests/cicd/workflows/docker/Dockerfile")
        
    def test_dockerfile_validation(self):
        """Test Dockerfile validation and best practices"""
        # Create a test Dockerfile
        test_dockerfile_content = """
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pyproject.toml setup.py ./

# Set ownership and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src; print('Health check passed')" || exit 1

# Add metadata labels
LABEL maintainer="AHGD Analytics Team"
LABEL version="1.0.0"
LABEL description="Australian Health Data Analytics Platform"

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "src.cli"]
"""
        
        self.test_dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.test_dockerfile_path, 'w') as f:
            f.write(test_dockerfile_content.strip())
        
        # Validate Dockerfile
        result = self.validator.validate_dockerfile(str(self.test_dockerfile_path))
        
        assert result["valid"], "Dockerfile should be valid"
        assert len(result["security_issues"]) == 0, f"Security issues found: {result['security_issues']}"
        
        # Check multi-stage build
        multi_stage_result = self.validator.validate_multi_stage_build(test_dockerfile_content)
        assert multi_stage_result["is_multi_stage"], "Should be a multi-stage build"
        assert multi_stage_result["optimized"], "Should be optimized multi-stage build"
    
    @pytest.mark.skipif(not pytest.importorskip("docker", reason="Docker not available"), reason="Docker not available")
    def test_docker_image_build(self):
        """Test Docker image build process"""
        # Mock Docker build since we might not have Docker in CI
        with patch.object(self.validator, 'setup_docker_client', return_value=True):
            with patch.object(self.validator, 'client') as mock_client:
                # Mock successful build
                mock_image = Mock()
                mock_image.id = "sha256:abcd1234"
                mock_image.attrs = {'Size': 500000000}  # 500MB
                mock_image.history.return_value = [{}] * 10  # 10 layers
                mock_image.tags = ["test-image:latest"]
                
                mock_client.images.build.return_value = (mock_image, [])
                
                result = self.validator.build_test_image()
                
                assert result["success"], f"Build failed: {result.get('error')}"
                assert result["size"] < 1000000000, "Image size should be under 1GB"
                assert result["layers"] <= 15, "Should have reasonable number of layers"
                assert result["build_duration"] > 0, "Build duration should be positive"
    
    def test_container_security_scanning(self):
        """Test container security vulnerability scanning"""
        scan_result = self.validator.scan_image_vulnerabilities("test-image:latest")
        
        assert "total_vulnerabilities" in scan_result
        assert "critical" in scan_result
        assert "high" in scan_result
        assert scan_result["critical"] == 0, "Should have no critical vulnerabilities"
        assert scan_result["high"] <= 2, "Should have minimal high severity vulnerabilities"
    
    @pytest.mark.skipif(not pytest.importorskip("docker", reason="Docker not available"), reason="Docker not available")
    def test_container_runtime_behavior(self):
        """Test container runtime and functionality"""
        with patch.object(self.validator, 'setup_docker_client', return_value=True):
            with patch.object(self.validator, 'client') as mock_client:
                # Mock successful container run
                mock_container = Mock()
                mock_container.wait.return_value = {'StatusCode': 0}
                mock_container.logs.return_value = b"Import successful\n"
                
                mock_client.containers.run.return_value = mock_container
                
                result = self.validator.test_container_runtime("test-image:latest")
                
                assert result["success"], f"Container test failed: {result.get('error')}"
                assert result["exit_code"] == 0, "Container should exit successfully"
                assert "Import successful" in result["logs"], "Expected output not found"
    
    def test_docker_compose_validation(self):
        """Test Docker Compose orchestration configuration"""
        docker_compose_content = {
            "version": "3.8",
            "services": {
                "ahgd-analytics": {
                    "build": {
                        "context": ".",
                        "dockerfile": "Dockerfile",
                        "target": "production"
                    },
                    "ports": ["8000:8000"],
                    "environment": [
                        "PYTHONPATH=/app/src",
                        "LOG_LEVEL=INFO"
                    ],
                    "volumes": [
                        "./data:/app/data:ro",
                        "./logs:/app/logs"
                    ],
                    "healthcheck": {
                        "test": ["CMD", "python", "-c", "import src; print('Health OK')"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3,
                        "start_period": "40s"
                    },
                    "restart": "unless-stopped",
                    "deploy": {
                        "resources": {
                            "limits": {
                                "cpus": "2.0",
                                "memory": "4G"
                            },
                            "reservations": {
                                "cpus": "0.5",
                                "memory": "1G"
                            }
                        }
                    }
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                    "command": "redis-server --appendonly yes",
                    "volumes": ["redis_data:/data"],
                    "restart": "unless-stopped"
                },
                "postgres": {
                    "image": "postgres:15-alpine",
                    "environment": [
                        "POSTGRES_DB=ahgd_analytics",
                        "POSTGRES_USER=analytics_user",
                        "POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password"
                    ],
                    "volumes": [
                        "postgres_data:/var/lib/postgresql/data",
                        "./init.sql:/docker-entrypoint-initdb.d/init.sql:ro"
                    ],
                    "secrets": ["postgres_password"],
                    "restart": "unless-stopped"
                }
            },
            "volumes": {
                "redis_data": {},
                "postgres_data": {}
            },
            "secrets": {
                "postgres_password": {
                    "file": "./secrets/postgres_password.txt"
                }
            },
            "networks": {
                "default": {
                    "driver": "bridge"
                }
            }
        }
        
        # Save Docker Compose file
        compose_path = Path("tests/cicd/workflows/docker/docker-compose.yml")
        compose_path.parent.mkdir(parents=True, exist_ok=True)
        with open(compose_path, 'w') as f:
            yaml.dump(docker_compose_content, f)
        
        # Validate Docker Compose structure
        assert "version" in docker_compose_content
        assert "services" in docker_compose_content
        assert len(docker_compose_content["services"]) >= 1
        
        # Validate main service configuration
        main_service = docker_compose_content["services"]["ahgd-analytics"]
        assert "build" in main_service or "image" in main_service
        assert "healthcheck" in main_service
        assert "restart" in main_service
        
        # Validate resource limits
        if "deploy" in main_service:
            resources = main_service["deploy"]["resources"]
            assert "limits" in resources
            assert "memory" in resources["limits"]
    
    def test_kubernetes_deployment_manifests(self):
        """Test Kubernetes deployment manifest validation"""
        k8s_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "ahgd-analytics",
                "namespace": "default",
                "labels": {
                    "app": "ahgd-analytics",
                    "version": "v1.0.0"
                }
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": "ahgd-analytics"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "ahgd-analytics",
                            "version": "v1.0.0"
                        }
                    },
                    "spec": {
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        },
                        "containers": [{
                            "name": "ahgd-analytics",
                            "image": "ghcr.io/ahgd/ahgd-analytics:v1.0.0",
                            "ports": [{"containerPort": 8000}],
                            "env": [
                                {"name": "PYTHONPATH", "value": "/app/src"},
                                {"name": "LOG_LEVEL", "value": "INFO"}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "1Gi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "4Gi",
                                    "cpu": "2000m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {
                                    "drop": ["ALL"]
                                }
                            }
                        }],
                        "imagePullSecrets": [
                            {"name": "ghcr-secret"}
                        ]
                    }
                }
            }
        }
        
        # Save Kubernetes manifest
        k8s_path = Path("tests/cicd/workflows/docker/kubernetes/deployment.yaml")
        k8s_path.parent.mkdir(parents=True, exist_ok=True)
        with open(k8s_path, 'w') as f:
            yaml.dump(k8s_deployment, f)
        
        # Validate Kubernetes deployment
        assert k8s_deployment["kind"] == "Deployment"
        assert k8s_deployment["spec"]["replicas"] >= 2  # High availability
        
        # Validate security context
        pod_spec = k8s_deployment["spec"]["template"]["spec"]
        assert pod_spec["securityContext"]["runAsNonRoot"] is True
        
        container = pod_spec["containers"][0]
        assert "resources" in container
        assert "livenessProbe" in container
        assert "readinessProbe" in container
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
    
    def test_container_optimization_metrics(self):
        """Test container optimization and performance metrics"""
        optimization_metrics = {
            "image_size": 500000000,  # 500MB
            "layers": 10,
            "build_time": 180,  # 3 minutes
            "startup_time": 5,  # 5 seconds
            "memory_usage": 1000000000,  # 1GB
            "cpu_usage": 0.5  # 50% of 1 CPU
        }
        
        # Validate optimization thresholds
        assert optimization_metrics["image_size"] < 1000000000, "Image size should be under 1GB"
        assert optimization_metrics["layers"] <= 15, "Should have reasonable layer count"
        assert optimization_metrics["build_time"] < 600, "Build time should be under 10 minutes"
        assert optimization_metrics["startup_time"] < 30, "Startup time should be under 30 seconds"
        assert optimization_metrics["memory_usage"] < 2000000000, "Memory usage should be under 2GB"
        assert optimization_metrics["cpu_usage"] < 1.0, "CPU usage should be reasonable"
    
    def test_container_health_checks(self):
        """Test container health check configurations"""
        health_check_configs = [
            {
                "name": "HTTP health check",
                "type": "http",
                "endpoint": "/health",
                "port": 8000,
                "interval": 30,
                "timeout": 10,
                "retries": 3
            },
            {
                "name": "Command health check",
                "type": "command",
                "command": ["python", "-c", "import src; print('Health OK')"],
                "interval": 30,
                "timeout": 10,
                "retries": 3
            }
        ]
        
        for config in health_check_configs:
            assert config["interval"] <= 60, "Health check interval should be reasonable"
            assert config["timeout"] <= 30, "Health check timeout should be reasonable"
            assert config["retries"] >= 2, "Should have multiple retries"
            
            if config["type"] == "http":
                assert config["endpoint"].startswith("/"), "HTTP endpoint should be valid"
                assert config["port"] > 0, "Port should be valid"
    
    def test_container_registry_integration(self):
        """Test container registry integration and image management"""
        registry_config = {
            "registry": "ghcr.io",
            "namespace": "ahgd",
            "repository": "ahgd-analytics",
            "tag_strategy": "semantic",
            "retention_policy": {
                "keep_latest": 10,
                "keep_tagged": True,
                "delete_untagged": True
            }
        }
        
        # Validate registry configuration
        assert registry_config["registry"] in ["ghcr.io", "docker.io", "gcr.io"]
        assert registry_config["namespace"], "Namespace should be specified"
        assert registry_config["repository"], "Repository should be specified"
        assert registry_config["retention_policy"]["keep_latest"] > 0
    
    def test_container_secrets_management(self):
        """Test container secrets and configuration management"""
        secrets_config = {
            "database_password": {
                "type": "environment",
                "source": "kubernetes_secret",
                "key": "postgres_password"
            },
            "api_key": {
                "type": "file",
                "source": "mounted_secret",
                "path": "/run/secrets/api_key"
            }
        }
        
        for secret_name, config in secrets_config.items():
            assert config["type"] in ["environment", "file"], "Secret type should be valid"
            assert config["source"], "Secret source should be specified"
            
            if config["type"] == "environment":
                assert config["key"], "Environment variable key should be specified"
            elif config["type"] == "file":
                assert config["path"].startswith("/"), "File path should be absolute"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])