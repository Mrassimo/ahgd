"""
AHGD ETL Pipeline Core Module

Defines the pipeline structure and step dependencies.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class StepStatus(Enum):
    """Pipeline step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Represents a single step in the ETL pipeline."""
    name: str
    description: str
    dependencies: List[str]
    required: bool = True
    
    def __hash__(self):
        return hash(self.name)


class Pipeline:
    """Defines the ETL pipeline structure and dependencies."""
    
    # Define all pipeline steps
    STEPS = {
        "download": PipelineStep(
            name="download",
            description="Download Census and geographic data",
            dependencies=[]
        ),
        "geo": PipelineStep(
            name="geo",
            description="Process geographic boundaries",
            dependencies=["download"]
        ),
        "time": PipelineStep(
            name="time",
            description="Generate time dimension",
            dependencies=[]
        ),
        "dimensions": PipelineStep(
            name="dimensions",
            description="Generate dimension tables",
            dependencies=[]
        ),
        "g01": PipelineStep(
            name="g01",
            description="Process population data (G01)",
            dependencies=["geo", "time"]
        ),
        "g17": PipelineStep(
            name="g17",
            description="Process income data (G17)",
            dependencies=["geo", "time", "dimensions"]
        ),
        "g18": PipelineStep(
            name="g18",
            description="Process assistance needed data (G18)",
            dependencies=["geo", "time", "dimensions"]
        ),
        "g19": PipelineStep(
            name="g19",
            description="Process health conditions data (G19)",
            dependencies=["geo", "time", "dimensions"]
        ),
        "g20": PipelineStep(
            name="g20",
            description="Process selected conditions data (G20)",
            dependencies=["geo", "time", "dimensions"]
        ),
        "g21": PipelineStep(
            name="g21",
            description="Process conditions by characteristics (G21)",
            dependencies=["geo", "time", "dimensions"]
        ),
        "g25": PipelineStep(
            name="g25",
            description="Process unpaid assistance data (G25)",
            dependencies=["geo", "time", "dimensions"]
        ),
        "validate": PipelineStep(
            name="validate",
            description="Validate data quality",
            dependencies=["g01", "g17", "g18", "g19", "g20", "g21", "g25"],
            required=False
        ),
        "fix": PipelineStep(
            name="fix",
            description="Apply data quality fixes",
            dependencies=["validate"],
            required=False
        ),
        "export": PipelineStep(
            name="export",
            description="Export data to target format",
            dependencies=["validate"],
            required=False
        )
    }
    
    # Step groups for convenience
    GROUPS = {
        "census": ["g01", "g17", "g18", "g19", "g20", "g21", "g25"],
        "base": ["download", "geo", "time", "dimensions"],
        "all": ["download", "geo", "time", "dimensions", 
                "g01", "g17", "g18", "g19", "g20", "g21", "g25",
                "validate"]
    }
    
    @classmethod
    def resolve_steps(cls, requested_steps: List[str]) -> List[str]:
        """
        Resolve requested steps including dependencies.
        
        Args:
            requested_steps: List of requested step names or group names
            
        Returns:
            List of steps in execution order
        """
        # Expand groups
        expanded_steps = set()
        for step in requested_steps:
            if step in cls.GROUPS:
                expanded_steps.update(cls.GROUPS[step])
            else:
                expanded_steps.add(step)
        
        # Add dependencies
        all_steps = set()
        for step in expanded_steps:
            cls._add_with_dependencies(step, all_steps)
        
        # Sort topologically
        return cls._topological_sort(all_steps)
    
    @classmethod
    def _add_with_dependencies(cls, step_name: str, result: Set[str]) -> None:
        """Recursively add step and its dependencies."""
        if step_name not in cls.STEPS:
            return
            
        result.add(step_name)
        step = cls.STEPS[step_name]
        
        for dep in step.dependencies:
            if dep not in result:
                cls._add_with_dependencies(dep, result)
    
    @classmethod
    def _topological_sort(cls, steps: Set[str]) -> List[str]:
        """Sort steps in dependency order using topological sort."""
        # Build adjacency list
        graph = {step: [] for step in steps}
        in_degree = {step: 0 for step in steps}
        
        for step_name in steps:
            if step_name in cls.STEPS:
                step = cls.STEPS[step_name]
                for dep in step.dependencies:
                    if dep in steps:
                        graph[dep].append(step_name)
                        in_degree[step_name] += 1
        
        # Find steps with no dependencies
        queue = [step for step in steps if in_degree[step] == 0]
        result = []
        
        while queue:
            # Sort queue to ensure deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)
            
            # Remove edges from current
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(steps):
            raise ValueError("Circular dependency detected in pipeline")
        
        return result
    
    @classmethod
    def validate_steps(cls, steps: List[str]) -> List[str]:
        """
        Validate that all requested steps exist.
        
        Args:
            steps: List of step names to validate
            
        Returns:
            List of invalid step names
            
        Raises:
            ValueError: If invalid steps are found
        """
        invalid = []
        for step in steps:
            if step not in cls.STEPS and step not in cls.GROUPS:
                invalid.append(step)
        
        if invalid:
            raise ValueError(f"Invalid steps: {invalid}")
        
        return invalid