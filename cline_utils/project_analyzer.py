#!/usr/bin/env python3
"""
Project Type Analyzer for CRCT Dev Container

This module analyzes project directories to detect languages, frameworks, and tools
in use, then determines which Dev Container Features should be added to support them.
"""

import os
import glob
from pathlib import Path


# Feature detection mapping - file indicator to Dev Container Feature ID
FEATURE_MAP = {
    # --- Languages & Runtimes ---
    # "pubspec.yaml": "ghcr.io/devcontainers/features/flutter:1",     # Flutter/Dart
    "package.json": "ghcr.io/devcontainers/features/node:1",      # Node.js/JavaScript/TypeScript
    "pom.xml": "ghcr.io/devcontainers/features/java:1",         # Java (Maven)
    "build.gradle": "ghcr.io/devcontainers/features/java:1",    # Java (Gradle)
    "build.gradle.kts": "ghcr.io/devcontainers/features/java:1", # Java (Gradle Kotlin DSL)
    "Gemfile": "ghcr.io/devcontainers/features/ruby:1",         # Ruby
    "requirements.txt": "ghcr.io/devcontainers/features/python:1", # Python
    "pyproject.toml": "ghcr.io/devcontainers/features/python:1", # Python
    "go.mod": "ghcr.io/devcontainers/features/go:1",            # Go
    "Cargo.toml": "ghcr.io/devcontainers/features/rust:1",        # Rust
    "composer.json": "ghcr.io/devcontainers/features/php:1",      # PHP
    "*.csproj": "ghcr.io/devcontainers/features/dotnet:1",      # .NET C#
    "*.fsproj": "ghcr.io/devcontainers/features/dotnet:1",      # .NET F#
    "*.vbproj": "ghcr.io/devcontainers/features/dotnet:1",      # .NET VB
    "mix.exs": "ghcr.io/devcontainers-contrib/features/elixir:1", # Elixir (Community)
    "build.sbt": "ghcr.io/devcontainers-contrib/features/scala:1", # Scala (Community)

    # --- Other Tools ---
    "Dockerfile": "ghcr.io/devcontainers/features/docker-in-docker:1",
    "terraform.tfvars": "ghcr.io/devcontainers/features/terraform:1",
    # Add more...
}

# Extensions to search for via glob
GLOB_PATTERNS = {
    "*.csproj": "ghcr.io/devcontainers/features/dotnet:1",          # .NET C#
    "*.fsproj": "ghcr.io/devcontainers/features/dotnet:1",          # .NET F#
    "*.vbproj": "ghcr.io/devcontainers/features/dotnet:1",          # .NET VB
}


def detect_project_features(project_root_dir):
    """
    Scans a project directory to identify languages, frameworks, and tools in use.
    
    Args:`
        project_root_dir (str): Path to the project's root directory
        
    Returns:
        dict: Dictionary mapping feature IDs to empty config objects 
              (e.g., {"ghcr.io/devcontainers/features/node:1": {}})
    """
    detected_features = {}
    
    # Ensure we're working with an absolute path
    project_root = os.path.abspath(project_root_dir)
    
    # Validate directory exists
    if not os.path.isdir(project_root):
        print(f"Warning: Project directory not found: {project_root}")
        return detected_features
    
    # 1. Check for exact file matches at root level
    for file_indicator, feature_id in FEATURE_MAP.items():
        file_path = os.path.join(project_root, file_indicator)
        if os.path.exists(file_path):
            # Skip Python feature if already detected to avoid duplicates
            if feature_id == "ghcr.io/devcontainers/features/python:1" and feature_id in detected_features:
                continue
            detected_features[feature_id] = {}
    
    # 2. Check for glob patterns that need recursive search
    for pattern, feature_id in GLOB_PATTERNS.items():
        # Skip if already detected from another indicator
        if feature_id in detected_features:
            continue
            
        # Use Path().rglob for recursive glob pattern matching
        matches = list(Path(project_root).rglob(pattern))
        if matches:
            detected_features[feature_id] = {}
    
    # 3. Special case: Docker Compose - Look for database services 
    # (could enhance later to actually parse docker-compose.yml)
    compose_file = os.path.join(project_root, "docker-compose.yml")
    if os.path.exists(compose_file):
        # For now, we're just detecting presence of docker-compose.yml
        # A more sophisticated version would parse it to find specific services
        # detected_features["ghcr.io/devcontainers/features/docker-compose:1"] = {}
        pass  # Commenting this out as Docker Compose is often already available

    return detected_features


if __name__ == "__main__":
    # Simple test if run directly
    import sys
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
        features = detect_project_features(test_dir)
        if features:
            print(f"Detected features for {test_dir}:")
            for feature, config in features.items():
                print(f"  - {feature}")
        else:
            print(f"No features detected for {test_dir}")
    else:
        print("Usage: python project_analyzer.py <project_directory>") 