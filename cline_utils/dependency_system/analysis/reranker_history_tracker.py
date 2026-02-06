# cline_utils/dependency_system/analysis/reranker_history_tracker.py

"""
Reranker Performance Tracking System

Parses suggestions.log after each analysis run to extract reranker assignments,
confidence scores, and performance metrics. Stores data by cycle number with
automatic rotation to keep only the last N cycles.
"""

import json
import logging
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from cline_utils.dependency_system.utils.path_utils import get_file_type, normalize_path

logger = logging.getLogger(__name__)

# Configuration
HISTORY_DIR = "cline_utils/dependency_system/analysis/reranker_history"
MAX_CYCLES_TO_KEEP = 5
SUGGESTIONS_LOG_FILENAME = "suggestions.log"
SCANS_LOG_FILENAME = "cline_utils/dependency_system/analysis/reranker_scans.jsonl"

# Regex pattern for parsing suggestions.log
# Format: h:/path/file1.ext -> h:/path/file2.ext ('TYPE') conf: 0.XXX (rel: ext->ext)
SUGGESTION_PATTERN = re.compile(
    r"([a-zA-Z]:/[^\s]+)\s+->\s+([a-zA-Z]:/[^\s]+)\s+\('([^']+)'\)\s+conf:\s+([\d.]+)\s+\(rel:\s+([^)]+)\)",
    re.IGNORECASE,
)
# Format: h:/path/file1.ext -> h:/path/file2.ext promoted to '<' (score: 0.XXX)
# Format: h:/path/file1.ext -> h:/path/file2.ext promoted to '<' (conf: 0.XXX, rel: md->py)
PROMOTED_PATTERN = re.compile(
    r"([a-zA-Z]:/[^\s]+)\s+->\s+([a-zA-Z]:/[^\s]+)\s+promoted to\s+'([^']+)'\s+\((?:score|conf):\s*([\d.]+)(?:,\s*rel:\s*([^)]+))?\)",
    re.IGNORECASE,
)


class RerankerAssignment:
    """Represents a single reranker assignment."""

    def __init__(
        self,
        source: str,
        target: str,
        rel_type: str,
        confidence: float,
        relationship: str,
    ):
        super().__init__()
        self.source = source
        self.target = target
        self.rel_type = rel_type  # S, s, etc.
        self.confidence = confidence
        self.relationship = relationship  # md->py, py->md, etc.

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.rel_type,
            "confidence": self.confidence,
            "relationship": self.relationship,
        }


def parse_suggestions_log(log_path: str) -> List[RerankerAssignment]:
    """
    Parse suggestions.log to extract reranker assignments.

    Args:
        log_path: Absolute path to suggestions.log

    Returns:
        List of RerankerAssignment objects
    """
    assignments = []

    if not os.path.exists(log_path):
        logger.warning(f"Suggestions log not found: {log_path}")
        return assignments

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = SUGGESTION_PATTERN.search(line)
                if match:
                    source, target, rel_type, conf_str, relationship = match.groups()
                    try:
                        confidence = float(conf_str)
                        assignment = RerankerAssignment(
                            source=source,
                            target=target,
                            rel_type=rel_type,
                            confidence=confidence,
                            relationship=relationship,
                        )
                        assignments.append(assignment)
                    except ValueError:
                        logger.debug(f"Invalid confidence value: {conf_str}")
                        continue
                else:
                    promoted_match = PROMOTED_PATTERN.search(line)
                    if promoted_match:
                        (
                            source,
                            target,
                            rel_type,
                            conf_str,
                            relationship,
                        ) = promoted_match.groups()
                        try:
                            confidence = float(conf_str)
                        except ValueError:
                            logger.debug(f"Invalid confidence value: {conf_str}")
                            continue

                        if not relationship:
                            relationship = (
                                f"{get_file_type(source)}->{get_file_type(target)}"
                            )

                        assignment = RerankerAssignment(
                            source=source,
                            target=target,
                            rel_type=rel_type,
                            confidence=confidence,
                            relationship=relationship,
                        )
                        assignments.append(assignment)

        logger.debug(f"Parsed {len(assignments)} reranker assignments from {log_path}")
        return assignments

    except Exception as e:
        logger.error(f"Error parsing suggestions log {log_path}: {e}", exc_info=True)
        return assignments


def parse_scans_log(log_path: str) -> List[Dict[str, str]]:
    """
    Parse reranker_scans.jsonl to extract all scanned pairs.
    """
    scans = []
    if not os.path.exists(log_path):
        return scans

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        scans.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return scans
    except Exception as e:
        logger.error(f"Error parsing scans log {log_path}: {e}")
        return scans


def aggregate_metrics(assignments: List[RerankerAssignment]) -> Dict:
    """
    Calculate aggregate metrics from reranker assignments.

    Args:
        assignments: List of RerankerAssignment objects

    Returns:
        Dictionary of aggregated metrics
    """
    if not assignments:
        return {
            "avg_confidence": 0.0,
            "median_confidence": 0.0,
            "std_dev_confidence": 0.0,
            "confidence_distribution": {},
            "relationship_types": {},
            "relationship_categories": {},
        }

    confidences = [a.confidence for a in assignments]

    # Confidence statistics
    avg_confidence = statistics.mean(confidences)
    median_confidence = statistics.median(confidences)
    std_dev = statistics.stdev(confidences) if len(confidences) > 1 else 0.0

    # Confidence distribution buckets
    distribution = {
        "0.9+": sum(1 for c in confidences if c >= 0.9),
        "0.8-0.9": sum(1 for c in confidences if 0.8 <= c < 0.9),
        "0.7-0.8": sum(1 for c in confidences if 0.7 <= c < 0.8),
        "<0.7": sum(1 for c in confidences if c < 0.7),
    }

    # Relationship type counts (S, s, etc.)
    rel_types = defaultdict(int)
    for a in assignments:
        rel_types[a.rel_type] += 1

    # Relationship category counts (md->py, py->md, etc.)
    rel_categories = defaultdict(int)
    for a in assignments:
        rel_categories[a.relationship] += 1

    return {
        "avg_confidence": round(avg_confidence, 4),
        "median_confidence": round(median_confidence, 4),
        "std_dev_confidence": round(std_dev, 4),
        "confidence_distribution": distribution,
        "relationship_types": dict(rel_types),
        "relationship_categories": dict(rel_categories),
    }


def get_top_assignments(
    assignments: List[RerankerAssignment], n: int = 10
) -> List[Dict]:
    """Get top N most confident assignments."""
    sorted_assignments = sorted(assignments, key=lambda a: a.confidence, reverse=True)
    return [a.to_dict() for a in sorted_assignments[:n]]


def get_bottom_assignments(
    assignments: List[RerankerAssignment], n: int = 10
) -> List[Dict]:
    """Get bottom N least confident assignments."""
    sorted_assignments = sorted(assignments, key=lambda a: a.confidence)
    return [a.to_dict() for a in sorted_assignments[:n]]


def save_cycle_data(
    cycle_number: int,
    assignments: List[RerankerAssignment],
    all_pairs: List[Dict[str, str]],
    project_root: str,
) -> bool:
    """
    Save reranker performance data for a cycle.

    Args:
        cycle_number: The cycle number
        assignments: List of RerankerAssignment objects
        all_pairs: List of all scanned pairs
        project_root: Project root directory

    Returns:
        True if successful, False otherwise
    """
    history_dir = normalize_path(os.path.join(project_root, HISTORY_DIR))
    os.makedirs(history_dir, exist_ok=True)

    # Calculate metrics
    metrics = aggregate_metrics(assignments)

    # Prepare data structure
    data = {
        "cycle": cycle_number,
        "timestamp": datetime.now().isoformat(),
        "total_suggestions": len(assignments),
        "metrics": metrics,
        "metrics": metrics,
        "top_10_confident": get_top_assignments(assignments, 10),
        "bottom_10_confident": get_bottom_assignments(assignments, 10),
        "all_pairs": all_pairs,
    }

    # Save to file
    cycle_file = os.path.join(history_dir, f"cycle_{cycle_number}.json")
    try:
        with open(cycle_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Saved reranker performance data for cycle {cycle_number} to {cycle_file}"
        )
        return True
    except Exception as e:
        logger.error(f"Error saving cycle data to {cycle_file}: {e}", exc_info=True)
        return False


def rotate_old_cycles(project_root: str, max_cycles: int = MAX_CYCLES_TO_KEEP) -> None:
    """
    Remove old cycle files, keeping only the most recent N cycles.

    Args:
        project_root: Project root directory
        max_cycles: Maximum number of cycles to keep
    """
    history_dir = normalize_path(os.path.join(project_root, HISTORY_DIR))

    if not os.path.exists(history_dir):
        return

    # Find all cycle files
    cycle_files = []
    for filename in os.listdir(history_dir):
        if filename.startswith("cycle_") and filename.endswith(".json"):
            match = re.match(r"cycle_(\d+)\.json", filename)
            if match:
                cycle_num = int(match.group(1))
                filepath = os.path.join(history_dir, filename)
                cycle_files.append((cycle_num, filepath))

    # Sort by cycle number (descending)
    cycle_files.sort(reverse=True)

    # Remove old files
    if len(cycle_files) > max_cycles:
        files_to_remove = cycle_files[max_cycles:]
        for cycle_num, filepath in files_to_remove:
            try:
                os.remove(filepath)
                logger.debug(f"Removed old cycle file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove old cycle file {filepath}: {e}")


def track_reranker_performance(cycle_number: int, project_root: str) -> bool:
    """
    Main entry point for tracking reranker performance.

    Args:
        cycle_number: The current cycle number
        project_root: Project root directory

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Tracking reranker performance for cycle {cycle_number}...")

    # Locate suggestions.log
    log_path = normalize_path(os.path.join(project_root, SUGGESTIONS_LOG_FILENAME))

    # Locate scans log
    scans_path = normalize_path(os.path.join(project_root, SCANS_LOG_FILENAME))

    # Parse assignments
    assignments = parse_suggestions_log(log_path)

    # Parse scans
    scanned_pairs = parse_scans_log(scans_path)

    if not assignments and not scanned_pairs:
        logger.warning(f"No reranker activity found. Skipping performance tracking.")
        return False

    # If we have assignments but no scans (legacy/fallback), use assignments as scans
    if not scanned_pairs and assignments:
        scanned_pairs = [
            {"source": a.source, "target": a.target, "confidence": a.confidence}
            for a in assignments
        ]

    # Save cycle data
    success = save_cycle_data(cycle_number, assignments, scanned_pairs, project_root)

    if success:
        # Rotate old cycles
        rotate_old_cycles(project_root)

        # Clean up scans log
        if os.path.exists(scans_path):
            try:
                os.remove(scans_path)
            except Exception as e:
                logger.warning(f"Failed to remove scans log {scans_path}: {e}")

        logger.info(
            f"Reranker performance tracking completed for cycle {cycle_number}. Found {len(assignments)} assignments and {len(scanned_pairs)} scanned pairs."
        )

    return success


def get_performance_comparison(
    project_root: str, cycles: Optional[List[int]] = None
) -> Dict:
    """
    Get performance comparison across multiple cycles.

    Args:
        project_root: Project root directory
        cycles: Specific cycle numbers to compare (or None for all available)

    Returns:
        Dictionary with comparison data
    """
    history_dir = normalize_path(os.path.join(project_root, HISTORY_DIR))

    if not os.path.exists(history_dir):
        return {"error": "No history data available"}

    # Load cycle data
    cycle_data = {}
    for filename in os.listdir(history_dir):
        if filename.startswith("cycle_") and filename.endswith(".json"):
            match = re.match(r"cycle_(\d+)\.json", filename)
            if match:
                cycle_num = int(match.group(1))
                if cycles is None or cycle_num in cycles:
                    filepath = os.path.join(history_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            cycle_data[cycle_num] = json.load(f)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load cycle data from {filepath}: {e}"
                        )

    if not cycle_data:
        return {"error": "No cycle data loaded"}

    # Build comparison
    sorted_cycles = sorted(cycle_data.keys())
    comparison = {
        "cycles": sorted_cycles,
        "confidence_trend": [
            cycle_data[c]["metrics"]["avg_confidence"] for c in sorted_cycles
        ],
        "total_suggestions_trend": [
            cycle_data[c]["total_suggestions"] for c in sorted_cycles
        ],
        "cycle_details": {c: cycle_data[c]["metrics"] for c in sorted_cycles},
    }

    return comparison


def get_historical_pairs(
    project_root: str, max_age_cycles: int = 5
) -> set[Tuple[str, str]]:
    """
    Retrieve all source-target pairs from history files.

    Args:
        project_root: Project root directory
        max_age_cycles: Only include pairs from the last N cycles (default 5).
                        This prevents pairs from being excluded indefinitely.

    Returns:
        Set of (source, target) tuples
    """
    history_dir = normalize_path(os.path.join(project_root, HISTORY_DIR))
    historical_pairs = set()

    if not os.path.exists(history_dir):
        return historical_pairs

    # Find all cycle files and determine the current max cycle
    cycle_files = []
    for filename in os.listdir(history_dir):
        if filename.startswith("cycle_") and filename.endswith(".json"):
            match = re.match(r"cycle_(\d+)\.json", filename)
            if match:
                cycle_num = int(match.group(1))
                cycle_files.append((cycle_num, os.path.join(history_dir, filename)))

    if not cycle_files:
        return historical_pairs

    # Sort by cycle number descending and keep only recent cycles
    cycle_files.sort(reverse=True)
    max_cycle = cycle_files[0][0]
    min_cycle = max_cycle - max_age_cycles + 1  # e.g., if max=145 and age=5, min=141

    for cycle_num, filepath in cycle_files:
        # Skip cycles older than max_age_cycles
        if cycle_num < min_cycle:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Try to get from 'all_pairs' first (new format)
                if "all_pairs" in data:
                    for pair in data["all_pairs"]:
                        historical_pairs.add((pair["source"], pair["target"]))
                else:
                    # Fallback to top/bottom lists (legacy format)
                    for item in data.get("top_10_confident", []):
                        historical_pairs.add((item["source"], item["target"]))
                    for item in data.get("bottom_10_confident", []):
                        historical_pairs.add((item["source"], item["target"]))

        except Exception as e:
            logger.warning(f"Failed to load history from {filepath}: {e}")

    logger.debug(
        f"Loaded {len(historical_pairs)} historical pairs from cycles {min_cycle}-{max_cycle}"
    )
    return historical_pairs
