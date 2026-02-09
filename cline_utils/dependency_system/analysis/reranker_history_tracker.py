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
MAX_CYCLES_TO_KEEP = 10
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

        # Generate performance history report
        try:
            report = generate_performance_history_report(project_root, save_report=True)
            if "error" not in report:
                logger.info(
                    f"Performance history report generated for cycles {report.get('cycle_range', {})}"
                )
        except Exception as e:
            logger.warning(f"Failed to generate performance history report: {e}")

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
    project_root: str, max_age_cycles: int = 10
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


def generate_performance_history_report(
    project_root: str, save_report: bool = True
) -> Dict:
    """
    Generate a comprehensive performance history report analyzing reranker
    performance across all available cycles.

    Args:
        project_root: Project root directory
        save_report: Whether to save the report to the history directory

    Returns:
        Dictionary containing the complete performance report
    """
    history_dir = normalize_path(os.path.join(project_root, HISTORY_DIR))

    if not os.path.exists(history_dir):
        return {"error": "No history data available"}

    # Load all cycle data
    cycle_data = {}
    for filename in os.listdir(history_dir):
        if filename.startswith("cycle_") and filename.endswith(".json"):
            match = re.match(r"cycle_(\d+)\.json", filename)
            if match:
                cycle_num = int(match.group(1))
                filepath = os.path.join(history_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        cycle_data[cycle_num] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cycle data from {filepath}: {e}")

    if not cycle_data:
        return {"error": "No cycle data loaded"}

    sorted_cycles = sorted(cycle_data.keys())
    report = {
        "report_timestamp": datetime.now().isoformat(),
        "cycles_analyzed": len(sorted_cycles),
        "cycle_range": {"start": sorted_cycles[0], "end": sorted_cycles[-1]},
        "summary": {},
        "trends": {},
        "cycle_comparison": {},
        "consistency_analysis": {},
        "recommendations": [],
    }

    # Calculate summary statistics across all cycles
    all_confidences = []
    all_suggestion_counts = []
    all_std_devs = []
    all_medians = []
    relationship_type_aggregates = defaultdict(list)
    relationship_category_aggregates = defaultdict(list)

    for cycle_num in sorted_cycles:
        data = cycle_data[cycle_num]
        metrics = data.get("metrics", {})

        all_confidences.append(metrics.get("avg_confidence", 0))
        all_suggestion_counts.append(data.get("total_suggestions", 0))
        all_std_devs.append(metrics.get("std_dev_confidence", 0))
        all_medians.append(metrics.get("median_confidence", 0))

        # Aggregate relationship types
        for rel_type, count in metrics.get("relationship_types", {}).items():
            relationship_type_aggregates[rel_type].append(count)

        # Aggregate relationship categories
        for rel_cat, count in metrics.get("relationship_categories", {}).items():
            relationship_category_aggregates[rel_cat].append(count)

    # Summary statistics
    report["summary"] = {
        "avg_confidence_across_cycles": (
            round(statistics.mean(all_confidences), 4) if all_confidences else 0
        ),
        "median_confidence_across_cycles": (
            round(statistics.median(all_confidences), 4) if all_confidences else 0
        ),
        "avg_suggestions_per_cycle": (
            round(statistics.mean(all_suggestion_counts), 1)
            if all_suggestion_counts
            else 0
        ),
        "total_suggestions_all_cycles": sum(all_suggestion_counts),
        "confidence_range": {
            "min": round(min(all_confidences), 4) if all_confidences else 0,
            "max": round(max(all_confidences), 4) if all_confidences else 0,
        },
        "avg_std_dev": round(statistics.mean(all_std_devs), 4) if all_std_devs else 0,
        "avg_median_confidence": (
            round(statistics.mean(all_medians), 4) if all_medians else 0
        ),
    }

    # Trend analysis
    if len(sorted_cycles) >= 2:
        # Calculate period-over-period changes
        confidence_changes = []
        suggestion_changes = []

        for i in range(1, len(sorted_cycles)):
            prev_cycle = sorted_cycles[i - 1]
            curr_cycle = sorted_cycles[i]

            prev_conf = cycle_data[prev_cycle]["metrics"].get("avg_confidence", 0)
            curr_conf = cycle_data[curr_cycle]["metrics"].get("avg_confidence", 0)
            if prev_conf > 0:
                confidence_changes.append((curr_conf - prev_conf) / prev_conf * 100)

            prev_sugg = cycle_data[prev_cycle].get("total_suggestions", 0)
            curr_sugg = cycle_data[curr_cycle].get("total_suggestions", 0)
            if prev_sugg > 0:
                suggestion_changes.append((curr_sugg - prev_sugg) / prev_sugg * 100)

        report["trends"] = {
            "confidence_trend_direction": (
                "improving" if all_confidences[-1] > all_confidences[0] else "declining"
            ),
            "confidence_trend_pct": (
                round(
                    (all_confidences[-1] - all_confidences[0])
                    / all_confidences[0]
                    * 100,
                    2,
                )
                if all_confidences[0] > 0
                else 0
            ),
            "suggestions_trend_direction": (
                "increasing"
                if all_suggestion_counts[-1] > all_suggestion_counts[0]
                else "decreasing"
            ),
            "suggestions_trend_pct": (
                round(
                    (all_suggestion_counts[-1] - all_suggestion_counts[0])
                    / all_suggestion_counts[0]
                    * 100,
                    2,
                )
                if all_suggestion_counts[0] > 0
                else 0
            ),
            "avg_confidence_change_per_cycle_pct": (
                round(statistics.mean(confidence_changes), 2)
                if confidence_changes
                else 0
            ),
            "avg_suggestion_change_per_cycle_pct": (
                round(statistics.mean(suggestion_changes), 2)
                if suggestion_changes
                else 0
            ),
            "confidence_trend_series": [round(c, 4) for c in all_confidences],
            "suggestions_trend_series": all_suggestion_counts,
        }

    # Most recent vs previous cycle comparison
    if len(sorted_cycles) >= 2:
        latest_cycle = sorted_cycles[-1]
        prev_cycle = sorted_cycles[-2]

        latest_data = cycle_data[latest_cycle]
        prev_data = cycle_data[prev_cycle]

        latest_conf = latest_data["metrics"].get("avg_confidence", 0)
        prev_conf = prev_data["metrics"].get("avg_confidence", 0)
        conf_diff = latest_conf - prev_conf

        latest_sugg = latest_data.get("total_suggestions", 0)
        prev_sugg = prev_data.get("total_suggestions", 0)
        sugg_diff = latest_sugg - prev_sugg

        report["cycle_comparison"] = {
            "latest_cycle": latest_cycle,
            "previous_cycle": prev_cycle,
            "confidence_comparison": {
                "latest": round(latest_conf, 4),
                "previous": round(prev_conf, 4),
                "difference": round(conf_diff, 4),
                "pct_change": (
                    round(conf_diff / prev_conf * 100, 2) if prev_conf > 0 else 0
                ),
                "assessment": (
                    "higher" if conf_diff > 0 else "lower" if conf_diff < 0 else "same"
                ),
            },
            "suggestions_comparison": {
                "latest": latest_sugg,
                "previous": prev_sugg,
                "difference": sugg_diff,
                "pct_change": (
                    round(sugg_diff / prev_sugg * 100, 2) if prev_sugg > 0 else 0
                ),
                "assessment": (
                    "more" if sugg_diff > 0 else "fewer" if sugg_diff < 0 else "same"
                ),
            },
            "std_dev_comparison": {
                "latest": round(latest_data["metrics"].get("std_dev_confidence", 0), 4),
                "previous": round(prev_data["metrics"].get("std_dev_confidence", 0), 4),
            },
            "median_comparison": {
                "latest": round(latest_data["metrics"].get("median_confidence", 0), 4),
                "previous": round(prev_data["metrics"].get("median_confidence", 0), 4),
            },
        }

    # Consistency analysis
    if len(all_confidences) >= 3:
        conf_std = statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0
        sugg_std = (
            statistics.stdev(all_suggestion_counts)
            if len(all_suggestion_counts) > 1
            else 0
        )

        report["consistency_analysis"] = {
            "confidence_stability": (
                "stable"
                if conf_std < 0.05
                else "moderate" if conf_std < 0.1 else "volatile"
            ),
            "confidence_std_across_cycles": round(conf_std, 4),
            "suggestions_stability": (
                "stable"
                if sugg_std < 50
                else "moderate" if sugg_std < 100 else "volatile"
            ),
            "suggestions_std_across_cycles": round(sugg_std, 2),
            "coefficient_of_variation_confidence": (
                round(conf_std / statistics.mean(all_confidences) * 100, 2)
                if statistics.mean(all_confidences) > 0
                else 0
            ),
        }

    # Relationship type averages across cycles
    rel_type_avgs = {}
    for rel_type, counts in relationship_type_aggregates.items():
        rel_type_avgs[rel_type] = {
            "avg_per_cycle": round(statistics.mean(counts), 1),
            "total": sum(counts),
            "trend": (
                "increasing"
                if counts[-1] > counts[0]
                else "decreasing" if counts[-1] < counts[0] else "stable"
            ),
        }

    rel_cat_avgs = {}
    for rel_cat, counts in relationship_category_aggregates.items():
        rel_cat_avgs[rel_cat] = {
            "avg_per_cycle": round(statistics.mean(counts), 1),
            "total": sum(counts),
            "trend": (
                "increasing"
                if counts[-1] > counts[0]
                else "decreasing" if counts[-1] < counts[0] else "stable"
            ),
        }

    report["relationship_analysis"] = {
        "type_averages": rel_type_avgs,
        "category_averages": rel_cat_avgs,
    }

    # Generate recommendations based on analysis
    recommendations = []

    # Confidence-based recommendations
    avg_conf = report["summary"]["avg_confidence_across_cycles"]
    if avg_conf < 0.6:
        recommendations.append(
            {
                "priority": "high",
                "category": "confidence",
                "message": "Average confidence is below 0.6. Consider adjusting reranker thresholds or improving embedding quality.",
            }
        )
    elif avg_conf < 0.7:
        recommendations.append(
            {
                "priority": "medium",
                "category": "confidence",
                "message": "Average confidence is moderate (0.6-0.7). Monitor for declining trends.",
            }
        )

    # Trend-based recommendations
    if len(sorted_cycles) >= 2:
        if report["trends"]["confidence_trend_direction"] == "declining":
            recommendations.append(
                {
                    "priority": "high",
                    "category": "trend",
                    "message": f"Confidence trend is declining ({report['trends']['confidence_trend_pct']}%). Investigate recent changes to reranker configuration.",
                }
            )

        if (
            report["trends"]["suggestions_trend_direction"] == "increasing"
            and report["trends"]["suggestions_trend_pct"] > 50
        ):
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "volume",
                    "message": f"Suggestions increased by {report['trends']['suggestions_trend_pct']}%. Ensure quality isn't being sacrificed for quantity.",
                }
            )

    # Consistency-based recommendations
    if "confidence_stability" in report.get("consistency_analysis", {}):
        if report["consistency_analysis"]["confidence_stability"] == "volatile":
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "consistency",
                    "message": "Confidence is volatile across cycles. Consider stabilizing reranker parameters.",
                }
            )

    # Latest cycle comparison recommendations
    if "cycle_comparison" in report:
        if report["cycle_comparison"]["confidence_comparison"]["assessment"] == "lower":
            recommendations.append(
                {
                    "priority": "high",
                    "category": "recent_performance",
                    "message": f"Latest cycle ({report['cycle_comparison']['latest_cycle']}) shows lower confidence than previous. Review recent changes.",
                }
            )

    if not recommendations:
        recommendations.append(
            {
                "priority": "info",
                "category": "status",
                "message": "Reranker performance is within acceptable parameters. Continue monitoring.",
            }
        )

    report["recommendations"] = recommendations

    # Save report to history directory
    if save_report:
        report_path = os.path.join(history_dir, "performance_history_report.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Performance history report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")

    return report


def format_report_summary(report: Dict) -> str:
    """
    Format a performance history report as a human-readable string.

    Args:
        report: The report dictionary from generate_performance_history_report

    Returns:
        Formatted string summary of the report
    """
    if "error" in report:
        return f"Error: {report['error']}"

    lines = []
    lines.append("=" * 60)
    lines.append("RERANKER PERFORMANCE HISTORY REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {report['report_timestamp']}")
    lines.append(
        f"Cycles Analyzed: {report['cycles_analyzed']} (Cycle {report['cycle_range']['start']} - {report['cycle_range']['end']})"
    )
    lines.append("")

    # Summary section
    lines.append("-" * 40)
    lines.append("SUMMARY")
    lines.append("-" * 40)
    summary = report["summary"]
    lines.append(f"Average Confidence: {summary['avg_confidence_across_cycles']}")
    lines.append(f"Median Confidence: {summary['median_confidence_across_cycles']}")
    lines.append(
        f"Confidence Range: {summary['confidence_range']['min']} - {summary['confidence_range']['max']}"
    )
    lines.append(f"Avg Suggestions/Cycle: {summary['avg_suggestions_per_cycle']}")
    lines.append(f"Total Suggestions: {summary['total_suggestions_all_cycles']}")
    lines.append(f"Avg Std Deviation: {summary['avg_std_dev']}")
    lines.append("")

    # Trends section
    if report.get("trends"):
        lines.append("-" * 40)
        lines.append("TRENDS")
        lines.append("-" * 40)
        trends = report["trends"]
        lines.append(
            f"Confidence Trend: {trends['confidence_trend_direction']} ({trends['confidence_trend_pct']}%)"
        )
        lines.append(
            f"Suggestions Trend: {trends['suggestions_trend_direction']} ({trends['suggestions_trend_pct']}%)"
        )
        lines.append(
            f"Avg Confidence Change/Cycle: {trends['avg_confidence_change_per_cycle_pct']}%"
        )
        lines.append("")

    # Cycle comparison section
    if report.get("cycle_comparison"):
        lines.append("-" * 40)
        lines.append("LATEST VS PREVIOUS CYCLE")
        lines.append("-" * 40)
        comp = report["cycle_comparison"]
        lines.append(
            f"Comparing Cycle {comp['latest_cycle']} vs {comp['previous_cycle']}:"
        )
        conf_comp = comp["confidence_comparison"]
        lines.append(
            f"  Confidence: {conf_comp['previous']} -> {conf_comp['latest']} ({conf_comp['assessment']}, {conf_comp['pct_change']}%)"
        )
        sugg_comp = comp["suggestions_comparison"]
        lines.append(
            f"  Suggestions: {sugg_comp['previous']} -> {sugg_comp['latest']} ({sugg_comp['assessment']}, {sugg_comp['pct_change']}%)"
        )
        lines.append("")

    # Consistency section
    if report.get("consistency_analysis"):
        lines.append("-" * 40)
        lines.append("CONSISTENCY ANALYSIS")
        lines.append("-" * 40)
        consistency = report["consistency_analysis"]
        lines.append(f"Confidence Stability: {consistency['confidence_stability']}")
        lines.append(
            f"Coefficient of Variation: {consistency['coefficient_of_variation_confidence']}%"
        )
        lines.append("")

    # Recommendations section
    if report.get("recommendations"):
        lines.append("-" * 40)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for rec in report["recommendations"]:
            lines.append(
                f"[{rec['priority'].upper()}] {rec['category']}: {rec['message']}"
            )
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
