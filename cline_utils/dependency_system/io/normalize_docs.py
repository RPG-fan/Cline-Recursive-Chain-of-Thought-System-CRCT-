import sys
import os
import json
import re
import argparse
from typing import Any, Match, cast

# Add project root to python path to import cline_utils modules
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from cline_utils.dependency_system.analysis.local_llm_processor import LocalLLMProcessor
from cline_utils.dependency_system.utils.path_utils import get_project_root
from cline_utils.dependency_system.utils.config_manager import ConfigManager

REQUIRED_MARKERS = [
    "---TAGS_START---",
    "---TAGS_END---",
    "STATION_HEADER_START",
    "STATION_HEADER_END",
    "---CONTEXT_START---",
    "---CONTEXT_END---",
    "---OVERVIEW_START---",
    "---OVERVIEW_END---",
    "---DETAILS_START---",
    "---DETAILS_END---",
    "---REFERENCES_START---",
    "---REFERENCES_END---",
]

# Local model context limit definitions (matching the resolve-placeholders workflow)
MAX_MODEL_CTX = 32768
OUTPUT_HEADROOM = 1024
MAX_INPUT_TOKENS = 30000  # 30k tokens max input


def load_transparency_registry(project_root: str) -> set[str]:
    from cline_utils.dependency_system.core import resolve_state_path
    
    core_dir = os.path.join(
        project_root,
        "cline_utils",
        "dependency_system",
        "core",
    )
    registry_path = resolve_state_path("transparency_registry.json", core_dir)
    if not os.path.isfile(registry_path):
        print(f"Warning: Transparency registry not found at {registry_path}")
        return set()
    with open(registry_path, "r", encoding="utf-8") as f:
        registry: dict[str, Any] = json.load(f)
    files_dict: dict[str, Any] = registry.get("files", {})
    return {os.path.normpath(k).lower() for k in files_dict.keys()}


def find_normalization_candidates(
    project_root: str,
    doc_dirs: list[str],
    registered_files: set[str],
    file_path_arg: str | None = None,
) -> list[str]:
    """
    Finds and returns list of candidate documentation files that need normalization
    or placeholder resolution.
    """
    candidates: list[str] = []
    if file_path_arg:
        full_path = os.path.abspath(file_path_arg)
        if os.path.isfile(full_path):
            candidates.append(full_path)
        else:
            raise FileNotFoundError(f"Specified file not found: {file_path_arg}")
    else:
        for doc_dir in doc_dirs:
            if not os.path.isdir(doc_dir):
                continue
            for root, _, files in os.walk(doc_dir):
                for file in sorted(files):
                    if file.endswith(".md"):
                        filepath = os.path.normpath(os.path.join(root, file))
                        filepath_lower = filepath.lower()

                        # Skip if registered
                        if filepath_lower in registered_files:
                            continue

                        # Check compliance and placeholders
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()

                        # If the tagging system is already present, skip to avoid re-processing converted files
                        if "---TAGS_START---" in content:
                            continue

                        # We always normalize files that have placeholders or are missing markers.
                        missing_markers = [
                            m for m in REQUIRED_MARKERS if m not in content
                        ]
                        has_placeholders = "[FILL:" in content
                        has_links_in_details = False

                        # Check if "Documentation Links" or other reference headers exist under details
                        if "## Details" in content:
                            details_section = content.split("## Details")[1].split(
                                "## References"
                            )[0]
                            if any(
                                hdr in details_section
                                for hdr in [
                                    "Documentation Links",
                                    "Links",
                                    "See Also",
                                    "Related Files",
                                ]
                            ):
                                has_links_in_details = True

                        if missing_markers or has_placeholders or has_links_in_details:
                            candidates.append(filepath)

    return candidates


def clean_body_content(content: str) -> str:
    """
    Remove title, tags section, station header, and all compliance markers
    along with redundant section headers to extract clean body content.
    """
    # Remove tags block if present
    content = re.sub(r"---TAGS_START---.*?---TAGS_END---", "", content, flags=re.DOTALL)

    # Remove station header if present
    content = re.sub(
        r"<!--\s*---\s*STATION_HEADER_START\s*---\s*\[AUTO\].*?---\s*STATION_HEADER_END\s*---\s*\[AUTO\]\s*-->",
        "",
        content,
        flags=re.DOTALL,
    )

    # Remove the first H1 header line
    lines = content.split("\n")
    cleaned_lines: list[str] = []
    removed_title = False
    for line in lines:
        if not removed_title and line.strip().startswith("# "):
            removed_title = True
            continue
        cleaned_lines.append(line)
    content = "\n".join(cleaned_lines).strip()

    # Strip all compliance markers
    content = re.sub(r"---[A-Z_]+_START---", "", content)
    content = re.sub(r"---[A-Z_]+_END---", "", content)

    # Strip standalone ## Context, ## Overview, ## Details headers (case-insensitive, optionally with spaces)
    content = re.sub(
        r"^\s*##\s*(Context|Overview|Details|References)\s*$",
        "",
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    return content.strip()


def extract_references(text: str) -> tuple[str, str]:
    """
    Separate the details content and the references section, stripping markers if present.
    Supports a wide array of reference headers to prevent links from leaking into Details.
    """
    ref_pattern = re.compile(
        r"^(#{2,3}\s*(?:References|Documentation Links|See Also|Related Files|Links|Related Documentation|External Resources)\b.*)",
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    match = ref_pattern.search(text)
    if match:
        ref_text = match.group(1).strip()
        # Clean markers from references section if any
        ref_text = re.sub(r"---REFERENCES_(START|END)---", "", ref_text)
        details_text = text[: match.start()].strip()
        return details_text, ref_text.strip()
    return text.strip(), ""


def parse_station_header(content: str) -> tuple[str, str, str, str]:
    """
    Extract existing STATION_HEADER values: ROLE, LAYER, CRCT_KEY, TRACKER_REF.
    """
    header_match = re.search(
        r"<!--\s*---\s*STATION_HEADER_START\s*---\s*\[AUTO\](.*?)---\s*STATION_HEADER_END\s*---\s*\[AUTO\]\s*-->",
        content,
        re.DOTALL,
    )
    role, layer, crct_key, tracker_ref = "", "", "", ""
    if header_match:
        inner = header_match.group(1)
        role_match = re.search(r"ROLE:\s*(.*?)(?:\n|$)", inner)
        if role_match:
            role = role_match.group(1).strip()
        layer_match = re.search(r"LAYER:\s*(.*?)(?:\n|$)", inner)
        if layer_match:
            layer = layer_match.group(1).strip()
        key_match = re.search(r"CRCT_KEY:\s*(.*?)(?:\[AUTO\]|\n|$)", inner)
        if key_match:
            crct_key = key_match.group(1).strip()
        ref_match = re.search(r"TRACKER_REF:\s*(.*?)(?:\[AUTO\]|\n|$)", inner)
        if ref_match:
            tracker_ref = ref_match.group(1).strip()

    return role, layer, crct_key, tracker_ref


def parse_json_safely(text: str) -> dict[str, Any] | None:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            result = json.loads(json_str)
            if isinstance(result, dict):
                return cast(dict[str, Any], result)
    except Exception as e:
        print(f"JSON parse error: {e}")
    return None


def prepare_normalization(
    filepath: str, processor: LocalLLMProcessor, project_root: str
) -> str | None:
    """
    Normalizes a single documentation file by querying the local LLM, adding tags,
    documentation layers, roles, context, overview, details, and dynamic reference/citation blocks.
    Returns the normalized text if successfully processed, None otherwise.
    """
    rel_path = os.path.relpath(filepath, project_root)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Extract basic components
        title_match = re.search(r"^#\s+(.+)$", original_content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = (
                os.path.splitext(os.path.basename(filepath))[0]
                .replace("_", " ")
                .title()
            )

        existing_role, existing_layer, crct_key, tracker_ref = parse_station_header(
            original_content
        )
        if not crct_key or not tracker_ref:
            filename = os.path.basename(filepath)
            if not crct_key:
                crct_key = f"[FILL: CRCT key for {filename}]"
            if not tracker_ref:
                tracker_ref = f"[FILL: path to tracker for {filename}]"
            print(
                f"Bootstrapping {rel_path} with placeholder STATION_HEADER fields."
            )


        # Extract existing context and overview to leverage in prompting
        existing_context = ""
        existing_overview = ""

        context_match = re.search(
            r"## Context\s*\n(.*?)(?:\n##|\n---)",
            original_content,
            re.DOTALL | re.IGNORECASE,
        )
        if context_match:
            existing_context = context_match.group(1).strip()
        overview_match = re.search(
            r"## Overview\s*\n(.*?)(?:\n##|\n---)",
            original_content,
            re.DOTALL | re.IGNORECASE,
        )
        if overview_match:
            existing_overview = overview_match.group(1).strip()

        # Setup the prompting instructions
        system_prompt = (
            "You are an expert technical writer and documentation architect. Your task is to analyze the provided markdown document and generate metadata to normalize it.\n"
            "Analyze the document content and return a JSON object with the following schema:\n"
            "{\n"
            '  "tags": ["tag1", "tag2"],\n'
            '  "related_tags": ["rtag1", "rtag2"],\n'
            '  "role": "A single concise sentence describing this specific file\'s responsibility in the project.",\n'
            '  "layer": "The classification layer of the file (e.g., Practices & Guidelines, System Reference, Design Specification, Sourcebook Chapter, API, Legacy Archive, Utility)",\n'
            '  "context": "A concise 1-2 sentence paragraph explaining what the document is about and what specific problem or topic it addresses.",\n'
            '  "overview": "A concise paragraph summarizing the high-level concepts, architecture, or structure described in the document."\n'
            "}\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Do NOT rewrite or return the bulk details of the document. Only output the metadata requested in the JSON schema.\n"
            "- If the document already contains a 'Context' or 'Overview' section, read them and refine/condense them into the JSON fields rather than inventing them from scratch.\n"
            "- Output ONLY valid JSON. Do NOT wrap it in markdown backticks or include any extra text."
        )

        user_prompt = f"Title: {title}\n"
        if existing_context:
            user_prompt += f"Existing Context: {existing_context}\n"
        if existing_overview:
            user_prompt += f"Existing Overview: {existing_overview}\n"
        user_prompt += f"Document content:\n```markdown\n{original_content}\n```"

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        prompt_tokens = processor.get_token_count(prompt)
        if prompt_tokens > MAX_INPUT_TOKENS:
            print(
                f"Skipping {rel_path}: Document total token count ({prompt_tokens}) exceeds maximum input context size ({MAX_INPUT_TOKENS} tokens). Must be processed via cloud LLM."
            )
            return None

        print(
            f"File size verified suitable ({prompt_tokens} tokens). Running analysis..."
        )

        result_text = processor.generate(
            prompt,
            max_tokens=OUTPUT_HEADROOM,
            stop=["<|im_end|>"],
            temperature=0.1,
            echo=False,
        )

        metadata = parse_json_safely(result_text)

        if not metadata:
            print(f"Error: Model did not return valid JSON for {rel_path}. Skipping.")
            return None

        tags = metadata.get("tags", ["documentation"])
        related_tags = metadata.get("related_tags", ["project"])
        role = metadata.get("role", "No role provided.")
        layer = metadata.get("layer", "Documentation")

        # Preserve existing non-placeholder role/layer
        if existing_role and not existing_role.strip().startswith("[FILL:"):
            role = existing_role.strip()
        if existing_layer and not existing_layer.strip().startswith("[FILL:"):
            layer = existing_layer.strip()

        context = metadata.get("context", "")
        overview = metadata.get("overview", "")

        # Rebuild non-compliant file completely to ensure perfect structure
        raw_body = clean_body_content(original_content)
        details_body, references_body = extract_references(raw_body)

        # If details body is empty or only whitespace (e.g. in index files), provide a clean placeholder sentence
        if not details_body.strip():
            details_body = f"This directory serves as a centralized reference point. Navigation links to specific documentation and implementation details are provided in the references section below."

        # Seed our reference tracking set and lists
        seen_ref_urls: dict[str, int] = {}
        url_list: list[tuple[int, str, str]] = []
        existing_ref_lines: list[str] = []

        if references_body:
            # Find any links already in the extracted references section
            ref_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
            for match in ref_link_pattern.finditer(references_body):
                link_text = match.group(1).strip()
                link_url = match.group(2).strip()
                if not link_url.startswith("#"):
                    norm_url = link_url.lower()
                    if norm_url not in seen_ref_urls:
                        index = len(url_list) + 1
                        seen_ref_urls[norm_url] = index
                        url_list.append((index, link_text, link_url))

            # Keep original references text clean of boundaries
            references_body_clean = re.sub(
                r"---REFERENCES_(START|END)---", "", references_body
            ).strip()
            existing_ref_lines.append(references_body_clean)

        # Replace inline links in details_body with reference citations
        def replace_link(match: Match[str]) -> str:
            link_text = match.group(1).strip()
            link_url = match.group(2).strip()

            # Skip internal section anchors
            if link_url.startswith("#"):
                return str(match.group(0))

            norm_url = link_url.lower()
            if norm_url not in seen_ref_urls:
                index = len(url_list) + 1
                seen_ref_urls[norm_url] = index
                url_list.append((index, link_text, link_url))
            else:
                index = seen_ref_urls[norm_url]

            return f"[{link_text}][{index}]"

        details_body_normalized = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)", replace_link, details_body
        )

        # Format the dynamic references section
        if url_list:
            if not references_body:
                existing_ref_lines.append("## References\n")
            else:
                existing_ref_lines.append("\n### Citation References\n")

            bullet_lines: list[str] = []
            definition_lines: list[str] = []
            for index, link_text, link_url in url_list:
                clean_text = link_text.replace("`", "").strip()
                bullet_lines.append(f"- [{index}] [{clean_text}]({link_url})")
                definition_lines.append(f"[{index}]: {link_url}")

            existing_ref_lines.append("\n".join(bullet_lines))
            existing_ref_lines.append("")
            existing_ref_lines.append("\n".join(definition_lines))

        # Set references block to empty if no verified references could be found/assembled
        if not references_body and not url_list:
            final_references = ""
        else:
            final_references = "\n\n".join(existing_ref_lines).strip()

        tags_block = f"""---TAGS_START---
tags: {json.dumps(tags)}
related_tags: {json.dumps(related_tags)}
---TAGS_END---"""

        station_header = f"""<!-- --- STATION_HEADER_START --- [AUTO]
ROLE:    {role}
LAYER:   {layer}
CRCT_KEY:   {crct_key} [AUTO]
TRACKER_REF: {tracker_ref} [AUTO]
--- STATION_HEADER_END --- [AUTO] -->"""

        normalized_text = f"""{tags_block}
{station_header}

# {title}

---CONTEXT_START---
## Context

{context}
---CONTEXT_END---

---OVERVIEW_START---
## Overview

{overview}
---OVERVIEW_END---

---DETAILS_START---
## Details

{details_body_normalized}
---DETAILS_END---
"""

        normalized_text += f"""
---REFERENCES_START---
{final_references}
---REFERENCES_END---
"""
        return normalized_text
    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return None


def normalize_single_file(
    filepath: str,
    processor: LocalLLMProcessor,
    project_root: str,
    dry_run: bool = False,
) -> bool:
    """
    Normalizes a single documentation file by querying the local LLM.
    Synchronous fallback wrapper.
    """
    rel_path = os.path.relpath(filepath, project_root)
    normalized_text = prepare_normalization(filepath, processor, project_root)
    if not normalized_text:
        return False

    try:
        if dry_run:
            dry_run_dir = os.path.join(project_root, "scratch", "dry_run")
            os.makedirs(dry_run_dir, exist_ok=True)
            dry_run_path = os.path.join(dry_run_dir, os.path.basename(filepath))
            with open(dry_run_path, "w", encoding="utf-8") as f:
                f.write(normalized_text)
            print(f"Dry-run saved to: {os.path.relpath(dry_run_path, project_root)}")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(normalized_text)
            print(f"Successfully normalized and saved: {rel_path}")
        return True
    except Exception as e:
        print(f"Error saving {rel_path}: {e}")
        return False


def background_write_and_virtualize(
    filepath: str, normalized_text: str, project_root: str, dry_run: bool
) -> None:
    """
    Writes the normalized text to disk and immediately runs virtualization
    to relocate markers to the transparency registry. Runs on a background thread.
    """
    from cline_utils.dependency_system.io.transparency_manager import (
        get_transparency_manager,
    )

    manager = get_transparency_manager()
    rel_path = os.path.relpath(filepath, project_root)
    try:
        if dry_run:
            dry_run_dir = os.path.join(project_root, "scratch", "dry_run")
            os.makedirs(dry_run_dir, exist_ok=True)
            dry_run_path = os.path.join(dry_run_dir, os.path.basename(filepath))
            with open(dry_run_path, "w", encoding="utf-8") as f:
                f.write(normalized_text)
            print(
                f"  [BACKGROUND WRITE] Dry-run saved to: {os.path.relpath(dry_run_path, project_root)}"
            )
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(normalized_text)
            print(f"  [BACKGROUND WRITE] Successfully normalized and saved: {rel_path}")

            # Step 2: Virtualize (remove physical markers and relocate to registry)
            print(
                f"  [BACKGROUND WRITE] Relocating compliance markers for {rel_path} to transparency registry..."
            )
            if manager.remove_markers(filepath):
                print(
                    f"  [BACKGROUND WRITE] Successfully virtualized metadata for {rel_path} to registry."
                )
            else:
                print(
                    f"  [BACKGROUND WRITE] Warning: Failed to virtualize metadata for {rel_path}."
                )
    except Exception as e:
        print(f"  [BACKGROUND WRITE ERROR] Failed to process {rel_path}: {e}")


def normalize_docs_batch(
    candidates: list[str],
    processor: LocalLLMProcessor,
    project_root: str,
    limit: int,
    dry_run: bool,
) -> int:
    """
    Processes candidate files in batch:
    1. Measures exact token size of prompt contexts.
    2. Sorts candidates descending (largest -> smallest context size).
    3. Runs local LLM on the main thread for sequential analysis.
    4. Offloads write and virtualization tasks to a single-threaded background executor.
    """
    import concurrent.futures

    if not candidates:
        print("No candidates provided for batch normalization.")
        return 0

    print(
        f"Measuring exact prompt token counts for {len(candidates)} candidate file(s)..."
    )
    valid_candidates: list[dict[str, Any]] = []

    # Setup prompt template parts for precise eager measurement
    system_prompt = (
        "You are an expert technical writer and documentation architect. Your task is to analyze the provided markdown document and generate metadata to normalize it.\n"
        "Analyze the document content and return a JSON object with the following schema:\n"
        "{\n"
        '  "tags": ["tag1", "tag2"],\n'
        '  "related_tags": ["rtag1", "rtag2"],\n'
        '  "role": "A single concise sentence describing this specific file\'s responsibility in the project.",\n'
        '  "layer": "The classification layer of the file (e.g., Practices & Guidelines, System Reference, Design Specification, Sourcebook Chapter, API, Legacy Archive, Utility)",\n'
        '  "context": "A concise 1-2 sentence paragraph explaining what the document is about and what specific problem or topic it addresses.",\n'
        '  "overview": "A concise paragraph summarizing the high-level concepts, architecture, or structure described in the document."\n'
        "}\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Do NOT rewrite or return the bulk details of the document. Only output the metadata requested in the JSON schema.\n"
        "- If the document already contains a 'Context' or 'Overview' section, read them and refine/condense them into the JSON fields rather than inventing them from scratch.\n"
        "- Output ONLY valid JSON. Do NOT wrap it in markdown backticks or include any extra text."
    )

    for filepath in candidates:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                original_content = f.read()

            title_match = re.search(r"^#\s+(.+)$", original_content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                title = (
                    os.path.splitext(os.path.basename(filepath))[0]
                    .replace("_", " ")
                    .title()
                )


            existing_context = ""
            existing_overview = ""
            context_match = re.search(
                r"## Context\s*\n(.*?)(?:\n##|\n---)",
                original_content,
                re.DOTALL | re.IGNORECASE,
            )
            if context_match:
                existing_context = context_match.group(1).strip()
            overview_match = re.search(
                r"## Overview\s*\n(.*?)(?:\n##|\n---)",
                original_content,
                re.DOTALL | re.IGNORECASE,
            )
            if overview_match:
                existing_overview = overview_match.group(1).strip()

            user_prompt = f"Title: {title}\n"
            if existing_context:
                user_prompt += f"Existing Context: {existing_context}\n"
            if existing_overview:
                user_prompt += f"Existing Overview: {existing_overview}\n"
            user_prompt += f"Document content:\n```markdown\n{original_content}\n```"

            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            tokens = processor.get_token_count(prompt)
            if tokens <= MAX_INPUT_TOKENS:
                valid_candidates.append({"filepath": filepath, "tokens": tokens})
            else:
                rel_path = os.path.relpath(filepath, project_root)
                print(
                    f"  [SKIP] {rel_path}: Prompt size ({tokens} tokens) exceeds MAX_INPUT_TOKENS ({MAX_INPUT_TOKENS})."
                )
        except Exception as e:
            rel_path = os.path.relpath(filepath, project_root)
            print(f"  [ERROR] Could not measure {rel_path}: {e}")

    if not valid_candidates:
        print("No valid candidates within input size limits found.")
        return 0

    # Sort candidates descending (largest context size first)
    valid_candidates.sort(key=lambda x: cast(int, x["tokens"]), reverse=True)

    # Establish actual limit bounds
    actual_limit = min(2 if dry_run else limit, len(valid_candidates))
    if dry_run:
        print(
            f"\n=== DRY RUN MODE: Will process up to {actual_limit} candidate(s) (sorted largest -> smallest) ==="
        )
    else:
        print(
            f"\nProcessing up to {actual_limit} candidate(s) (sorted largest -> smallest context size)..."
        )

    processed_count = 0
    write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    write_futures: list[concurrent.futures.Future[None]] = []

    for item in valid_candidates:
        if processed_count >= actual_limit:
            break

        filepath = cast(str, item["filepath"])
        rel_path = os.path.relpath(filepath, project_root)
        print(
            f"\n[{processed_count+1}/{actual_limit}] Processing: {rel_path} (Context: {item['tokens']} tokens)"
        )

        normalized_text = prepare_normalization(
            filepath=filepath, processor=processor, project_root=project_root
        )

        if normalized_text:
            processed_count += 1
            future = write_executor.submit(
                background_write_and_virtualize,
                filepath,
                normalized_text,
                project_root,
                dry_run,
            )
            write_futures.append(future)

    if write_futures:
        print(
            f"\nWaiting for {len(write_futures)} background write/virtualization task(s) to complete..."
        )
        concurrent.futures.wait(write_futures)

    write_executor.shutdown()
    print(f"\nDone! Successfully processed and virtualized {processed_count} file(s).")
    return processed_count


def main():
    parser = argparse.ArgumentParser(
        description="Normalize documentation files to structured format using local LLM."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only process the first 2 files and write to scratch/dry_run/",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Max files to process in this run"
    )
    parser.add_argument("--file", type=str, help="Process a single specific file")
    args = parser.parse_args()

    project_root = get_project_root()
    config_mgr = ConfigManager()

    # Get doc directories dynamically
    doc_dirs = config_mgr.get_doc_directories()
    print(f"Resolved doc directories from default-rules.md: {doc_dirs}")

    registered_files = load_transparency_registry(project_root)
    print(
        f"Loaded {len(registered_files)} registered files from transparency registry."
    )

    # Find candidate files
    try:
        candidates = find_normalization_candidates(
            project_root=project_root,
            doc_dirs=doc_dirs,
            registered_files=registered_files,
            file_path_arg=args.file,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    total_candidates = len(candidates)
    print(
        f"Found {total_candidates} candidate files that need normalization or placeholder resolution."
    )

    if total_candidates == 0:
        print("All files are fully compliant and resolved!")
        return 0

    # Initialize Local LLM
    model_path = os.path.join(
        project_root, "models", "Qwen3-4B-Instruct-2507-Q8_0.gguf"
    )
    if not os.path.exists(model_path):
        print(f"Error: Local LLM model not found at {model_path}")
        return 1

    print(f"Initializing local LLM from {model_path}...")
    processor = LocalLLMProcessor(model_path=model_path)

    # Use the reusable batch normalization function
    normalize_docs_batch(
        candidates=candidates,
        processor=processor,
        project_root=project_root,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    processor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
