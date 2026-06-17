from pathlib import Path

from cline_utils.dependency_system.analysis.local_llm_processor import LocalLLMProcessor
from cline_utils.dependency_system.core.key_manager import KeyInfo
from cline_utils.dependency_system.utils.placeholder_resolver import (
    PlaceholderResolver,
    PreparedPair,
)


def _key(key: str, path: Path) -> KeyInfo:
    return KeyInfo(
        key_string=key,
        norm_path=str(path),
        parent_path=str(path.parent),
        tier=1,
        is_directory=False,
    )


def _prepared(
    source_key: str, source_path: str, target_key: str, target_path: str
) -> PreparedPair:
    return PreparedPair(
        srckey=source_key,
        srcpath=source_path,
        tgtkey=target_key,
        tgtpath=target_path,
        srcbase=f"{source_key}.py",
        tgtbase=f"{target_key}.py",
        srccontent=source_key,
        tgtcontent=target_key,
        stokens=1,
        ttokens=1,
    )


def test_placeholder_resolver_pins_kv_state_within_source_group(monkeypatch) -> None:
    class FakeProcessor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def clear_pinned_state(self) -> None:
            self.calls.append("clear")

        def restore_pinned_state(self) -> None:
            self.calls.append("restore")

        def save_pinned_state(self) -> None:
            self.calls.append("save")

        def determine_dependency(self, **_: object) -> tuple[str, str]:
            self.calls.append("determine")
            return "n", "no dependency"

    processor = FakeProcessor()
    resolver = PlaceholderResolver(processor)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "cline_utils.dependency_system.utils.placeholder_resolver.background_commit",
        lambda *args, **kwargs: None,
    )
    source = _key("A", Path("a.py"))
    first_target = _key("B", Path("b.py"))
    second_target = _key("C", Path("c.py"))
    tasks = [
        ("A", source.norm_path, "B", first_target.norm_path),
        ("A", source.norm_path, "C", second_target.norm_path),
        ("B", first_target.norm_path, "C", second_target.norm_path),
    ]

    resolver.resolve_batch(
        tasks=tasks,
        tracker_path="main_tracker.md",
        global_map={
            source.norm_path: source,
            first_target.norm_path: first_target,
            second_target.norm_path: second_target,
        },
        tracker_type="main",
        prepare_func=_prepared,
    )

    assert processor.calls.count("clear") >= 3
    assert "save" in processor.calls
    assert "restore" in processor.calls
    assert processor.calls.index("save") < processor.calls.index("restore")


def test_local_llm_processor_exposes_clear_pinned_state() -> None:
    processor = LocalLLMProcessor(model_path="models/fake.gguf")
    processor._pinned_state = object()  # pyright: ignore[reportPrivateUsage]

    processor.clear_pinned_state()

    assert processor._pinned_state is None  # pyright: ignore[reportPrivateUsage]
