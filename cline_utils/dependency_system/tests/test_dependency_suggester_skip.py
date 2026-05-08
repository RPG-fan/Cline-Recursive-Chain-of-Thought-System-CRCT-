from cline_utils.dependency_system.analysis import dependency_suggester
from cline_utils.dependency_system.core.key_manager import KeyInfo


def _key_info(key: str, path: str) -> KeyInfo:
    norm_path = dependency_suggester.normalize_path(path)
    return KeyInfo(
        key_string=key,
        norm_path=norm_path,
        parent_path=dependency_suggester.normalize_path(
            str(norm_path.rsplit("/", 1)[0])
        ),
        tier=1,
        is_directory=False,
    )


def test_sql_structural_suggestion_is_filtered_when_pair_already_x(
    tmp_path, monkeypatch
):
    source_path = dependency_suggester.normalize_path(
        str(tmp_path / "00_relationship_types.sql")
    )
    target_path = dependency_suggester.normalize_path(
        str(tmp_path / "01_relationships.sql")
    )
    (tmp_path / "00_relationship_types.sql").write_text(
        "create table relationship_types (id int);", encoding="utf-8"
    )
    (tmp_path / "01_relationships.sql").write_text(
        "create table relationships (type_id int);", encoding="utf-8"
    )

    source_ki = _key_info("1A1", source_path)
    target_ki = _key_info("1A2", target_path)
    path_to_key_info = {
        source_path: source_ki,
        target_path: target_ki,
    }
    file_analysis_results = {
        source_path: {
            "tables_defined": ["relationship_types"],
            "tables_referenced": [],
        }
    }
    monkeypatch.setattr(
        dependency_suggester,
        "load_project_symbol_map",
        lambda: {
            source_path: {"tables_defined": ["relationship_types"]},
            target_path: {"tables_referenced": ["relationship_types"]},
        },
    )

    suggestions, _ast_links = dependency_suggester.suggest_dependencies(
        source_path,
        path_to_key_info,
        dependency_suggester.normalize_path(str(tmp_path)),
        file_analysis_results,
        existing_state={
            (source_ki.key_string, target_ki.key_string): ("x", {"tracker"})
        },
    )

    assert suggestions == []


def test_sql_structural_suggestion_is_filtered_when_reverse_pair_already_n(
    tmp_path, monkeypatch
):
    source_path = dependency_suggester.normalize_path(
        str(tmp_path / "01_relationships.sql")
    )
    target_path = dependency_suggester.normalize_path(
        str(tmp_path / "00_relationship_types.sql")
    )
    (tmp_path / "01_relationships.sql").write_text(
        "create table relationships (type_id int);", encoding="utf-8"
    )
    (tmp_path / "00_relationship_types.sql").write_text(
        "create table relationship_types (id int);", encoding="utf-8"
    )

    source_ki = _key_info("1A2", source_path)
    target_ki = _key_info("1A1", target_path)
    path_to_key_info = {
        source_path: source_ki,
        target_path: target_ki,
    }
    file_analysis_results = {
        source_path: {
            "tables_defined": [],
            "tables_referenced": ["relationship_types"],
        }
    }
    monkeypatch.setattr(
        dependency_suggester,
        "load_project_symbol_map",
        lambda: {
            source_path: {"tables_referenced": ["relationship_types"]},
            target_path: {"tables_defined": ["relationship_types"]},
        },
    )

    suggestions, _ast_links = dependency_suggester.suggest_dependencies(
        source_path,
        path_to_key_info,
        dependency_suggester.normalize_path(str(tmp_path)),
        file_analysis_results,
        existing_state={
            (target_ki.key_string, source_ki.key_string): ("n", {"tracker"})
        },
    )

    assert suggestions == []
