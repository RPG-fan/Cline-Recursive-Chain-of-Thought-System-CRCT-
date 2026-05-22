from unittest.mock import patch, MagicMock


from code_analysis.report_generator import main


@patch("code_analysis.report_generator.config")
@patch("code_analysis.report_generator.subprocess.run")
@patch("code_analysis.report_generator.maybe_run_runtime_inspector")
@patch("code_analysis.report_generator.load_runtime_data")
@patch("code_analysis.report_generator.RuntimeIndex")
@patch("code_analysis.report_generator.os.walk")
@patch("code_analysis.report_generator.os.path.exists")
@patch("code_analysis.report_generator.scan_file")
@patch("code_analysis.report_generator.runtime_only_findings")
@patch("code_analysis.report_generator.enrich_issue")
@patch("code_analysis.report_generator.get_unused_items")
@patch("code_analysis.report_generator.export_json")
@patch("code_analysis.report_generator.format_markdown")
@patch("builtins.open", new_callable=MagicMock)
def test_main_happy_path(
    mock_open, mock_format_markdown, mock_export_json, mock_get_unused_items,
    mock_enrich_issue, mock_runtime_only_findings, mock_scan_file,
    mock_exists, mock_walk, mock_runtime_index, mock_load_runtime_data,
    mock_maybe_run_inspector, mock_subprocess_run, mock_config
):
    """Test the happy path of the main orchestration script."""
    # Setup mocks
    mock_config.get_code_root_directories.return_value = ["src"]
    mock_config.get_excluded_paths.return_value = ["tests"]

    mock_subprocess_result = MagicMock()
    mock_subprocess_result.returncode = 0
    mock_subprocess_run.return_value = mock_subprocess_result

    mock_load_runtime_data.return_value = {"some": "data"}

    mock_exists.return_value = True

    # Mock os.walk to return one file
    mock_walk.return_value = [("src", [], ["test.py"])]

    # Mock scan_file returning one issue
    mock_scan_file.return_value = [{"file": "src/test.py", "line": 1, "subtype": "test_issue"}]

    # Mock runtime_only_findings returning one finding
    mock_runtime_only_findings.return_value = [{"file": "src/other.py", "line": 2, "subtype": "rt_issue"}]

    # Mock enrich_issue to just pass through
    def enrich_side_effect(issue, idx):
        return {**issue, "context": {"severity": "High"}}
    mock_enrich_issue.side_effect = enrich_side_effect

    mock_get_unused_items.return_value = ["unused_func"]

    # Run the main function
    main()

    # Verify the sequence of calls
    mock_config.get_code_root_directories.assert_called_once()
    mock_subprocess_run.assert_called_once()
    mock_maybe_run_inspector.assert_called_once()
    mock_load_runtime_data.assert_called_once()
    mock_walk.assert_called_once()
    mock_scan_file.assert_called_once()
    mock_runtime_only_findings.assert_called_once()
    
    # Verify get_unused_items was called with project_root
    from code_analysis.report_generator import project_root
    mock_get_unused_items.assert_called_once_with(project_root)

    # export_json and format_markdown should be called with 2 issues (1 from scan, 1 from runtime)
    # and the unused items
    mock_export_json.assert_called_once()
    args, _ = mock_export_json.call_args
    assert len(args[0]) == 2 # 2 issues
    assert args[1] == ["unused_func"] # unused items

    mock_format_markdown.assert_called_once()


@patch("code_analysis.report_generator.config")
@patch("code_analysis.report_generator.subprocess.run")
@patch("code_analysis.report_generator.load_runtime_data")
@patch("code_analysis.report_generator.os.path.exists")
@patch("code_analysis.report_generator.os.walk")
@patch("code_analysis.report_generator.export_json")
@patch("code_analysis.report_generator.format_markdown")
@patch("builtins.open", new_callable=MagicMock)
def test_main_missing_code_root(
    mock_open, mock_format_markdown, mock_export_json, mock_walk, mock_exists,
    mock_load_runtime_data, mock_subprocess_run, mock_config
):
    """Test main when code root directory is missing."""
    mock_config.get_code_root_directories.return_value = ["missing_dir"]
    mock_subprocess_run.return_value = MagicMock(returncode=0)
    mock_load_runtime_data.return_value = None

    # Simulate missing directory
    mock_exists.return_value = False

    main()

    # os.walk should not be called if directory doesn't exist
    mock_walk.assert_not_called()

    # Reports should still be generated, but with empty issues
    mock_export_json.assert_called_once()
    mock_format_markdown.assert_called_once()
    args, _ = mock_export_json.call_args
    assert len(args[0]) == 0


@patch("code_analysis.report_generator.config")
@patch("code_analysis.report_generator.subprocess.run")
@patch("code_analysis.report_generator.load_runtime_data")
@patch("code_analysis.report_generator.os.path.exists")
@patch("code_analysis.report_generator.os.walk")
@patch("code_analysis.report_generator.export_json")
@patch("code_analysis.report_generator.format_markdown")
@patch("builtins.open", new_callable=MagicMock)
def test_main_pyright_error(
    mock_open, mock_format_markdown, mock_export_json, mock_walk, mock_exists,
    mock_load_runtime_data, mock_subprocess_run, mock_config, capsys
):
    """Test main handles pyright subprocess exception gracefully."""
    mock_config.get_code_root_directories.return_value = []

    # Simulate exception during pyright run
    mock_subprocess_run.side_effect = Exception("Pyright failed miserably")
    mock_load_runtime_data.return_value = None

    main()

    # Verify exception was caught and printed
    captured = capsys.readouterr()
    assert "Warning: Unexpected error running pyright: Pyright failed miserably" in captured.out

    # Reports should still be generated
    mock_export_json.assert_called_once()


@patch("code_analysis.report_generator.config")
@patch("code_analysis.report_generator.subprocess.run")
@patch("code_analysis.report_generator.load_runtime_data")
@patch("code_analysis.report_generator.os.walk")
@patch("code_analysis.report_generator.os.path.exists")
@patch("code_analysis.report_generator.scan_file")
@patch("code_analysis.report_generator.export_json")
@patch("code_analysis.report_generator.format_markdown")
@patch("code_analysis.scanner.runtime_bridge.score_severity")
@patch("builtins.open", new_callable=MagicMock)
def test_main_no_runtime_data(
    mock_open, mock_score_severity, mock_format_markdown, mock_export_json,
    mock_scan_file, mock_exists, mock_walk, mock_load_runtime_data,
    mock_subprocess_run, mock_config
):
    """Test main processes static analysis correctly without runtime data."""
    mock_config.get_code_root_directories.return_value = ["src"]
    mock_config.get_excluded_paths.return_value = []

    mock_subprocess_run.return_value = MagicMock(returncode=0)

    # No runtime data
    mock_load_runtime_data.return_value = None

    mock_exists.return_value = True
    mock_walk.return_value = [("src", [], ["test.py"])]

    issue = {"file": "src/test.py", "line": 1, "subtype": "static_issue"}
    mock_scan_file.return_value = [issue.copy()]

    mock_score_severity.return_value = "Medium"

    main()

    # Verify score_severity was used as fallback
    mock_score_severity.assert_called_once()

    mock_export_json.assert_called_once()
    args, _ = mock_export_json.call_args
    issues = args[0]
    assert len(issues) == 1
    assert issues[0]["context"]["severity"] == "Medium"
