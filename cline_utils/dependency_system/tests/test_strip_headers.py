from cline_utils.dependency_system.io.file_io import strip_auto_generated_blocks

def test_strip_python_hyphenated_station_header():
    content = (
        "# --- STATION_HEADER_START --- [AUTO]\n"
        "# ROLE:    Test module role.\n"
        "# LAYER:   Test layer.\n"
        "# --- STATION_HEADER_END --- [AUTO]\n"
        "\n"
        "def hello():\n"
        "    return 'world'\n"
    )
    
    # 1. Test with preserve_lines=True (default)
    stripped_preserve = strip_auto_generated_blocks(content, "test.py", preserve_lines=True)
    expected_preserve = (
        "\n"  # STATION_HEADER_START
        "\n"  # ROLE
        "\n"  # LAYER
        "\n"  # STATION_HEADER_END
        "\n"
        "def hello():\n"
        "    return 'world'\n"
    )
    assert stripped_preserve == expected_preserve
    
    # 2. Test with preserve_lines=False
    stripped_remove = strip_auto_generated_blocks(content, "test.py", preserve_lines=False)
    expected_remove = (
        "\n"
        "def hello():\n"
        "    return 'world'\n"
    )
    assert stripped_remove == expected_remove


def test_strip_javascript_hyphenated_station_header():
    content = (
        "// --- STATION_HEADER_START --- [AUTO]\n"
        "// ROLE:    Test JS role.\n"
        "// --- STATION_HEADER_END --- [AUTO]\n"
        "\n"
        "console.log('hello');\n"
    )
    
    # preserve_lines=False
    stripped = strip_auto_generated_blocks(content, "test.js", preserve_lines=False)
    expected = (
        "\n"
        "console.log('hello');\n"
    )
    assert stripped == expected


def test_strip_markdown_hyphenated_station_header():
    content = (
        "<!-- --- STATION_HEADER_START --- [AUTO]\n"
        "ROLE:    Test MD role.\n"
        "CRCT_KEY:   1Ad8 [AUTO]\n"
        "--- STATION_HEADER_END --- [AUTO] -->\n"
        "\n"
        "# Title\n"
    )
    
    # preserve_lines=False
    stripped = strip_auto_generated_blocks(content, "test.md", preserve_lines=False)
    expected = (
        "\n"
        "# Title\n"
    )
    assert stripped == expected


def test_strip_non_hyphenated_station_header():
    content = (
        "# STATION_HEADER_START\n"
        "# ROLE:    Test non-hyphenated role.\n"
        "# STATION_HEADER_END\n"
        "\n"
        "def test():\n"
        "    pass\n"
    )
    
    stripped = strip_auto_generated_blocks(content, "test.py", preserve_lines=False)
    expected = (
        "\n"
        "def test():\n"
        "    pass\n"
    )
    assert stripped == expected


def test_strip_one_off_auto_tags():
    content = (
        "def test():\n"
        "    # --- CONNECTION_MAP: 1Bd --- test_func [AUTO]\n"
        "    pass\n"
    )
    
    stripped = strip_auto_generated_blocks(content, "test.py", preserve_lines=False)
    expected = (
        "def test():\n"
        "    pass\n"
    )
    assert stripped == expected
