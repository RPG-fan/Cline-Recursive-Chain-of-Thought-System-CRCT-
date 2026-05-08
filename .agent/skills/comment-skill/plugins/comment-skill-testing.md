## comment-skill-testing.md — PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   `test_*.py`, `*.spec.ts`, `*.test.js`, `tests/` — test suites
OVERRIDES: Station Header (test-centric); Connection Map (replaced by TEST INTENT)
ADDS:    Unit Under Test tracking; Fixture navigation; Mock/Stub declarations

---

## Test-Centric Station Header

Test files must immediately declare which part of the production codebase they
are verifying.

### Template
```python
# ============================================================
# ROLE:    Unit tests for JWT issuance, validation, and revocation.
# UNIT UNDER TEST: src/auth/token_manager.py
# LAYER:   Test Suite
# ============================================================
```

---

## TEST INTENT (Replaces Connection Map)

In test files, standard Connection Maps are often redundant. Replace them with
a **TEST INTENT** block for complex test cases or test classes.

### Template
```python
# ┌─ TEST INTENT ──────────────────────────────────────────────────────
# │ SCENARIO: Token revocation via blacklist.
# │ GOAL:     Verify that once a token is added to the Redis blacklist,
# │           subsequent `verify()` calls return `TokenRevokedError`.
# │
# │ MOCKS:    services/redis_client.py → RedisClient.sadd (success return)
# │ FIXTURES: active_user_token (from tests/conftest.py)
# │
# │ → Unit under test: src/auth/token_manager.py:TokenManager.revoke()
# └────────────────────────────────────────────────────────────────────
def test_revoke_token_adds_to_blacklist(mock_redis, active_user_token):
```

---

## Fixture and Mock Navigation

Use Goto Pointers liberally to point to shared fixtures, especially in large
projects where fixtures live in `conftest.py` or separate files.

```python
# → Fixture definition: tests/fixtures/db_fixtures.py:clean_db
@pytest.mark.usefixtures("clean_db")
def test_db_insert():
```

---

## CRCT Integration

1.  **Tracker Link**: Link to the `test_tracker.md` or the `HDTA_TASK` that mandated
    this test suite.
2.  **Coverage Intent**: Use `NOTE: intended to cover [Path/to/file:L100-150]` to
    explicitly state what code segment this test is targeting.

→ For core rules: SKILL.md
→ For CRCT fields: plugins/comment-skill-crct.md
