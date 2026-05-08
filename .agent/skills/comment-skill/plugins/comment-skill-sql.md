## comment-skill-sql.md -- PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   `.sql`, `db/migrations/`, schema definitions
OVERRIDES:
  - Connection Map: single-line `-- --- CONNECTION_MAP:` sentinel for the
    source model file reference PLUS a domain RELATIONSHIPS block for foreign
    key and index topology. The RELATIONSHIPS block is a named domain comment,
    NOT a Connection Map.
ADDS:    Relationship maps; Migration dependency chains; Index intent

---

## SQL Station Header

SQL files declare the schema layer and their position in the migration history:

```sql
-- ============================================================
-- ROLE:    Creates the `revoked_tokens` table for JWT revocation.
-- LAYER:   Database Schema / Migration
-- CRCT_KEY:   2Bb4                                    [AUTO]
-- ============================================================
```

---

## Connection Map (SQL variant)

The CONNECTION_MAP sentinel references the primary source-code model or data
access layer that owns this table:

```sql
-- --- CONNECTION_MAP: 3Ba2 > --- revoked_tokens [AUTO]
```

Following the CONNECTION_MAP line, the RELATIONSHIPS domain block documents
foreign key and index topology. This is a **domain comment**, not a Connection Map.
FK relationships are table-level entities, not file-level CRCT key pairs:

```sql
-- --- CONNECTION_MAP: src/auth/token_manager.py > --- revoked_tokens [AUTO]
-- REFS: users(id) ON DELETE CASCADE | INDEX: token_jti UNIQUE hash
CREATE TABLE revoked_tokens (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token_jti   VARCHAR(255) UNIQUE NOT NULL,
    revoked_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Format rules for the RELATIONSHIPS block:**
- `FK:` for foreign key constraints with delete/update behavior.
- `UNIQUE:` / `INDEX:` for index intent and performance rationale.
- `CONSUMED BY:` for the primary ORM/data-access model that reads this table.
- Keep entries brief -- one line each where possible.

---

## Migration Chains

Always document the previous and expected next migration to prevent out-of-order
execution errors:

```sql
-- -> Previous migration: db/migrations/003_add_user_roles.sql
-- -> Expected next:      db/migrations/005_add_audit_log.sql
```

---

## File-Type Rules

1. **One CONNECTION_MAP line** references only the primary source-code model file.
2. **RELATIONSHIPS domain block** documents FK/index topology -- separate from
   CONNECTION_MAP. Labeled `RELATIONSHIPS:` not `CONNECT:` to distinguish it.
3. **Migration chain Goto Pointers** at the top of every migration file.
4. **CRCT_KEY** goes inside the Station Header block.
5. **WIP Beacons use `--` block format** consistent with SQL comment syntax.

---

## Example: Full migration file header

```sql
-- ============================================================
-- ROLE:    Creates the `revoked_tokens` table for JWT revocation.
-- LAYER:   Database Schema / Migration
-- CRCT_KEY:   2Bb4                                    [AUTO]
-- ============================================================
-- -> Previous migration: db/migrations/003_add_user_roles.sql
-- -> Expected next:      db/migrations/005_add_audit_log.sql

-- --- CONNECTION_MAP: src/auth/token_manager.py > --- revoked_tokens [AUTO]
-- REFS: users(id) ON DELETE CASCADE | INDEX: token_jti UNIQUE hash
CREATE TABLE revoked_tokens (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token_jti   VARCHAR(255) UNIQUE NOT NULL,
    revoked_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## CRCT Integration Notes

- Track table status in `db_tracker.md` if your project uses one.
- `CRCT_KEY` inside the Station Header block -- SQL files are tracked in
  `global_key_map.json` like any other source file.
- The RELATIONSHIPS domain block is intentionally outside the CRCT key system.
  Model mapping is done via the CONNECTION_MAP sentinel pointing to the
  ORM/data-access layer.

-> For CRCT field definitions: `plugins/comment-skill-crct.md`
-> For core comment categories: `SKILL.md`