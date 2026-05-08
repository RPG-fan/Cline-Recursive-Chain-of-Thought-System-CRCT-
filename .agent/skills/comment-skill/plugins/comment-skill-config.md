## comment-skill-config.md — PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   `.yaml`, `.toml`, `.json`, `.env`, `.ini` — configuration files
OVERRIDES: Station Header (consumer-centric variant)
ADDS:    Inline key documentation; sensitive field markers; type hinting for JSON

---

## Consumer-Centric Station Header

For configuration files, the most important information is **who uses it**. The
Station Header must explicitly list the code modules or services that consume
these settings.

### Template (YAML/TOML)
```yaml
# ============================================================
# ROLE:    PostgreSQL connection pool and schema settings.
# CONSUMED BY: src/db/connection_pool.py
#              src/db/migrations/orchestrator.py
# EXITS TO:   ENVIRONMENT (DB_PASSWORD, DB_HOST)
# ============================================================
```

### Template (.env)
```bash
# ROLE:    Environment variable blueprint for local development.
# CONSUMED BY: All services (via python-dotenv or process.env)
# REQUIRED FOR: Docker-compose orchestration
```

---

## Inline Key Documentation

Document non-obvious keys or specific value constraints inline.

### YAML/TOML
```yaml
db_timeout: 30  # Seconds — must be >= 5 to prevent pool starvation
max_retries: 3  # → See: src/db/retry_policy.py for exponential backoff logic
```

### JSON
JSON does not natively support comments. For critical JSON configs, use a
parallel `config_name.json.md` file or a top-level `"_comment"` key:

```json
{
  "_comment": "STATION: config/app_settings.json | CONSUMED BY: src/main.py",
  "feature_flags": {
    "enable_beta_ai": true
  }
}
```

---

## Sensitive Fields and Security

Always mark sensitive fields that must NOT be committed to version control.

```yaml
# [SENSITIVE] Do not provide plain text here. Use env var interpolation.
api_key: ${STRIPE_SECRET_KEY}
```

---

## CRCT Integration

1.  **Dependency Mapping**: `EXITS TO` should point to the environment or external
    APIs (e.g., `Stripe API`, `AWS S3`) if the config manages those connections.
2.  **Tracker Reference**: Many projects track configuration completeness in a
    `config_tracker.md`. Use `TRACKER_REF` to point there.

→ For core rules: SKILL.md
→ For CRCT fields: plugins/comment-skill-crct.md
