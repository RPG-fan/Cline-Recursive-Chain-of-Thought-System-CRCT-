## comment-skill-python.md -- PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   `.py` files -- all Python source files
OVERRIDES:
  - Station Header: module docstring variant (not `# ===` blocks)
  - Connection Map: defers entirely to core single-line `CONNECTION_MAP` format
ADDS:    Decorator handling; async/await annotation; dataclass field docs;
         property/descriptor patterns; docstring alignment rules

---

## Station Header (Python Variant)

Python Station Headers live inside the **module-level docstring**, not in a
`# ===` comment block. CRCT infrastructure fields (`CRCT_KEY`, `TRACKER_REF`)
are placed as `#` comments immediately after the docstring closes.

```python
"""
ROLE:    [FILL: 1-2 sentence purpose]
LAYER:   Service -- sits between route handlers and the DB layer.
"""
# CRCT_KEY:    1Ab2                          [AUTO]
# TRACKER_REF: src/auth/auth_module.md       [AUTO]
```

**Rules:**
- Module docstring IS the Station Header. Never duplicate fields in both a docstring
  and a `# ===` block.
- If the file already has a module docstring, extend it with Station Header fields
  rather than adding a second docstring.
- `CRCT_KEY` / `TRACKER_REF` go below the closing `"""` as `#`-comments -- they are
  infrastructure metadata, not documentation content.

---

## Connection Map (Python Variant)

Use the **core single-line `CONNECTION_MAP` format** as a `#` comment immediately
above the `def` or `class` keyword -- outside the docstring.

```python
# --- CONNECTION_MAP: 1Ab2 >, 3Ba1 <, 2C3 x --- process_payment [AUTO]
def process_payment(order_id: str, payment_method: PaymentMethod) -> PaymentResult:
    """Regular docstring: purpose, Args, Returns, Raises. No connection fields here."""
```

**Why outside the docstring:** The `CONNECTION_MAP` line is machine-managed (`[AUTO]`).
Embedding it inside docstrings makes the populate utility harder to update without
corrupting docstring formatting. Docstrings are human-readable API docs; CONNECTION_MAP
is navigation infrastructure.

**Rules:**
- One line per public function, class, or property with external connections.
- `none` when the symbol has zero external file connections.
- Type annotations encode input/output shapes -- do not duplicate them here.
- For `async def`, no special annotation needed in the CONNECTION_MAP line -- async
  nature is visible from the signature.

---

## Async / Await

For async functions with non-obvious awaitable dependency chains, add a single
Goto Pointer below the CONNECTION_MAP line:

```python
# --- CONNECTION_MAP: 1Ab2 >, 3Ba1 < --- fetch_user_profile [AUTO]
# -> Async deps: db/user_store.py:UserStore.find_async(), cache/redis_client.py:RedisClient.get()
async def fetch_user_profile(user_id: str) -> UserProfile:
```

Do not create a separate `AWAITS:` block. The Goto Pointer covers it.

---

## Decorator Handling

Document decorators whose effect is non-obvious from the decorator name alone.
Place the comment immediately above the decorator stack.

```python
# -> Route: POST /api/v1/auth/login -- bypasses token check (issues tokens here)
@app.route("/api/v1/auth/login", methods=["POST"])
@bypass_auth
def login():
```

Do not comment `@staticmethod`, `@property`, `@dataclass`, `@classmethod` -- self-explanatory.
Only document decorators that change runtime behavior, lifecycle, or routing.

---

## Dataclass Fields

Place the CONNECTION_MAP above the `@dataclass` decorator. Annotate individual fields
inline only when there is a non-obvious constraint or cross-reference.

```python
# --- CONNECTION_MAP: 3Ba1 >, 2C1 > --- PaymentResult [AUTO]
@dataclass
class PaymentResult:
    success: bool           # True only if gateway confirmed charge
    transaction_id: str     # Gateway-assigned; empty string on failure
    error_code: Optional[str]  # -> Error catalog: errors/payment_errors.py
    amount_charged: Decimal    # Actual charged (may differ from requested)
```

Do not comment self-evident fields like `user_id: str` or `created_at: datetime`.

---

## Property / Descriptor Patterns

```python
@property
def is_expired(self) -> bool:
    # COMPUTE: Compares self._exp (UTC epoch) against time.time()
    # -> Expiry config: config/jwt_config.py:JWT_EXPIRY_SECONDS
    return time.time() > self._exp
```

---

## WIP Beacons

Always `#`-style blocks, never docstrings. WIP content in docstrings risks
surviving into generated API documentation.

```python
# +--------------------------------------------------------------------
# | WIP: OrderState -- add `payment_method` field
# | INTENT:   Persist payment method so refunds route correctly.
# | STATUS:   Field absent. DB migration not written.
# | NEXT:     1. Add `payment_method: Optional[str] = None` to dataclass
# |           2. Write: db/migrations/004_add_payment_method.py
# |           3. Update Orders.get() to populate field from DB row
# | REQUIRES: Migration 003 applied first.
# |           -> db/migrations/003_add_line_items.py
# +--------------------------------------------------------------------
@dataclass
class OrderState:
```

---

## File-Type-Specific Rules

1. Module docstring = Station Header. No `# ===` blocks in `.py`.
2. CONNECTION_MAP goes above `def`/`class`, outside the docstring.
3. Type annotations reduce comment burden -- do not restate typed signatures.
4. WIP Beacons are `#`-style only, never docstrings.
5. Goto Pointers use `# ->` prefix, consistent with core rules.

---

## CRCT Integration Notes

- `CRCT_KEY` and `TRACKER_REF` as `#` comments after the module docstring.
- CONNECTION_MAP keys sourced from `global_key_map.json`; refresh via
  `cline_utils/dependency_system/utils/populate_comments.py` after every `analyze-project` run.
- `HDTA_REF` placed below `TRACKER_REF` as a `#` comment (agent-filled, no [AUTO]).

-> For CRCT field definitions: plugins/comment-skill-crct.md
-> For core comment categories: SKILL.md