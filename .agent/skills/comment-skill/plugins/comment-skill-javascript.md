## comment-skill-javascript.md -- PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   `.js`, `.ts`, `.tsx`, `.jsx` -- JavaScript, TypeScript, React
OVERRIDES:
  - Station Header: JSDoc file-level comment variant
  - Connection Map: defers entirely to core single-line `CONNECTION_MAP` format
ADDS:    JSDoc alignment rules; ESM/CJS module pattern docs;
         React component lifecycle docs; Promise chain docs; TypeScript patterns

---

## Station Header (JS/TS Variant)

Use a JSDoc `/** */` block at the top of the file. CRCT infrastructure fields
go as `//` comments immediately after the closing `*/`.

```typescript
/**
 * ROLE:    [FILL: 1-2 sentence purpose]
 * LAYER:   Service -- between route handlers and the DB layer.
 * @module auth/tokenManager
 */
// CRCT_KEY:    1Ab2                          [AUTO]
// TRACKER_REF: src/auth/auth_module.md       [AUTO]
```

For React component files, add `ENTRY FROM` as the rendered-by parent component --
UI is particularly hard to navigate without knowing the render chain.

---

## Connection Map (JS/TS Variant)

Use the **core single-line `CONNECTION_MAP` format** as a `//` comment immediately
above the `export function`, `export class`, or React component declaration.

```typescript
// --- CONNECTION_MAP: 1Ab2 >, 3Ba1 <, 2C3 x --- validateToken [AUTO]
export async function validateToken(token: string): Promise<string | null> {
```

```typescript
// --- CONNECTION_MAP: 2Aa3 >, 1B1 < --- LoginForm [AUTO]
export function LoginForm({ onSuccess, redirectTo }: LoginFormProps) {
```

**Rules:**
- One line per exported function, class, or component with external connections.
- Private/unexported symbols: add a CONNECTION_MAP line only if they have
  cross-file dependencies (e.g., call an imported utility).
- TypeScript type signatures encode input/output -- do not restate them.
- `none` for symbols with zero external file connections.
- For `.tsx`/`.jsx` files, ensure the Station Header `ENTRY FROM` lists the
  render-parent component.

---

## Promise / Async Chains

For async functions with complex chain topology, add a single Goto Pointer below
the CONNECTION_MAP line:

```typescript
// --- CONNECTION_MAP: 1Aa1 >, 2B3 <, 3Ca1 x --- processOrder [AUTO]
// -> Chain: validateOrder -> reserveInventory -> chargePayment -> sendReceipt (sequential)
// -> Error strategy: each step throws OrderProcessingError with step identifier
async function processOrder(orderId: string): Promise<OrderResult>
```

Only document chain topology when the sequence is non-obvious or when parallelism
is intentionally avoided.

---

## ESM / CJS Module Boundaries

Document module boundary decisions that affect bundling or runtime behavior:

```typescript
// MODULE: ESM -- named exports for tree-shaking. Do not convert to default export
// without checking: src/api/routes/auth.ts, src/workers/tokenWorker.ts
// -> Re-exported from: src/auth/index.ts (public surface)
export { validateToken, issueToken, revokeToken };
```

---

## TypeScript Type Aliases / Interfaces

```typescript
// TYPE: AuthPayload -- decoded JWT body shape
// USED BY: validateToken() (output), issueToken() (input to jwt.sign())
// -> JWT library fields must match jsonwebtoken expected payload shape
interface AuthPayload {
  userId: string;
  role: UserRole;    // -> UserRole enum: src/auth/types.ts
  exp: number;       // Unix epoch SECONDS, not milliseconds
  iat: number;
}
```

---

## React Hooks (useEffect / useMemo)

Document non-obvious hook dependency arrays and their rationale:

```typescript
useEffect(() => {
  // EFFECT: WebSocket subscription for live order status updates
  // RUNS: On mount and when orderId changes
  // CLEANUP: Unsubscribes on unmount/orderId change to prevent memory leak
  // -> WebSocket manager: src/services/wsManager.ts:subscribe()
  const unsubscribe = wsManager.subscribe("order:update", handleUpdate, orderId);
  return () => unsubscribe();
}, [orderId]); // handleUpdate is stable (useCallback) -- not a missing dep
```

---

## WIP Beacons

Always `//` block format, not JSDoc. WIP content in JSDoc risks ending up in
generated documentation or IDE hover hints.

```typescript
// +--------------------------------------------------------------------
// | WIP: AuthProvider
// | INTENT:   Migrate from local state to persistent session cookies.
// | STATUS:   Cookie setting/reading in login() works. Logout does not
// |           clear the cookie yet.
// | NEXT:     1. Add deleteCookie("session_id") to logout handler
// |           2. Verify cross-tab logout via StorageEvent listener
// | REQUIRES: utils/cookie_manager.ts exported and tested.
// +--------------------------------------------------------------------
```

---

## File-Type-Specific Rules

1. JSDoc `/** */` block = Station Header. No duplicate `// ===` blocks.
2. CONNECTION_MAP is a `//` line above the export declaration, outside JSDoc.
3. TypeScript type annotations reduce comment burden -- do not restate typed shapes.
4. WIP Beacons use `//` block format, not JSDoc.
5. Goto Pointers use `// ->` prefix.
6. For `.tsx`/`.jsx`: `ENTRY FROM` in Station Header must name the render-parent.

---

## CRCT Integration Notes

- `CRCT_KEY` and `TRACKER_REF` as `//` comments after the JSDoc Station Header block.
- CONNECTION_MAP keys from `global_key_map.json`; refresh via
  `cline_utils/dependency_system/utils/populate_comments.py`.

-> For CRCT field definitions: plugins/comment-skill-crct.md
-> For core comment categories: SKILL.md