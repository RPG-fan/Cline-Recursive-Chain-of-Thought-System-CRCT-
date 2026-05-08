## comment-skill-csharp.md -- PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   `.cs` files -- all C# source, with Unity MonoBehaviour lifecycle emphasis
OVERRIDES:
  - Station Header: XML `<summary>` variant at class level
  - Connection Map: defers entirely to core single-line `CONNECTION_MAP` format
ADDS:    Unity MonoBehaviour lifecycle docs; serialized field docs;
         coroutine chain docs; ScriptableObject patterns; Jobs/Burst annotation;
         XML doc comment alignment; `[UNITY:]` tag

---

## Station Header (C# Variant)

C# Station Headers live inside the XML `<summary>` block on the class declaration.
CRCT infrastructure fields go as `//` comments immediately after the closing `>` of
the XML block, before the class keyword.

```csharp
/// <summary>
/// ROLE:    [FILL: 1-2 sentence purpose]
/// LAYER:   System -- sits between NetworkManager and PlayerController.
/// </summary>
// CRCT_KEY:    3Cd4                          [AUTO]
// TRACKER_REF: Assets/Scripts/Systems/systems_module.md [AUTO]
public class TokenManager : MonoBehaviour
```

**Rule:** The XML `<summary>` IS the Station Header. Do not add a separate `// ===`
block above it. One authoritative block per class.

---

## Connection Map (C# Variant)

Use the **core single-line `CONNECTION_MAP` format** as a `//` comment immediately
above the method or class opening, for both public and private symbols.

```csharp
// --- CONNECTION_MAP: 2Ab1 >, 1Ac3 <, 1B2 x --- ValidateToken [AUTO]
public string ValidateToken(string token)
{
```

```csharp
// --- CONNECTION_MAP: 1Aa2 < --- BuildTokenPayload [AUTO]
private Dictionary<string, object> BuildTokenPayload(string playerID)
{
```

**Rules:**
- One line per method/class with external connections.
- Public API methods: `//` comment above the method (before any XML `<summary>`).
- Private/internal methods: same format, same position.
- `none` for methods with zero external file connections.
- XML `<summary>` and `<remarks>` still hold purpose/param docs -- the CONNECTION_MAP
  line is navigation infrastructure, separate from API documentation.

---

## Unity MonoBehaviour Lifecycle

Document the full lifecycle sequence once per class, immediately after the Station
Header. Each implemented lifecycle method gets a one-line role comment.

```csharp
// LIFECYCLE: Awake -> OnEnable -> Start -> [FixedUpdate -> Update -> LateUpdate] -> OnDisable -> OnDestroy
// This class: Awake (cache refs), Start (register input callbacks), FixedUpdate (physics)

private void Awake()
{
    // INIT: Cache components -- safe before scene fully loads; no cross-object refs here
    _rb = GetComponent<Rigidbody2D>();
    _animator = GetComponent<Animator>();
}

private void Start()
{
    // INIT: Cross-object setup -- all Awake() calls complete by now
    // -> Game state: GameManager.cs -> GameManager.Instance.RegisterPlayer()
    GameManager.Instance.RegisterPlayer(this);
}
```

Note any lifecycle methods intentionally omitted when non-obvious:
```csharp
// OnDestroy not implemented -- cleanup handled by ObjectPool
```

---

## Serialized Fields

Document `[SerializeField]` fields only when there are non-obvious constraints,
tuning ranges, or cross-component dependencies.

```csharp
[Header("Movement Config")]
[SerializeField]
// TUNABLE: Base speed in world-units/second. Range: 3-8. Applied in Update() -> MovePlayer().
private float _moveSpeed = 5f;

[SerializeField]
// TUNABLE: Jump impulse. GUARD: Do not exceed 15 -- causes platform clipping.
// -> Applied in: HandleJump() -> _rb.AddForce()
private float _jumpForce = 8f;
```

Do not comment self-evident fields (`[SerializeField] private string _playerName`).

---

## Coroutine Chains

```csharp
// COROUTINE: SpawnEnemyWave
// TRIGGERED BY: GameManager.cs -> StartWave()
// DURATION:  ~(waveCount * _spawnInterval) seconds
// BLOCKS:    Player input disabled for first 2s (InvincibilityBuffer)
// -> Enemy prefabs: Assets/Prefabs/Enemies/ (Resources.Load)
private IEnumerator SpawnEnemyWave(int waveCount)
```

---

## Unity Events

```csharp
// EVENT: onPlayerDeath
// PUBLISHERS: PlayerHealth.cs -> TakeDamage() when health == 0
// SUBSCRIBERS: GameManager.cs -> HandlePlayerDeath(), UIManager.cs -> ShowDeathScreen()
// -> Response logic: GameManager.cs:HandlePlayerDeath()
public UnityEvent onPlayerDeath;
```

---

## Unity Jobs / Burst

```csharp
// JOB: PathfindingJob (IJobParallelFor) -- Burst-compiled, worker threads, NO managed allocs
// INPUT:  NativeArray<float2> startPositions (read-only), NativeArray<float2> targetPositions (read-only)
// OUTPUT: NativeArray<float2> results (write)
// -> Caller: PathfindingSystem.cs -> SchedulePathfinding()
[BurstCompile]
public struct PathfindingJob : IJobParallelFor
```

---

## `[UNITY:]` Tag

Use for Unity-specific architecture constraints that are not connection information:

```csharp
// [UNITY: Singleton -- ensure only one instance exists in the scene]
// [UNITY: Requires PhysicsMaterial2D on Collider2D -- set in Inspector]
```

---

## File-Type-Specific Rules

1. XML `<summary>` = Station Header for the class. No separate `// ===` blocks.
2. CONNECTION_MAP is a `//` line above the method/class, outside any XML doc blocks.
3. `[UNITY:]` tags for architecture constraints not covered by CONNECTION_MAP.
4. Lifecycle sequence comment at class top when 2+ lifecycle methods are implemented.
5. WIP Beacons use `//` block format, not XML comments.
6. Goto Pointers use `// ->` prefix.

---

## CRCT Integration Notes

- `CRCT_KEY` and `TRACKER_REF` as `//` comments after the XML `<summary>` block,
  before the class keyword.
- CONNECTION_MAP keys from `global_key_map.json`; refresh via
  `cline_utils/dependency_system/utils/populate_comments.py`.

-> For CRCT field definitions: plugins/comment-skill-crct.md
-> For core comment categories: SKILL.md