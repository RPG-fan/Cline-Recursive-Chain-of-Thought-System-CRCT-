## comment-skill-glsl.md -- PLUGIN HEADER

EXTENDS: core (SKILL.md)
SCOPE:   `.glsl`, `.hlsl`, `.wgsl`, `.frag`, `.vert` -- shader files
OVERRIDES:
  - Connection Map: single-line `// --- CONNECTION_MAP:` sentinel for CPU-side
    bridge file reference PLUS a domain UNIFORMS block for binding topology.
    The UNIFORMS block is a named domain comment, NOT a Connection Map.
ADDS:    Uniform/Buffer binding docs; Pipeline stage markers;
         Input/Output layout markers

---

## Shader Station Header

Shader files must declare their pipeline stage and the CPU-side bridge class
that binds uniforms and manages the program:

```glsl
// ============================================================
// ROLE:    Fragment shader for procedural water surface ripples.
// STAGE:   Fragment Shader
// BRIDGE:  src/render/materials/WaterMaterial.ts
// CRCT_KEY: 3Ba2                               [AUTO]
// ============================================================
```

---

## Connection Map (Shader variant)

The CONNECTION_MAP sentinel references the CPU bridge file -- the single file
responsible for binding this shader's uniforms and managing the program lifecycle:

```glsl
// --- CONNECTION_MAP: 2Ab3 x --- water_ripple.frag [AUTO]
```

Following the CONNECTION_MAP line, the UNIFORMS domain block documents the
GPU-side binding topology. This is a **domain comment**, not a Connection Map.
It cannot be expressed as CRCT key/char pairs because uniform slots are not
file-level entities in the key map:

```glsl
// --- CONNECTION_MAP: src/render/materials/WaterMaterial.ts <, shaders/wave_compute.glsl < --- uniforms [AUTO]
// -> Uniforms: u_time (float), u_resolution (vec2), WaveData SSBO, u_normalMap (sampler2D)
uniform float u_time;
```

**Format rules for the UNIFORMS block:**
- One uniform per line: `name`, `type`, `<-` or `->` direction, source/target.
- `<-` = value flows from CPU/another stage into this shader.
- `->` = value written out from this shader (e.g., output varyings).
- Keep column-aligned for readability -- trim if file bloat becomes an issue.

---

## Layout and Goto Pointers

Use Goto Pointers to link the vertex/fragment shader pair and any related docs:

```glsl
// -> Vertex shader pair: shaders/water_ripple.vert
// -> Pipeline doc: docs/render/water-pipeline.md
// -> Bridge class: src/render/materials/WaterMaterial.ts
```

---

## File-Type Rules

1. **One CONNECTION_MAP line** references only the CPU bridge file.
2. **UNIFORMS domain block** documents binding topology -- separate from CONNECTION_MAP.
   It is labeled `UNIFORMS:` not `CONNECT:` to distinguish it from the Connection Map system.
3. **Goto Pointers** link shader pairs and pipeline docs.
4. **WIP Beacons use `//` block format** -- no structural difference from other file types.

---

## Example: Full shader file header

```glsl
// ============================================================
// ROLE:    Fragment shader for procedural water surface ripples.
// STAGE:   Fragment Shader
// BRIDGE:  src/render/materials/WaterMaterial.ts
// CRCT_KEY: 3Ba2                               [AUTO]
// ============================================================
// -> Vertex shader pair: shaders/water_ripple.vert
// -> Pipeline doc: docs/render/water-pipeline.md

// --- CONNECTION_MAP: src/render/materials/WaterMaterial.ts <, shaders/wave_compute.glsl < --- uniforms [AUTO]
// -> Uniforms: u_time (float), u_resolution (vec2), WaveData SSBO, u_normalMap (sampler2D)
uniform float u_time;

#version 300 es
precision mediump float;

uniform float u_time;
uniform vec2 u_resolution;
```

---

## CRCT Integration Notes

- `CRCT_KEY` goes inside the Station Header block -- shader files are tracked in
  `global_key_map.json` like any other asset.
- The UNIFORMS domain block is intentionally outside the CRCT key system.
  Bridge tracking is done via the CONNECTION_MAP sentinel pointing to the
  CPU-side material class.

-> For CRCT field definitions: `plugins/comment-skill-crct.md`
-> For core comment categories: `SKILL.md`