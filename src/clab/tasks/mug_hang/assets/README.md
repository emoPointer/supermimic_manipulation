# MugHang assets

The mug is ShapeNet model `b4ae56d6` (an asset identifier, not a revision), as distributed
with robosuite's ShapeNet mug models. The model's intrinsic 0.15 scale is baked into the 2
visual and 32 convex collision meshes. The runtime scale is one, and the body origin,
topology and handle-hole decomposition are unchanged.

| Property | Value |
| --- | --- |
| Size | 69.6 × 104.0 × 82.7 mm |
| Mass | 9.81 g, from collision density 100 kg/m³ (the source value; no substitute) |
| Inertia | Computed from collision geoms only, as in robosuite; matches the source model within numerical precision |
| Contact | Friction (0.95, 0.3, 0.1); solimp (0.998, 0.998, 0.001); solref (0.001, 1) |
| Free joint damping | 0.0005 |

A collision-only ray along local X through `(0, 0.03, 0)` passes through the handle hole.

The fixed tree is built in code from robosuite's `MugTreeObject`:

- base 0.16 × 0.16 × 0.03 m, post 0.03 × 0.03 × 0.16 m, branch 0.08 × 0.005 × 0.015 m;
- density 5000 kg/m³.

It is a mocap body placed 1 mm above the table, and only a reset moves it.

## License

ShapeNet models can carry model-specific redistribution restrictions. Verify the license
before a public release; this repository asserts no broader license.
