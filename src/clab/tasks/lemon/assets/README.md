# Lemon task assets

The lemon and plate meshes come from robosuite's `objects/lemon/` and `objects/plate/`.
robosuite is not a runtime dependency. Each packaged mesh is the source mesh scaled by its
source constructor's factor, so the runtime scale is one and body origins are unchanged. The
materials are flat colours, so no texture images are included.

| Property | Lemon | Plate |
| --- | --- | --- |
| Baked scale | `(0.75, 1, 1)` | `(0.5, 0.5, 0.5)` |
| Visual size (mm) | 39.7 × 59.8 × 58.5 | 129.9 × 130.0 × 13.2 |
| Collision | 2 convex STL pieces | 5 convex STL pieces (keeps the plate concave) |
| Mass (kg) | 0.080 (source 0.029, changed on purpose) | 0.279 (mocap body) |
| Diagonal inertia (kg m²) | 0.001 each | 0.001 each |
| Friction | 2.0, 0.5, 0.1 | 0.95, 0.3, 0.1 |

The authored inertias are kept as they are, not recomputed from the scaled mesh, and they
are not measured values. Visual geoms are massless and non-colliding (group 1), and
collision geoms are in group 3. The lemon has a damped free joint. The plate is a mocap
body, so a reset may move it but episode physics cannot. `PLATE_CENTRE_FLOOR_Z` is the
collision-surface height at the plate centre, measured by a ray; success uses actual
lemon–plate contact rather than this height.

## License

robosuite's MIT notice is reproduced in `ROBOSUITE_LICENSE`. No separate per-mesh license
was found, so verify redistribution rights before a public release.
