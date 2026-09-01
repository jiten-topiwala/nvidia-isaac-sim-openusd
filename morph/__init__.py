"""morph — the MORPH-I Isaac Sim application package (was one 5,900-line play_isaac.py).

IMPORT ORDER MATTERS. Several modules here import `isaacsim` / `pxr` / `omni`, whose APIs come from
runtime-loaded plugins and do not exist until `SimulationApp(...)` has been constructed. The entry
script `play_isaac.py` builds the app first and only then imports this package, so anything in here
may use Isaac APIs at module level. Do not import `morph` before that point.

Two modules are exceptions and are safe to import at any time — `config` (pure data) and `geometry`
(pure math). Both are dependency-free by design; keep them that way.
"""
