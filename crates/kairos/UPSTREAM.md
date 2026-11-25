# Kairos upstream

This crate incorporates [Kairos](https://github.com/FirePlank/5d-chess-engine) at
commit `194e4263d7a7dc8ef0584f0ab77de30f054b401d`. The workspace supplies
the build profiles. Kairos retains its engine, 5DUCI executable, tools, model
files, and test fixtures. The shared state and move generator now live in
`chronofish-5d-rules`, and the game formats live in `chronofish-notation`.
Kairos re-exports their original module paths for existing callers.

Each crate carries the upstream NOTICE for attribution. The manifests scope
the Clippy allowances to the imported code, and `kairos/prof` enables profiling
in the shared rules crate.
