GRINRAY:
A raytracing framework for optical force simulation
with support for gradient index materials 
=========================================

This Rust crate implements basic raytracing primitives to construct optical simulations with a
particular focus on materials with a gradient index (i.e., materials in which light does not
propagate in a straight line). This software is intended for use in simulations of refractive
microparticles, i.e., particles that move due to the momentum transfer associated with light
refraction.

Installation
============

To use the Rust crate, simply add this repo as a dependency to your own Rust application and let
Cargo handle the rest.
See [here](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html) for dependency
management with Cargo.

This crate also exports Python bindings. We recommend that you use
[Maturin](https://github.com/PyO3/maturin) to build the Python bindings. First, install Maturin
itself (e.g. via `pip install maturin`) and then, in the root of this project, run
```
maturin develop --release
```
to install the bindings in your current Python environment. It might be a good idea to use a
[Python virtual environment](https://docs.python.org/3/library/venv.html) for testing.

Usage
=====

Please refer to the examples and the documentation for guidance on how to use this crate. It is
strongly recommended to only use debug builds for very small calculations, so you might want to
adapt the problem sizes / sample rates in the examples before starting to debug.

The Rust example can be run via `cargo run --release --example forcecalc`.
The Python example can be run via `python3 examples/render.py`.

This is an early release, so feel free to contact us or create an issue if you encounter any problems.

License
=======

This work is licensed under the Mozilla Public License, version 2.0. You can find the full license
in the `LICENSE` file.