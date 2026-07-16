//! Augur standard library: probability distributions used by the compiler,
//! runtime, and user models.
//!
//! The library is intentionally dependency-light: it provides concrete
//! distributions with exact log-densities and samplers so that the inference
//! engines can remain agnostic to the specific families they operate over.

#![warn(missing_docs)]

pub mod dist;
pub mod special;

pub use dist::{seeded_rng, std_normal, Dist};
