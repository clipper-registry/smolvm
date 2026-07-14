//! CUDA Driver-API remoting for smolvm: a guest microVM forwards `cu*` calls
//! over vsock to a host server that runs them on the host's NVIDIA GPU.
//!
//! - `proto` — the wire protocol (framing, request/response codec). No deps.
//! - `client` — guest-side marshalling over any `Read`/`Write` stream.
//! - `host` — host-side dispatch with a `Backend` trait; ships a real GPU
//!   backend (`GpuBackend`, `gpu` feature) and a CPU emulation backend
//!   (`CpuBackend`) for GPU-less verification.
//!
//! The guest binary depends on this crate with `default-features = false` to
//! pull only `proto` + `client` (no `libloading`), keeping the musl build lean.

pub mod client;
pub mod proto;
/// Shared-memory command/completion rings (low-latency in-VM transport).
pub mod ring;

/// Fingerprint of the wire-defining source. The client sends it in the
/// `Init` handshake; the host rejects a mismatch, turning a stale shim/server
/// pairing into a loud error instead of silent data corruption.
///
/// FNV-1a over the files both the client (shim) and host (server) compile,
/// i.e. the wire contract. Computed via `include_bytes!` at compile time so
/// the inputs are ordinary tracked deps and no build script re-run can
/// cascade rebuilds.
pub const PROTO_HASH: u64 = {
    // A manual epoch to force a bump on wire changes the file set misses.
    let mut h = fnv1a(0xcbf2_9ce4_8422_2325, b"epoch-1");
    h = fnv1a(h, include_bytes!("proto.rs"));
    h = fnv1a(h, include_bytes!("client.rs"));
    h = fnv1a(h, include_bytes!("ring.rs"));
    h = fnv1a(h, include_bytes!("generated/cublas_guest.rs"));
    h = fnv1a(h, include_bytes!("generated/cudnn_guest.rs"));
    h
};

const fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    let mut i = 0;
    while i < bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    h
}

/// Shared-memory bulk-data channel (zero-copy memcpy). Linux-only.
#[cfg(target_os = "linux")]
pub mod shm;

#[cfg(feature = "host")]
pub mod host;
