//! The browser's threaded build (wasm32 with atomics) runs every search thread on one shared
//! WebAssembly memory. When one thread grows it (the allocator asking for more), the browser
//! updates the other threads' idea of its size lazily, and their bulk copies and fills
//! (`memory.copy` / `memory.fill`, i.e. every `memcpy` / `memset`) are bounds-checked against
//! that idea: a thread copying into or out of a block that lies in memory another thread just
//! grew trapped with "memory access out of bounds" (Chromium; WebKit had the same bug). That
//! killed Lazy SMP helpers mid-search on the site, and the main thread later aborted
//! ("unreachable") out of memory.
//!
//! `memory.grow(0)` brings the calling thread's view up to date. The allocator does it for
//! every block it hands out that lies past what this thread has seen, or once the memory has
//! grown since it last looked; `sync` does it where a thread picks up another thread's data.
//! Elsewhere all of this compiles to nothing.

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod imp {
    use core::arch::wasm32::{memory_grow, memory_size};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    const PAGE: usize = 65536;

    thread_local! {
        /// Pages of the memory this thread is known to see.
        static SEEN: Cell<usize> = const { Cell::new(0) };
    }

    /// Refresh this thread's view if the memory has grown since it last looked, or if it must
    /// reach `pages` pages (counted in pages: 4 GB is past `usize` in bytes).
    #[inline]
    pub fn sync_pages(pages: usize) {
        SEEN.with(|seen| {
            if pages > seen.get() || memory_size::<0>() > seen.get() {
                memory_grow::<0>(0);
                seen.set(memory_size::<0>());
            }
        });
    }

    /// `System` with the view refreshed before and after each allocation. Zeroing and
    /// reallocating are the trait's defaults (through `alloc`), so their `memset` / `memcpy`
    /// run here, after the refresh, not inside the C allocator before it.
    pub struct Allocator;

    unsafe impl GlobalAlloc for Allocator {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            sync_pages(0);
            // SAFETY: as `GlobalAlloc::alloc`, forwarded.
            let p = unsafe { System.alloc(layout) };
            if !p.is_null() {
                sync_pages((p as usize).saturating_add(layout.size()).div_ceil(PAGE));
            }
            p
        }

        #[inline]
        unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
            // SAFETY: as `GlobalAlloc::dealloc`, forwarded.
            unsafe { System.dealloc(p, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;
}

/// Bring this thread's view of the shared memory up to date (before reading data another thread
/// allocated); a no-op outside the threaded browser build.
#[inline]
pub fn sync() {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    imp::sync_pages(0);
}
