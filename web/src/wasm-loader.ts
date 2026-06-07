interface WasmBindgenExports extends WebAssembly.Exports {
  __wbindgen_export_2?: WebAssembly.Table;
}

interface WasmBindgenBindings {
  imports: WebAssembly.Imports;
  setExports(nextExports: WebAssembly.Exports): void;
}

interface TimerHost {
  performance?: {
    now(): number;
  };
  now?(): number;
}

function wasmBindgenImports(): WasmBindgenBindings {
  let exports: WasmBindgenExports | null = null;
  const global = globalThis as typeof globalThis & { window?: TimerHost };
  const timerHost = {
    performance: global.performance ?? {
      now: () => Date.now()
    }
  };
  const windowLike = global.window ?? timerHost;
  const heap: unknown[] = [undefined, null, true, false];
  let heapNext = heap.length;

  function addHeapObject(value: unknown): number {
    if (heapNext === heap.length) {
      heap.push(heap.length + 1);
    }
    const index = heapNext;
    heapNext = heap[index] as number;
    heap[index] = value;
    return index;
  }

  function getObject(index: number): unknown {
    return heap[index];
  }

  function dropObject(index: number): void {
    if (index < 4) {
      return;
    }
    heap[index] = heapNext;
    heapNext = index;
  }

  const imports: WebAssembly.Imports = {
    __wbindgen_placeholder__: {
      __wbindgen_object_drop_ref: dropObject,
      __wbindgen_describe: () => {},
      __wbindgen_object_clone_ref: (index: number) => addHeapObject(getObject(index)),
      __wbg_instanceof_Window_e093be59ee9a8e14: (index: number) =>
        getObject(index) === windowLike
        || (typeof Window !== "undefined" && getObject(index) instanceof Window),
      __wbg_performance_68499ca0718837f5: (index: number) =>
        addHeapObject((getObject(index) as TimerHost).performance),
      __wbg_now_f565250295e2d180: (index: number) => (getObject(index) as { now(): number }).now(),
      __wbg_static_accessor_GLOBAL_THIS_a1a35cec07001a8a: () => addHeapObject(windowLike),
      __wbg_static_accessor_SELF_4c59f6c7ea29a144: () => addHeapObject(windowLike),
      __wbg_static_accessor_GLOBAL_9d53f2689e622ca1: () => addHeapObject(windowLike),
      __wbg_static_accessor_WINDOW_e70ae9f2eb052253: () => addHeapObject(windowLike),
      __wbg___wbindgen_throw_1506f2235d1bdba0: () => {
        throw new Error("WASM timer initialization failed.");
      },
      __wbg___wbindgen_is_undefined_67b456be8673d3d7: (index: number) => getObject(index) === undefined
    },
    __wbindgen_externref_xform__: {
      __wbindgen_externref_table_grow: (delta: number) => {
        const table = exports?.__wbindgen_export_2;
        if (!table) {
          return -1;
        }
        const index = table.grow(delta);
        table.set(index, undefined);
        return index;
      },
      __wbindgen_externref_table_set_null: (index: number) => {
        exports?.__wbindgen_export_2?.set(index, undefined);
      }
    }
  };

  return {
    imports,
    setExports(nextExports: WebAssembly.Exports) {
      exports = nextExports;
    }
  };
}

export async function instantiateChronofishWasm(path: string): Promise<WebAssembly.Instance> {
  const bindings = wasmBindgenImports();
  const { instance } = await WebAssembly.instantiateStreaming(fetch(path), bindings.imports);
  bindings.setExports(instance.exports);
  return instance;
}
