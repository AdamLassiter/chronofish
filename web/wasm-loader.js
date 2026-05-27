function wasmBindgenImports() {
  let exports = null;
  const global = globalThis;
  const timerHost = {
    performance: global.performance ?? {
      now: () => Date.now()
    }
  };
  const windowLike = global.window ?? timerHost;
  const heap = [undefined, null, true, false];
  let heapNext = heap.length;

  function addHeapObject(value) {
    if (heapNext === heap.length) {
      heap.push(heap.length + 1);
    }
    const index = heapNext;
    heapNext = heap[index];
    heap[index] = value;
    return index;
  }

  function getObject(index) {
    return heap[index];
  }

  function dropObject(index) {
    if (index < 4) {
      return;
    }
    heap[index] = heapNext;
    heapNext = index;
  }

  const imports = {
    __wbindgen_placeholder__: {
      __wbindgen_object_drop_ref: dropObject,
      __wbindgen_describe: () => {},
      __wbindgen_object_clone_ref: (index) => addHeapObject(getObject(index)),
      __wbg_instanceof_Window_e093be59ee9a8e14: (index) =>
        getObject(index) === windowLike
        || (typeof Window !== "undefined" && getObject(index) instanceof Window),
      __wbg_performance_68499ca0718837f5: (index) =>
        addHeapObject(getObject(index).performance),
      __wbg_now_f565250295e2d180: (index) => getObject(index).now(),
      __wbg_static_accessor_GLOBAL_THIS_a1a35cec07001a8a: () => addHeapObject(windowLike),
      __wbg_static_accessor_SELF_4c59f6c7ea29a144: () => addHeapObject(windowLike),
      __wbg_static_accessor_GLOBAL_9d53f2689e622ca1: () => addHeapObject(windowLike),
      __wbg_static_accessor_WINDOW_e70ae9f2eb052253: () => addHeapObject(windowLike),
      __wbg___wbindgen_throw_1506f2235d1bdba0: () => {
        throw new Error("WASM timer initialization failed.");
      },
      __wbg___wbindgen_is_undefined_67b456be8673d3d7: (index) => getObject(index) === undefined
    },
    __wbindgen_externref_xform__: {
      __wbindgen_externref_table_grow: (delta) => {
        const table = exports?.__wbindgen_export_2;
        if (!table) {
          return -1;
        }
        const index = table.grow(delta);
        table.set(index, undefined);
        return index;
      },
      __wbindgen_externref_table_set_null: (index) => {
        exports?.__wbindgen_export_2?.set(index, undefined);
      }
    }
  };

  return {
    imports,
    setExports(nextExports) {
      exports = nextExports;
    }
  };
}

export async function instantiateChronofishWasm(path) {
  const bindings = wasmBindgenImports();
  const { instance } = await WebAssembly.instantiateStreaming(fetch(path), bindings.imports);
  bindings.setExports(instance.exports);
  return instance;
}
