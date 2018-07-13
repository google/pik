importScripts('jxrdeclib.js');

function jxrDecode(input, options) {
    var sequence;
    if (input instanceof ArrayBuffer)
        sequence = Promise.resolve(input);
    else if (input instanceof Blob)
        sequence = readBlob(input);
    return sequence.then(function(buffer) {
        FS.writeFile("input.jxr", new Uint8Array(buffer), {
            encoding: "binary"
        });
        return EmscriptenUtility.FileSystem.synchronize(true);
    }).then(function() {
        var arguments = EmscriptenUtility.allocateStringArray(["./this.program", "-v", "-i", "input.jxr", "-o", "output.bmp"]);
        var resultCode = Module.ccall("mainFn", "number", ["number", "number"], [arguments.content.length, arguments.pointer]);

        if (resultCode !== 0)
            throw new Error("Decoding failed: error code " + resultCode);
        EmscriptenUtility.deleteStringArray(arguments);
        FS.unlink("input.jxr");
        return EmscriptenUtility.FileSystem.synchronize(false);
    }).then(function() {
        var result = FS.readFile("output.bmp", {
            encoding: "binary"
        });
        FS.unlink("output.bmp");
        return result;
    });
}
var EmscriptenUtility;
(function(EmscriptenUtility) {
    function allocateString(input) {
        var array = Module.intArrayFromString(input, false);
        var pointer = Module._malloc(array.length);
        Module.HEAP8.set(new Int8Array(array), pointer);
        return pointer;
    }

    function allocateStringArray(input) {
        var array = [];
        input.forEach(function(item) {
            return array.push(allocateString(item));
        });
        var pointer = Module._calloc(array.length, 4);
        Module.HEAP32.set(new Uint32Array(array), pointer / 4);
        return {
            content: array,
            pointer: pointer
        };
    }
    EmscriptenUtility.allocateStringArray = allocateStringArray;

    function deleteStringArray(input) {
        input.content.forEach(function(item) {
            return Module._free(item);
        });
        Module._free(input.pointer);
    }
    EmscriptenUtility.deleteStringArray = deleteStringArray;
})(EmscriptenUtility || (EmscriptenUtility = {}));
var EmscriptenUtility;
(function(EmscriptenUtility) {
    var FileSystem;
    (function(FileSystem) {
        function writeBlob(path, blob) {
            return new Promise(function(resolve, reject) {
                var reader = new FileReader();
                reader.onload = function() {
                    return resolve(reader.result);
                };
                reader.readAsArrayBuffer(blob);
            }).then(function(result) {
                FS.writeFile(path, new Uint8Array(result), {
                    encoding: "binary"
                });
            });
        }
        FileSystem.writeBlob = writeBlob;

        function synchronize(populate) {
            return new Promise(function(resolve, reject) {
                FS.syncfs(populate, function() {
                    return resolve();
                });
            });
        }
        FileSystem.synchronize = synchronize;
    })(FileSystem = EmscriptenUtility.FileSystem || (EmscriptenUtility.FileSystem = {}));
})(EmscriptenUtility || (EmscriptenUtility = {}));

self.onmessage = function(event) {
    var jxr = event.data;

    jxrDecode(jxr).then(function(arr) { self.postMessage(arr, [arr.buffer]); });
}