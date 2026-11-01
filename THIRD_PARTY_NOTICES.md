# Third-party notices

## Clearra

Source: [daejunnom/Clearra](https://github.com/daejunnom/Clearra/tree/5c98563332706a718330bc7ae8e403d2583afe9d), revision `5c98563332706a718330bc7ae8e403d2583afe9d`.

Neyra adapts the following Clearra source:

| Clearra source | Neyra use |
| --- | --- |
| `core-c/src/rules/srs_kicks.c` | Historical reference for the original handwritten tables; replaced by the reviewed official-source fixture and generated Rust module described below |
| `packages/clearra-ui/src/lib/workspace/SolutionBoardPreview.svelte` | Board grid and cell semantics in `apps/neyra-web/src/lib/BoardPreview.svelte`; tetromino palette in `apps/neyra-web/src/lib/buildDocument.ts` |
| `packages/clearra-ui/src/lib/workspace/WorkspaceProgressStatus.svelte` | Progress status and track presentation in `apps/neyra-web/src/lib/ProgressStrip.svelte` |
| `packages/clearra-ui/src/lib/workspace/WorkspaceShell.svelte` | Workspace structure in `apps/neyra-web/src/routes/+page.svelte` and `styles.css` |
| `apps/clearra-web/svelte.config.js` | Static adapter and base-path configuration in `apps/neyra-web/svelte.config.js` |

The piece palette is also checked against revision `f504c8f02d771da518000f901d2b8704a1d6830b`. Neyra uses Clearra’s standalone **ctk3 0.1.2** package for native document encoding, from [`packages/ctk3`](https://github.com/daejunnom/Clearra/tree/f504c8f02d771da518000f901d2b8704a1d6830b/packages/ctk3). That package carries the same Clearra MIT notice below. Its native API preserves Neyra’s 24-row fields.

Original license, reproduced without modification:

```text
MIT License

Copyright (c) 2026 Clearra contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## tetris-fumen

Neyra uses **tetris-fumen 1.1.3**, published by knewjade, to encode Fumen bag diagrams. Source: [knewjade/tetris-fumen](https://github.com/knewjade/tetris-fumen). The package version and integrity are pinned by the lockfile. This package is also a dependency of ctk3.

The installed package’s original license is reproduced without modification:

```text
MIT License

Copyright (c) 2019

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Triangle.js, distributed with Spilink

Source bundle: [daejunnom/Spilink — `src/lib/vendor/engine.js`](https://github.com/daejunnom/Spilink/blob/3f209d1fd7e2961ca4fde5cdb58d5ea6ee11413f/src/lib/vendor/engine.js), revision `3f209d1fd7e2961ca4fde5cdb58d5ea6ee11413f`.

The bundle identifies Triangle.js revision `7837ee5bde8de2719472e0de3458abe6708593f5`. Neyra's `crates/neyra-core/src/physics.rs` adapts or references its SRS piece matrices, spawn placement, and All-mini+ spin behavior. Its I 180° offsets were also an original reference, before all kick data moved to the official-source fixture described below. Embedded source paths include `.cache/triangle/src/engine/utils/kicks/data.ts`, `.cache/triangle/src/engine/utils/kicks/index.ts`, and `.cache/triangle/src/engine/utils/tetromino/data.ts`.

Original bundle notice, reproduced without modification:

```text
/*! Triangle.js 7837ee5bde8de2719472e0de3458abe6708593f5
Copyright (c) 2025 halp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/
```

## Official TETR.IO SRS+ data

Authority: [TETR.IO client bundle](https://tetr.io/js/tetrio.js), retrieved on 2026-10-31, source SHA-256 `9e57c958c43520deb5dd91f5e1773bf779bfb91e7d273dc8494a93a3a0866d2f`.

The reviewed [Clearra fixture](https://github.com/daejunnom/Clearra/blob/f504c8f02d771da518000f901d2b8704a1d6830b/tests/fixtures/rules/tetrio_srs_plus_standard_tetromino_kicks.json), revision `f504c8f02d771da518000f901d2b8704a1d6830b`, is shared as Neyra's `crates/neyra-core/tests/fixtures/tetrio_srs_plus_standard_tetromino_kicks.json`. `scripts/generate-srs-plus.py` generates `crates/neyra-core/src/srs_plus.rs` from its ordered numeric transitions; it neither downloads nor executes the client. The fixture also supplies independent geometry for rotation regression tests.

The complete TETR.IO bundle is not distributed here, and this project's MIT license does not claim to relicense that bundle. The data records 84 standard transitions, the implicit origin attempt, y-axis conversion, source spans and hashes. Clearra's canonical Rust search profile retains 80 transitions by omitting four geometrically redundant O half turns; the official reference retains all 84. Neyra excludes O spins independently of its physical rotation data. The old C I 180° table is not a current authority.

## Portable model hash verification dependencies

Neyra uses RustCrypto SHA-256 to verify the actual float32 model weights before inference. The versions below are pinned in `Cargo.lock`; target and build dependencies are included for attribution. Dual-licensed dependencies are used under their MIT option. The installed packages' MIT notices are reproduced without modification.


### sha2 0.10.9

Source: [https://github.com/RustCrypto/hashes](https://github.com/RustCrypto/hashes).

```text
Copyright (c) 2006-2009 Graydon Hoare
Copyright (c) 2009-2013 Mozilla Foundation
Copyright (c) 2016 Artyom Pavlov

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```


### digest 0.10.7

Source: [https://github.com/RustCrypto/traits](https://github.com/RustCrypto/traits).

```text
Copyright (c) 2017 Artyom Pavlov

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```


### block-buffer 0.10.4

Source: [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils).

```text
Copyright (c) 2018-2019 The RustCrypto Project Developers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```


### crypto-common 0.1.7

Source: [https://github.com/RustCrypto/traits](https://github.com/RustCrypto/traits).

```text
Copyright (c) 2021 RustCrypto Developers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```


### generic-array 0.14.7

Source: [https://github.com/fizyk20/generic-array.git](https://github.com/fizyk20/generic-array.git).

```text
The MIT License (MIT)

Copyright (c) 2015 Bartłomiej Kamiński

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


### typenum 1.20.1

Source: [https://github.com/paholg/typenum](https://github.com/paholg/typenum).

```text
The MIT License (MIT)

Copyright (c) 2014 Paho Lurie-Gregg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


### cpufeatures 0.2.17

Source: [https://github.com/RustCrypto/utils](https://github.com/RustCrypto/utils).

```text
Copyright (c) 2020-2025 The RustCrypto Project Developers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```


### version_check 0.9.5

Source: [https://github.com/SergioBenitez/version_check](https://github.com/SergioBenitez/version_check).

```text
The MIT License (MIT)
Copyright (c) 2017-2018 Sergio Benitez

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
