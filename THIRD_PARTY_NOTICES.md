# Third-party notices

## Clearra

Source: [daejunnom/Clearra](https://github.com/daejunnom/Clearra/tree/5c98563332706a718330bc7ae8e403d2583afe9d), revision `5c98563332706a718330bc7ae8e403d2583afe9d`.

Neyra adapts the following Clearra source:

| Clearra source | Neyra use |
| --- | --- |
| `core-c/src/rules/srs_kicks.c` | Ordered JLSTZ 90°/180° and I 90° kick tables in `crates/neyra-core/src/physics.rs` |
| `packages/clearra-ui/src/lib/workspace/SolutionBoardPreview.svelte` | Board grid and cell semantics in `apps/neyra-web/src/lib/BoardPreview.svelte`; tetromino colors in `src/routes/+page.svelte` |
| `packages/clearra-ui/src/lib/workspace/WorkspaceProgressStatus.svelte` | Progress status and track presentation in `apps/neyra-web/src/lib/ProgressStrip.svelte` |
| `packages/clearra-ui/src/lib/workspace/WorkspaceShell.svelte` | Workspace structure in `apps/neyra-web/src/routes/+page.svelte` and `styles.css` |
| `apps/clearra-web/svelte.config.js` | Static adapter and base-path configuration in `apps/neyra-web/svelte.config.js` |

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

## Triangle.js, distributed with Spilink

Source bundle: [daejunnom/Spilink — `src/lib/vendor/engine.js`](https://github.com/daejunnom/Spilink/blob/3f209d1fd7e2961ca4fde5cdb58d5ea6ee11413f/src/lib/vendor/engine.js), revision `3f209d1fd7e2961ca4fde5cdb58d5ea6ee11413f`.

The bundle identifies Triangle.js revision `7837ee5bde8de2719472e0de3458abe6708593f5`. Neyra's `crates/neyra-core/src/physics.rs` adapts or references its SRS piece matrices, I 180° offsets, spawn placement, and All-mini+ spin behavior. Embedded source paths include `.cache/triangle/src/engine/utils/kicks/data.ts`, `.cache/triangle/src/engine/utils/kicks/index.ts`, and `.cache/triangle/src/engine/utils/tetromino/data.ts`.

Original bundle notice, reproduced without modification:

```text
/*! Triangle.js 7837ee5bde8de2719472e0de3458abe6708593f5
Copyright (c) 2025 halp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/
```
