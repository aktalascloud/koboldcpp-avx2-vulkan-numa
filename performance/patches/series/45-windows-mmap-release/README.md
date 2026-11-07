# Release copied Windows weight pages from the working set

Keeps the file mapping and its bytes, releases complete pages in ranges the loader already marks unused, and clamps and aligns ranges without integer wrap. Working-set evidence only; no model speed claim.

[Source](https://github.com/stew675/llama-cpp-rdna-boosts/pull/269)
