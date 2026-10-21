# Deepkit Code documentation

[Back to the project](../README.md)

Start with [setup and usage](usage.md): build the Mac/Linux beta, try local
search without a model, and connect your coding assistant over MCP.

| What you need | Guide |
| --- | --- |
| Install, search, configure models or connect MCP | [Setup and usage](usage.md) |
| Supported beta platforms, installation status and release checks | [Beta status](beta-release.md) |
| History coverage, repaired defects and remaining boundaries | [Beta audit](beta-audit.md) |
| Measured search quality, agent costs and known misses | [Benchmarks](benchmarks.md) |
| Reproduce benchmark runs | [Benchmark tools](../scripts/bench/README.md) |
| Inspect query fixtures and expected files | [Search questions](../bench/goldens/) |
| Inspect retained measurements | [Benchmark results](../bench/results/beta-20261014/) |
| Understand third-party licensing | [Dependency notices](../THIRD_PARTY_NOTICES.md) |

DKC stores its index in `.deepkit/code.db`. Models are optional and downloaded
separately. A local profile keeps embedding and reranking on your machine;
remote profiles send the corresponding work to their configured providers.
Read the [privacy and configuration guide](usage.md#privacy-and-configuration)
before enabling remote work.

The beta is still being checked on real repositories. The benchmark guide
separates lexical, vector, graph and reranking results, and records failures
alongside gains. It should help you judge what has been demonstrated for your
own use case.
