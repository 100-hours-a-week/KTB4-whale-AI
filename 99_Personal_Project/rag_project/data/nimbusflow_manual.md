# NimbusFlow Data Pipeline Engine — User Manual

## 1. Product Overview

NimbusFlow is a lightweight data pipeline orchestration engine designed for small to mid-sized data teams. It allows users to define, schedule, and monitor data transformation jobs through a single configuration file called a `flowspec`.

NimbusFlow was first released as version 1.0 in 2023 and is currently maintained at version 4.2. The core engine is written in Go, while the client SDK is available in Python and JavaScript.

The product's internal codename during development was "Project Driftwood."

## 2. Installation

To install the NimbusFlow CLI, run:

```
curl -sSL https://install.nimbusflow.dev/get.sh | bash
```

After installation, verify the version with:

```
nimbusflow --version
```

The default installation directory is `/opt/nimbusflow`, and the default configuration file is located at `/opt/nimbusflow/config/flowspec.yaml`.

## 3. Configuration

NimbusFlow uses a YAML-based configuration file. The three most important top-level keys are:

- `engine_mode`: Accepts one of three values — `solo`, `cluster`, or `hybrid_sync`. The default value is `solo`.
- `retry_policy`: Controls how failed jobs are retried. The default retry count is 5, and the default backoff multiplier is 2.4.
- `checkpoint_interval_sec`: Defines how often the engine saves pipeline state, in seconds. The default value is 90 seconds.

A unique feature of NimbusFlow is the `hybrid_sync` engine mode, which allows the pipeline to alternate between local execution and cloud execution depending on a metric called the **Drift Score**. If the Drift Score exceeds 0.73, NimbusFlow automatically switches execution to the cloud worker pool.

## 4. API Usage

NimbusFlow exposes a REST API on port `8842` by default.

### 4.1 Authentication

All API requests require a header named `X-Nimbus-Token`. Tokens are generated using the CLI command:

```
nimbusflow token create --scope pipeline:read,pipeline:write
```

Tokens expire after 14 days by default, but this can be changed via the `token_ttl_days` configuration key.

### 4.2 Triggering a Pipeline Run

To trigger a pipeline run via the API, send a POST request to:

```
POST /v2/pipelines/{pipeline_id}/runs
```

The request body must include a field called `run_mode`, which accepts `standard`, `dry_run`, or `shadow_replay`. The `shadow_replay` mode re-executes a previous run using cached input data, and is primarily used for regression testing.

### 4.3 Checking Run Status

```
GET /v2/pipelines/{pipeline_id}/runs/{run_id}
```

The response includes a field called `phase_marker`, which can be one of: `queued`, `loading`, `transforming`, `committing`, or `archived`.

## 5. Error Codes

NimbusFlow uses a custom error code system prefixed with `NF-`.

- **NF-103**: Checkpoint write failure. Usually caused by insufficient disk space in the checkpoint directory.
- **NF-227**: Drift Score calculation timeout. Occurs when the metrics collector does not respond within 4 seconds.
- **NF-318**: Token scope mismatch. The provided `X-Nimbus-Token` does not have permission for the requested operation.
- **NF-409**: Duplicate `pipeline_id` detected during registration.
- **NF-512**: Hybrid sync conflict. Occurs when both local and cloud workers attempt to commit the same checkpoint simultaneously.

## 6. Frequently Asked Questions

**Q: Can I run NimbusFlow without an internet connection?**
A: Yes, as long as `engine_mode` is set to `solo`. The `cluster` and `hybrid_sync` modes require network access.

**Q: What is the maximum number of pipelines per instance?**
A: The community edition supports up to 50 concurrent pipelines. The enterprise edition supports up to 5,000.

**Q: Who maintains NimbusFlow?**
A: NimbusFlow is maintained by a fictional company called Driftwood Systems, founded in 2022.
