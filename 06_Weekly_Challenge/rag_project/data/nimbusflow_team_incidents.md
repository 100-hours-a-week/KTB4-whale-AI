# NimbusFlow Operations Log — Team Incident & Configuration History

This document records the operational history of three teams using NimbusFlow. It was written to trace the relationship between configuration changes and the errors that followed them.

## 1. Team Falcon

Team Falcon operates the payment pipeline and runs NimbusFlow with `engine_mode` set to `hybrid_sync`. The responsible engineer is Mina Park.

In March 2026, Team Falcon changed `checkpoint_interval_sec` from the default value of 90 seconds down to 15 seconds, aiming to reduce recovery time after failures by saving checkpoints more frequently.

After this change, Team Falcon began experiencing repeated **NF-227 errors (Drift Score calculation timeout)**. Analysis showed that setting `checkpoint_interval_sec` too low was the root cause — the increased frequency of checkpoint writes left the Drift Score metrics collector without enough resources to respond in time, causing it to exceed the 4-second limit.

## 2. Team Orion

Team Orion operates the log analytics pipeline and runs NimbusFlow with `engine_mode` set to `solo`. The responsible engineer is Daniel Cho.

Team Orion lowered the `retry_policy` backoff multiplier from the default value of 2.4 down to 1.2, intending to speed up retries. After this change, Team Orion began experiencing **NF-103 errors (checkpoint write failure)**. The cause was traced to retries happening too rapidly after lowering the backoff multiplier, which overwhelmed the checkpoint directory with concurrent disk writes and delayed space allocation.

## 3. Team Atlas

Team Atlas operates the notification delivery pipeline and runs NimbusFlow with `engine_mode` set to `hybrid_sync`. The responsible engineer is Sofia Reyes.

Team Atlas reduced `token_ttl_days` from the default value of 14 days down to 1 day, as part of a stricter security policy. After this change, Team Atlas began experiencing frequent **NF-318 errors (token scope mismatch)**. The cause was that tokens expired too quickly, and the automatic token-renewal script sometimes called the API with a newly issued token before its permission scope had been refreshed.

## 4. Summary Table

| Team        | engine_mode | Engineer    | Configuration Changed                | Error Experienced |
| ----------- | ----------- | ----------- | ------------------------------------ | ----------------- |
| Team Falcon | hybrid_sync | Mina Park   | checkpoint_interval_sec (90s -> 15s) | NF-227            |
| Team Orion  | solo        | Daniel Cho  | retry_policy backoff (2.4 -> 1.2)    | NF-103            |
| Team Atlas  | hybrid_sync | Sofia Reyes | token_ttl_days (14 -> 1 day)         | NF-318            |
