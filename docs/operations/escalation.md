# Escalation Guide

## When to Escalate

Escalate when:

- The on-call runbook doesn't cover the failure mode you're
  seeing.
- The mitigation in a runbook doesn't restore service.
- The root cause is unclear after 30 minutes of triage.
- A data-integrity issue is suspected (e.g. chunks missing, query
  results inconsistent across replicas).
- A security incident is suspected (unauthorised access, leaked
  secrets in logs, anomalous traffic patterns).

## Escalation Path

| Tier | Contact | When |
| --- | --- | --- |
| L1 — On-call SRE | PagerDuty schedule `semrag-oncall` | First responder for any alert |
| L2 — Platform team lead | #semrag-platform Slack channel | L1 is unavailable, or runbook doesn't apply |
| L3 — Service owner | See `CODEOWNERS` | Code-level investigation needed |

## Information to Provide

When escalating, include:

1. **Alert name(s)** — copy/paste the full alert title and
   description from Prometheus / Alertmanager.
2. **Time of incident** — UTC timestamp.
3. **Symptoms observed** — what users are seeing, what the logs
   show, what `/readyz` and `/version` report.
4. **Mitigation attempts** — what runbook steps you tried and
   what happened.
5. **Hypothesis** — your best guess at the root cause, even if
   unconfirmed.

## Communication

- Update the incident channel every 15 minutes, even if there's no
  progress. "Still investigating" is a valid status.
- Use `X-Request-ID` from user reports to correlate to log lines.
- Use `trace_id` from log lines to correlate to OpenTelemetry traces
  (when tracing is enabled).
- Post the resolution and postmortem link in the incident channel
  once the incident is closed.

## After-Hours

For incidents that wake someone up:

- L1 on-call is the primary responder. They follow the runbooks
  and escalate to L2 if needed.
- L2 should be paged only when the on-call SRE has spent more than
  30 minutes without resolving, or when the issue is clearly a
  bug in the service code.
- L3 (service owner) is paged for security incidents, data
  integrity issues, or anything requiring a code change to fix.

## Postmortem

Every incident that triggered a page requires a postmortem within
48 hours. Use the template in
`docs/operations/postmortem-template.md` (TODO: add template).