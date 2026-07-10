# Buili product specification implementation matrix

This matrix traces `Buili_Product_Web_Specification_KR.pdf` v1.0 (2026-07-10) to implementation and acceptance evidence. It deliberately distinguishes a visible demo from a server-enforced workflow. A row is **Complete** only when the critical state is persisted and covered by an automated test.

Status legend: **Complete** = implemented and exercised; **Partial** = a usable slice exists but one or more required guarantees are absent; **Open** = no production-capable implementation was found during the audit.

| Spec area | Priority | Status | Implementation / evidence | Remaining production criterion |
| --- | --- | --- | --- | --- |
| English app shell and project navigation | P0 | Complete | Next.js shell; desktop and mobile navigation; Playwright navigation checks | Add authenticated organization context before external use |
| Project creation and upload intake | P0 | Complete | Seven-step accessible wizard, project/profile/settings/directory APIs, classified multi-file intake, size/type validation, original SHA-256 and upload audit | External email delivery still requires a configured provider |
| Current/superseded drawing revisions | P0 | Complete | Additive revision registry, explicit activation, current/superseded history, stale-issue routing and `REVISION_ACTIVATED` before/after audit; acceptance-tested | Multi-party revision approval can be added as a customer policy |
| Source-cited requirements and search | P0 | Complete | Page/bbox citations, current-revision-only RAG, universal scoped search, frozen issue/report source snapshots and unresolved freshness state | Customer document terminology still requires pilot calibration |
| 2D spatial compile and source jump | P0 | Partial | Plan graph, overlay, page coordinates, GLB/spatial routes and correction-oriented UI exist | Every spatial object needs a tested round trip to original page coordinates; correction persistence needs role/audit enforcement |
| Lightweight 3D spatial index | P1 | Partial | Room/wall/opening assets and issue-level spatial evidence are available | Pilot accuracy and correction QA must be measured on the spec test matrix; this is not precision BIM |
| Field evidence intake | P0 | Complete | Photo/video/voice/measurement capture, author/time/location method, unlinked status, original hash, evidence update/link and issue-draft creation are persisted | Device-native QR scanning can be added when pilot hardware is known |
| Offline mobile capture | P1 | Complete | IndexedDB preserves original blobs, explicit retry state, SHA-verified idempotent sync, conflict rejection and offline-to-online Playwright coverage | Background sync may be enabled only after browser support and field policy validation |
| Evidence sufficiency | P1 | Complete | Structured location/context/detail/measurement/source prompts and editable evidence gaps feed the human review gate | Model-generated image-quality scoring remains advisory |
| Issue object | P0 | Complete | Validated create/detail/update APIs, source/evidence relations, structured gaps, route, impact, version and history-backed UI | Organization-specific numbering templates remain configurable follow-up work |
| PM review gate | P0 | Complete | Role-gated approve/reject/request-evidence API, structured reasons, versioned snapshots, notifications and audit; direct official PATCH/export is blocked | OIDC-backed identities are required before external deployment |
| Versioned reports and export | P0/P1 | Complete | Draft and immutable issued versions include reviewer, source/issue snapshots, SHA-256, issue IDs, version metadata and retained history | External delivery providers are integration-specific |
| Audit and chain of custody | P0/P1 | Complete | Critical actions persist actor, project, timestamp, before/after and metadata; project/entity query APIs and traversal tests are present | WORM storage/retention is an infrastructure policy |
| RBAC / tenant isolation | P0 | Partial | Project-scoped lookups and role-gated mutations are enforced; production can fail closed on trusted actor/role headers | Deploy behind OIDC/SAML proxy and enable `BUILI_REQUIRE_AUTH_HEADERS=true` |
| Universal search, notifications and directory | P1 | Complete | Current-project/historical search scopes, persisted in-app notifications, directory invitations/access status and settings forms are implemented | Email/push delivery requires configured providers |
| Integrations and external sync | P1/Later | Open | Configuration placeholders only | Idempotent jobs, external ID/history and failure/retry notifications are required |
| Accessibility | P0 | Partial | English document language, semantic controls and responsive layout exist | Full keyboard/focus/contrast/status-text audit is still required |
| Security operations | P0 | Partial | Upload limits, filename normalization, configurable CORS/storage/database | OIDC/SAML, encryption policy verification, secret manager, retention/legal hold, soft deletion and restore drills remain open |

## Release-blocking acceptance contracts

The automated suites should keep these invariants stable:

1. A client cannot set an issue to `issued`, create an official export, or assign contractual responsibility without an authorized persisted review.
2. Approve, reject, and request-evidence decisions preserve reviewer, reason, issue version, timestamp, and an audit event with before/after state.
3. Activating a revision never overwrites the prior source. It marks affected open issues stale and snapshots the exact source versions used by an issue/report.
4. Replaying an offline capture with the same client capture ID is idempotent. Reusing that ID with different bytes is rejected, and the stored SHA-256 matches the original.
5. Issued report versions and checksums cannot be overwritten. A later edit produces a new version while history remains downloadable.
6. Project and organization scope is enforced on every object lookup, not only on list routes.

## Audit conclusion

The evidence workflow release gates are now server-enforced and covered at the HTTP and browser boundaries. The remaining external-deployment boundary is verified enterprise identity and managed infrastructure: enable fail-closed trusted headers only behind OIDC/SAML, then use PostgreSQL, private versioned object storage, backups, retention controls and centralized logs before storing contractual or dispute-sensitive records.
