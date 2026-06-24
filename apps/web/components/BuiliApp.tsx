"use client";

import {
  AlertTriangle,
  Archive,
  Check,
  ClipboardCheck,
  FileDown,
  FileQuestion,
  FolderPlus,
  Gauge,
  ListChecks,
  Loader2,
  Map,
  MessageSquarePlus,
  MoreHorizontal,
  RefreshCcw,
  Search,
  ShieldCheck,
  Upload,
  X
} from "lucide-react";
import { ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  api,
  DocumentAsset,
  Issue,
  Job,
  Observation,
  Overlay,
  Project,
  SiteMediaAsset,
  TechnologyStatus
} from "@/lib/api";
import { InstallPrompt } from "@/components/InstallPrompt";

type View = "review" | "evidence" | "overlay" | "reports" | "pipeline";
type IssueFilter = "all" | "blocker" | "electrical" | "mechanical";
const PLAN_IMAGE_SRC = "/plans/utah-e11-electrical-plan.jpg";
const PLAN_LABEL = "E1.1 Electrical Plans";
const FIELD_IMAGE_SRC = "/site-media/construction-site-electrical-work.jpg";
const DEFAULT_RAG_QUERY = "AFCI GFCI smoke detector outlet electrical plan";
const DEMO_PROJECT_NAME = "Cooper Residence E1.1";

const views: Array<{ id: View; label: string; icon: React.ComponentType<{ size?: number }> }> = [
  { id: "review", label: "Issues", icon: ListChecks },
  { id: "evidence", label: "Evidence", icon: ShieldCheck },
  { id: "overlay", label: "Plan", icon: Map },
  { id: "reports", label: "Reports", icon: FileDown },
  { id: "pipeline", label: "Pipeline", icon: Gauge }
];

function selectInitialProject(incoming: Project[]) {
  return (
    incoming.find((item) => item.name === DEMO_PROJECT_NAME) ??
    incoming.find((item) => item.name.toLowerCase().includes("cooper")) ??
    incoming[0] ??
    null
  );
}

const severityRank: Record<string, number> = {
  blocker: 4,
  major: 3,
  minor: 2,
  informational: 1
};

export function BuiliApp() {
  const [view, setView] = useState<View>("review");
  const [issueFilter, setIssueFilter] = useState<IssueFilter>("all");
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [issues, setIssues] = useState<Issue[]>([]);
  const [documents, setDocuments] = useState<DocumentAsset[]>([]);
  const [mediaAssets, setMediaAssets] = useState<SiteMediaAsset[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [technologyStatus, setTechnologyStatus] = useState<TechnologyStatus[]>([]);
  const [selectedIssueId, setSelectedIssueId] = useState<string>("");
  const [job, setJob] = useState<Job | null>(null);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [query, setQuery] = useState(DEFAULT_RAG_QUERY);
  const [ragResults, setRagResults] = useState<Array<Record<string, unknown>>>([]);
  const [rfi, setRfi] = useState("");
  const [reportUrl, setReportUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [apiError, setApiError] = useState("");
  const [notice, setNotice] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedIssue = useMemo(
    () => issues.find((issue) => issue.issue_id === selectedIssueId) ?? issues[0],
    [issues, selectedIssueId]
  );

  const actionNeeded = useMemo(
    () => issues.filter((issue) => issue.status === "review_ready" && issue.confidence >= 0.55).length,
    [issues]
  );

  const orderedIssues = useMemo(
    () =>
      [...issues].sort(
        (a, b) =>
          (severityRank[b.severity] ?? 0) - (severityRank[a.severity] ?? 0) ||
          b.confidence - a.confidence
      ),
    [issues]
  );

  const filteredIssues = useMemo(
    () =>
      orderedIssues.filter((issue) => {
        if (issueFilter === "all") return true;
        if (issueFilter === "blocker") return issue.severity === "blocker";
        return issue.discipline === issueFilter;
      }),
    [issueFilter, orderedIssues]
  );

  const issueCounts = useMemo(
    () => ({
      all: issues.length,
      blocker: issues.filter((issue) => issue.severity === "blocker").length,
      electrical: issues.filter((issue) => issue.discipline === "electrical").length,
      mechanical: issues.filter((issue) => issue.discipline === "mechanical").length
    }),
    [issues]
  );

  const groupedIssues = useMemo(() => {
    const labels: Array<{ key: string; title: string }> = [
      { key: "blocker", title: "Blocker - Needs Review" },
      { key: "major", title: "Major - Awaiting Verification" },
      { key: "minor", title: "Minor - Informational" },
      { key: "informational", title: "Informational" }
    ];
    return labels
      .map((group) => ({
        ...group,
        items: filteredIssues.filter((issue) => issue.severity === group.key)
      }))
      .filter((group) => group.items.length);
  }, [filteredIssues]);

  const loadProjectData = useCallback(async (projectId: string) => {
    const [
      incomingIssues,
      incomingOverlay,
      incomingJob,
      incomingDocuments,
      incomingMedia,
      incomingObservations,
      incomingTechnology,
      incomingRag
    ] = await Promise.all([
      api.listIssues(projectId),
      api.getOverlay(projectId),
      api.latestJob(projectId),
      api.listDocuments(projectId),
      api.listMedia(projectId),
      api.listObservations(projectId),
      api.technologyStatus(projectId),
      api.ragSearch(projectId, DEFAULT_RAG_QUERY)
    ]);
    const nextIssues = incomingIssues;
    setIssues(nextIssues);
    setDocuments(incomingDocuments);
    setMediaAssets(incomingMedia);
    setObservations(incomingObservations);
    setTechnologyStatus(incomingTechnology);
    setRagResults(incomingRag.returned_context);
    setJob(incomingJob);
    setSelectedIssueId((current) =>
      nextIssues.some((issue) => issue.issue_id === current) ? current : nextIssues[0]?.issue_id || ""
    );
    setOverlay(incomingOverlay);
  }, []);

  useEffect(() => {
    if (!selectedIssue) {
      setRfi("");
      return;
    }
    setRfi(buildRfiPreview(selectedIssue));
  }, [selectedIssue]);

  const surfaceError = useCallback((error: unknown, fallback: string) => {
    const message = error instanceof Error ? error.message : fallback;
    setApiError(message);
    setNotice("");
  }, []);

  useEffect(() => {
    api
      .listProjects()
      .then(async (incoming) => {
        setProjects(incoming);
        const initial = selectInitialProject(incoming);
        setProject(initial);
        if (initial) await loadProjectData(initial.project_id);
      })
      .catch((error: Error) => {
        setApiError(`API offline: ${error.message}`);
        setIssues([]);
      });
  }, [loadProjectData]);

  useEffect(() => {
    if (!job || job.state === "review_ready" || job.state === "failed") return;
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.getJob(job.job_id);
        setJob(nextJob);
        if (nextJob.state === "review_ready" && project) {
          await loadProjectData(project.project_id);
      setNotice("Review complete. Issue candidates were refreshed.");
        }
      } catch (error) {
        surfaceError(error, "Failed to poll job");
      }
    }, 900);
    return () => window.clearInterval(timer);
  }, [job, loadProjectData, project]);

  async function createProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const name = String(form.get("name") || "").trim();
    if (!name) return;
    setBusy(true);
    setApiError("");
    try {
      const created = await api.createProject({
        name,
        address: String(form.get("address") || ""),
        project_type: "tenant_improvement"
      });
      setProjects((current) => [created, ...current]);
      setProject(created);
      setIssues([]);
      setOverlay(null);
      setSelectedIssueId("");
      setNotice(`Project created: ${created.name}`);
      event.currentTarget.reset();
    } catch (error) {
      surfaceError(error, "Project creation failed");
    } finally {
      setBusy(false);
    }
  }

  async function runAnalysis() {
    if (!project) return;
    setBusy(true);
    setApiError("");
    try {
      const createdJob = await api.analyzeProject(project.project_id);
      setJob(createdJob);
      setNotice("Review run queued. Pipeline progress will update automatically.");
    } catch (error) {
      surfaceError(error, "Analyze failed");
    } finally {
      setBusy(false);
    }
  }

  async function patchIssue(issueId: string, status: string) {
    try {
      const updated = await api.updateIssue(issueId, { status });
      setIssues((current) => current.map((issue) => (issue.issue_id === issueId ? updated : issue)));
      setNotice(`Issue marked as ${status.replaceAll("_", " ")}.`);
    } catch (error) {
      surfaceError(error, "Issue update failed");
    }
  }

  async function generateRfi(issueId: string) {
    try {
      const draft = await api.createRfi(issueId);
      setRfi(draft.markdown);
      setNotice("RFI draft generated.");
      setView("reports");
    } catch (error) {
      surfaceError(error, "RFI generation failed");
    }
  }

  async function generateReport(type: "punch" | "co_evidence", format: "pdf" | "csv") {
    if (!project) return;
    try {
      const report = await api.createReport(project.project_id, type, format);
      setReportUrl(report.download_url);
      setNotice(`${type.replace("_", " ")} ${format.toUpperCase()} report generated.`);
    } catch (error) {
      surfaceError(error, "Report generation failed");
    }
  }

  async function searchRag() {
    if (!project) return;
    try {
      const result = await api.ragSearch(project.project_id, query);
      setRagResults(result.returned_context);
      setNotice(`Found ${result.returned_context.length} citation candidates.`);
    } catch (error) {
      surfaceError(error, "RAG search failed");
    }
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file || !project) return;

    const kind =
      file.type.startsWith("image/") || file.type.startsWith("video/") || file.type.startsWith("audio/")
        ? "media"
        : "document";
    setIsUploading(true);
    setBusy(true);
    setApiError("");
    setNotice(`Uploading ${file.name}...`);
    try {
      const presigned = await api.presignUpload({
        project_id: project.project_id,
        filename: file.name,
        mime: file.type || "application/octet-stream",
        size: file.size,
        kind
      });
      await api.uploadFile(presigned.upload_url, file);
      await api.completeUpload(presigned.complete_url, {
        document_type: kind === "media" ? "media" : "plan",
        revision: "A"
      });
      setNotice(`${file.name} uploaded. Run review to refresh issue candidates.`);
      await loadProjectData(project.project_id);
    } catch (error) {
      surfaceError(error, "Upload failed");
    } finally {
      setIsUploading(false);
      setBusy(false);
      event.target.value = "";
    }
  }

  return (
    <main className="app-shell">
      <aside className="side-nav" aria-label="Primary">
        <button className="brand-mark" type="button" aria-label="Go to Issues" onClick={() => setView("review")}>
          <img src="/buili_favicon_transparent.png" alt="" />
          <strong>Buili</strong>
        </button>
        <nav>
          {views.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "nav-item active" : "nav-item"}
              type="button"
              onClick={() => setView(item.id)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <section className="workbench">
        <header className="top-bar">
          <div>
            <p className="eyebrow">Field to report</p>
            <button className="top-brand" type="button" onClick={() => setView("review")}>
              Buili
            </button>
            <p className="top-subtitle">Issue inbox, evidence chain, RFI, punch list, and CO support.</p>
          </div>
          <div className="top-actions">
            <InstallPrompt onNotice={setNotice} />
            <button className="icon-button primary" type="button" onClick={runAnalysis} disabled={!project || busy}>
              {busy ? <Loader2 className="spin" size={18} /> : <RefreshCcw size={18} />}
              <span>Run review</span>
            </button>
          </div>
        </header>

        <section className="project-strip">
          <div className="project-picker">
            <label htmlFor="project">Project</label>
            <select
              id="project"
              value={project?.project_id ?? ""}
              onChange={async (event) => {
                const next = projects.find((item) => item.project_id === event.target.value) ?? null;
                setProject(next);
                if (next) await loadProjectData(next.project_id);
              }}
            >
              {projects.map((item) => (
                <option key={item.project_id} value={item.project_id}>
                  {item.name}
                </option>
              ))}
            </select>
          </div>
          <form className="new-project" onSubmit={createProject}>
            <FolderPlus size={18} />
            <input name="name" placeholder="New project name" aria-label="New project name" />
            <input name="address" placeholder="Address" aria-label="Address" />
            <button type="submit" disabled={busy}>
              Add
            </button>
          </form>
          <input
            ref={fileInputRef}
            className="file-input"
            type="file"
            accept=".pdf,.docx,.txt,.csv,.xlsx,image/*,video/*,audio/*"
            onChange={handleUpload}
          />
          <button
            className="icon-button ghost"
            type="button"
            title="Upload"
            disabled={!project || isUploading}
            onClick={() => fileInputRef.current?.click()}
          >
            {isUploading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
            <span>{isUploading ? "Uploading" : "Upload"}</span>
          </button>
        </section>

        {apiError ? <p className="api-error">{apiError}</p> : null}
        {notice ? <p className="status-note">{notice}</p> : null}

        <section className="kpi-band" aria-label="Project metrics">
          <Metric label="Issues ready" value={issues.filter((issue) => issue.status === "review_ready").length} />
          <Metric label="Action needed" value={actionNeeded} />
          <Metric label="Evidence score" value={`${Math.round(avg(issues.map((issue) => issue.confidence)) * 100)}%`} />
          <Metric label="Job" value={job ? `${job.progress}%` : "idle"} />
        </section>

        {view === "review" && (
          <section className="content-grid review-grid">
            <div className="issue-list" aria-label="Issue inbox">
              <div className="section-title-row">
                <h2>Issue inbox</h2>
                <span>{filteredIssues.length} shown</span>
              </div>
              <div className="filter-row" aria-label="Issue filters">
                <FilterChip active={issueFilter === "all"} onClick={() => setIssueFilter("all")}>
                  All {issueCounts.all}
                </FilterChip>
                <FilterChip active={issueFilter === "blocker"} onClick={() => setIssueFilter("blocker")}>
                  Blocker {issueCounts.blocker}
                </FilterChip>
                <FilterChip active={issueFilter === "electrical"} onClick={() => setIssueFilter("electrical")}>
                  Elec {issueCounts.electrical}
                </FilterChip>
                <FilterChip active={issueFilter === "mechanical"} onClick={() => setIssueFilter("mechanical")}>
                  Mech {issueCounts.mechanical}
                </FilterChip>
              </div>
              {groupedIssues.map((group) => (
                <div className="issue-section" key={group.key}>
                  <p className="section-label">{group.title}</p>
                  {group.items.map((issue) => (
                    <button
                      key={issue.issue_id}
                      className={selectedIssue?.issue_id === issue.issue_id ? "issue-row selected" : "issue-row"}
                      type="button"
                      onClick={() => setSelectedIssueId(issue.issue_id)}
                    >
                      <span className={`severity-bar ${issue.severity}`} />
                      <span className="issue-row-main">
                        <span className="issue-type-line">
                          {issue.type.replaceAll("_", " ")} · {issue.discipline}
                        </span>
                        <strong>{issue.title}</strong>
                        <small>
                          {issue.room} · {String(issue.requirement.source ?? "source pending")}
                        </small>
                      </span>
                      <span className="confidence">{Math.round(issue.confidence * 100)}%</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
            <IssueInspector
              issue={selectedIssue}
              onApprove={(id) => patchIssue(id, "approved")}
              onReject={(id) => patchIssue(id, "rejected_false_positive")}
              onNeedMore={(id) => patchIssue(id, "needs_more_evidence")}
              onRfi={generateRfi}
            />
            <FieldSnapshot issue={selectedIssue} overlay={overlay} />
          </section>
        )}

        {view === "evidence" && (
          <section className="content-grid evidence-grid">
            <EvidenceViewer
              issue={selectedIssue}
              documents={documents}
              mediaAssets={mediaAssets}
              observations={observations}
            />
            <div className="rag-panel">
              <div className="section-title-row">
                <h2>Citations</h2>
                <button className="icon-only" type="button" onClick={searchRag} title="Search">
                  <Search size={18} />
                </button>
              </div>
              <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
              <div className="rag-results">
                {ragResults.map((result, index) => (
                  <article key={String(result.chunk_id ?? index)}>
                    <strong>{String(result.chunk_id ?? "chunk")}</strong>
                    <p>{String(result.text ?? "").slice(0, 260)}</p>
                    <small>score {String(result.score ?? "n/a")}</small>
                  </article>
                ))}
              </div>
            </div>
          </section>
        )}

        {view === "overlay" && <OverlayView overlay={overlay} />}

        {view === "reports" && (
          <section className="reports-panel">
            <div className="report-preview-grid">
              <ReportPreview
                title="Punch list"
                meta={`${issues.length} issue rows`}
                body={issues.slice(0, 3).map((issue) => issue.title).join(" · ")}
              />
              <ReportPreview
                title="RFI draft"
                meta={selectedIssue ? selectedIssue.room : "No issue"}
                body={selectedIssue?.rfi_draft || "Select an issue to preview the RFI question."}
              />
              <ReportPreview
                title="Change order evidence"
                meta={`${documents.length} docs · ${mediaAssets.length} media`}
                body="Requirement, observation, plan pin, citation, and field media evidence are bundled for review."
              />
            </div>
            <div className="report-actions">
              <button className="icon-button primary" type="button" onClick={() => generateReport("punch", "pdf")}>
                <ClipboardCheck size={18} />
                <span>Punch PDF</span>
              </button>
              <button className="icon-button" type="button" onClick={() => generateReport("punch", "csv")}>
                <Archive size={18} />
                <span>Punch CSV</span>
              </button>
              <button className="icon-button" type="button" onClick={() => generateReport("co_evidence", "pdf")}>
                <FileQuestion size={18} />
                <span>CO Evidence</span>
              </button>
            </div>
            {reportUrl ? (
              <a className="download-link" href={reportUrl} target="_blank" rel="noreferrer">
                <FileDown size={18} />
                Download generated report
              </a>
            ) : null}
            <div className="rfi-shell">
              <div className="rfi-ref">
                <span>Issue ref</span>
                <strong>{selectedIssue?.issue_id ?? "No issue selected"}</strong>
                <p>{selectedIssue?.title ?? "Select an issue and generate an RFI draft."}</p>
              </div>
              <textarea className="rfi-output" value={rfi} readOnly placeholder="RFI drafts appear here." />
            </div>
          </section>
        )}

        {view === "pipeline" && <PipelineView job={job} apiBase={API_BASE} technologyStatus={technologyStatus} />}
      </section>

      <nav className="bottom-nav" aria-label="Mobile">
        {views.map((item) => (
          <button
            key={item.id}
            className={view === item.id ? "bottom-item active" : "bottom-item"}
            type="button"
            onClick={() => setView(item.id)}
          >
            <item.icon size={19} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function FilterChip({
  active,
  children,
  onClick
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button className={active ? "filter-chip active" : "filter-chip"} type="button" onClick={onClick}>
      {children}
    </button>
  );
}

function IssueInspector({
  issue,
  onApprove,
  onReject,
  onNeedMore,
  onRfi
}: {
  issue?: Issue;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
  onNeedMore: (id: string) => void;
  onRfi: (id: string) => void;
}) {
  if (!issue) {
    return <div className="empty-state">Run analysis to generate issue candidates.</div>;
  }
  return (
    <article className="issue-inspector">
      <div className="inspector-head">
        <div>
          <p className="eyebrow">{issue.discipline}</p>
          <h2>{issue.title}</h2>
        </div>
        <span className={`status-pill ${issue.status}`}>{issue.status.replaceAll("_", " ")}</span>
      </div>
      <p>{issue.description}</p>
      <dl className="detail-list">
        <div>
          <dt>Room</dt>
          <dd>{issue.room}</dd>
        </div>
        <div>
          <dt>Severity</dt>
          <dd>{issue.severity}</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>{Math.round(issue.confidence * 100)}%</dd>
        </div>
        <div>
          <dt>Assignee</dt>
          <dd>{issue.assignee || "Unassigned"}</dd>
        </div>
      </dl>
      <div className="evidence-chain">
        <EvidenceBlock title="Requirement" text={String(issue.requirement.text ?? "")} source={String(issue.requirement.source ?? "")} />
        <EvidenceBlock title="Observation" text={String(issue.observation.text ?? "")} source={String(issue.observation.media_id ?? "")} />
        <EvidenceBlock title="Recommended action" text={issue.recommended_action} source="human review gate" />
      </div>
      <div className="action-bar">
        <button className="icon-button approve" type="button" onClick={() => onApprove(issue.issue_id)}>
          <Check size={18} />
          <span>Approve</span>
        </button>
        <button className="icon-button" type="button" onClick={() => onNeedMore(issue.issue_id)}>
          <MoreHorizontal size={18} />
          <span>More evidence</span>
        </button>
        <button className="icon-button" type="button" onClick={() => onRfi(issue.issue_id)}>
          <MessageSquarePlus size={18} />
          <span>RFI</span>
        </button>
        <button className="icon-button reject" type="button" onClick={() => onReject(issue.issue_id)}>
          <X size={18} />
          <span>Reject</span>
        </button>
      </div>
    </article>
  );
}

function EvidenceBlock({ title, text, source }: { title: string; text: string; source: string }) {
  return (
    <section className="evidence-block">
      <span>{title}</span>
      <p>{text}</p>
      <small>{source}</small>
    </section>
  );
}

function EvidenceViewer({
  issue,
  documents,
  mediaAssets,
  observations
}: {
  issue?: Issue;
  documents: DocumentAsset[];
  mediaAssets: SiteMediaAsset[];
  observations: Observation[];
}) {
  const issueObservations = observations.length
    ? observations
    : (issue?.evidence ?? []).map((evidence, index) => ({
        observation_id: `issue_obs_${index}`,
        media_id: String(issue?.observation.media_id ?? "field_verification_pending"),
        frame_id: String(evidence.ref_id ?? "pending"),
        object_type: String(issue?.type ?? "review_candidate"),
        bbox: evidence.bbox,
        text: String(issue?.observation.text ?? "Field evidence pending."),
        confidence: issue?.confidence ?? 0
      }));
  return (
    <section className="evidence-viewer">
      <div className="section-title-row">
        <h2>Evidence</h2>
        <span>{documents.length} docs · {mediaAssets.length} media</span>
      </div>
      <div className="crop-stage" aria-label="Field crop preview">
        <img className="field-image" src={FIELD_IMAGE_SRC} alt="Construction site electrical work sample" />
        <span className="crop-marker" />
      </div>
      <div className="recognition-list">
        {issueObservations.slice(0, 5).map((item) => (
          <article key={item.observation_id}>
            <strong>{item.object_type.replaceAll("_", " ")}</strong>
            <p>{item.text}</p>
            <small>
              media {shortId(item.media_id)} · confidence {Math.round(item.confidence * 100)}%
            </small>
          </article>
        ))}
      </div>
      <div className="evidence-table">
        {(issue?.evidence ?? []).map((item) => (
          <div key={item.evidence_id}>
            <strong>{item.label || item.evidence_type}</strong>
            <span>{item.r2_key}</span>
            <small>
              page {item.page || "-"} · ts {item.frame_ts || 0}s
            </small>
          </div>
        ))}
      </div>
      <div className="asset-list">
        {documents.slice(0, 3).map((item) => (
          <div key={item.doc_id}>
            <span>{item.type}</span>
            <strong>{item.filename}</strong>
            <small>{item.parsed_status} · rev {item.revision}</small>
          </div>
        ))}
        {mediaAssets.slice(0, 3).map((item) => (
          <div key={item.media_id}>
            <span>media</span>
            <strong>{item.filename}</strong>
            <small>{item.mime} · {shortId(item.hash)}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function FieldSnapshot({ issue, overlay }: { issue?: Issue; overlay: Overlay | null }) {
  const pins = overlay?.pins ?? [];
  const selectedPin = pins.find((pin) => pin.id === issue?.issue_id);
  return (
    <aside className="field-snapshot" aria-label="Field snapshot">
      <div className="section-title-row">
        <h2>Plan trace</h2>
        <span>{pins.length} pins</span>
      </div>
      <div className="mini-plan">
        <img className="plan-image" src={PLAN_IMAGE_SRC} alt="Cooper Residence E1.1 electrical plan" />
        {pins.slice(0, 12).map((pin) => (
          <span
            key={pin.id}
            className={pin.id === issue?.issue_id ? `mini-pin active ${pin.severity}` : `mini-pin ${pin.severity}`}
            style={{ left: `${pin.x * 100}%`, top: `${pin.y * 100}%` }}
            title={`${pin.label} ${pin.room}`}
          />
        ))}
      </div>
      <div className="snapshot-list">
        <div>
          <span>Sheet</span>
          <strong>{String(issue?.plan_location.sheet_id ?? overlay?.sheets[0]?.sheet_number ?? PLAN_LABEL)}</strong>
        </div>
        <div>
          <span>Location</span>
          <strong>{selectedPin?.room ?? issue?.room ?? "pending"}</strong>
        </div>
        <div>
          <span>Evidence</span>
          <strong>{issue?.evidence.length ?? 0} links</strong>
        </div>
      </div>
    </aside>
  );
}

function OverlayView({ overlay }: { overlay: Overlay | null }) {
  return (
    <section className="overlay-panel">
      <div className="section-title-row">
        <h2>Drawing overlay</h2>
        <span>{overlay?.pins.length ?? 0} issue pins</span>
      </div>
      <div className="plan-canvas">
        <img className="plan-image" src={PLAN_IMAGE_SRC} alt="Cooper Residence E1.1 electrical plan" />
        <div className="plan-title">{PLAN_LABEL}</div>
        {(overlay?.regions ?? []).slice(0, 28).map((region) => (
          <span
            key={region.id}
            className="region"
            style={{
              left: `${region.bbox[0] * 100}%`,
              top: `${region.bbox[1] * 100}%`,
              width: `${Math.max((region.bbox[2] - region.bbox[0]) * 100, 2)}%`,
              height: `${Math.max((region.bbox[3] - region.bbox[1]) * 100, 2)}%`
            }}
            title={`${region.type} ${Math.round(region.confidence * 100)}%`}
          />
        ))}
        {(overlay?.pins ?? []).map((pin) => (
          <button
            key={pin.id}
            className={`pin ${pin.severity}`}
            style={{ left: `${pin.x * 100}%`, top: `${pin.y * 100}%` }}
            title={`${pin.label} ${pin.room}`}
            type="button"
          >
            <AlertTriangle size={15} />
          </button>
        ))}
      </div>
    </section>
  );
}

function ReportPreview({ title, meta, body }: { title: string; meta: string; body: string }) {
  return (
    <article className="report-preview">
      <span>{meta}</span>
      <strong>{title}</strong>
      <p>{body}</p>
    </article>
  );
}

function PipelineView({
  job,
  apiBase,
  technologyStatus
}: {
  job: Job | null;
  apiBase: string;
  technologyStatus: TechnologyStatus[];
}) {
  const states = ["queued", "ingesting", "indexing", "extracting_frames", "detecting", "reasoning", "review_ready"];
  return (
    <section className="pipeline-panel">
      <div className="section-title-row">
        <h2>Pipeline</h2>
        <span>{apiBase}</span>
      </div>
      <div className="pipeline-steps">
        {states.map((state, index) => {
          const activeIndex = Math.max(0, states.indexOf(job?.state ?? "queued"));
          const done = index <= activeIndex;
          return (
            <div key={state} className={done ? "pipeline-step done" : "pipeline-step"}>
              <span>{index + 1}</span>
              <strong>{state.replaceAll("_", " ")}</strong>
            </div>
          );
        })}
      </div>
      <div className="technology-grid">
        {technologyStatus.map((item) => (
          <article key={item.key} className={item.status === "ready" ? "technology-item ready" : "technology-item"}>
            <span>{item.status.replaceAll("_", " ")}</span>
            <strong>{item.label}</strong>
            <p>{item.summary}</p>
            <small>{item.evidence_count} evidence records</small>
          </article>
        ))}
      </div>
      <pre className="job-log">{JSON.stringify(job ?? { state: "idle", progress: 0 }, null, 2)}</pre>
    </section>
  );
}

function avg(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function shortId(value: string) {
  if (!value) return "pending";
  return value.length > 12 ? `${value.slice(0, 10)}...` : value;
}

function buildRfiPreview(issue: Issue) {
  return [
    `# RFI Draft: ${issue.title}`,
    "",
    `**Location:** ${issue.room}`,
    "",
    `**Contract requirement:** ${String(issue.requirement.text ?? "")}`,
    "",
    `**Field observation:** ${String(issue.observation.text ?? "")}`,
    "",
    `**Question:** ${issue.rfi_draft}`,
    "",
    "PM review required before sending."
  ].join("\n");
}
