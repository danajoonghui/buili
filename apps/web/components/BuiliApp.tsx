"use client";

import {
  AlertTriangle,
  Archive,
  Box,
  Check,
  ClipboardCheck,
  FileDown,
  FileImage,
  FileQuestion,
  FileText,
  FolderPlus,
  Gauge,
  Grid3X3,
  Hand,
  ListChecks,
  Loader2,
  Map,
  MapPin,
  Maximize,
  MessageSquarePlus,
  MoreHorizontal,
  MousePointer2,
  RefreshCcw,
  RotateCcw,
  Ruler,
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
type IssueFilter = "all" | "open" | "review" | "resolved";
const PLAN_IMAGE_SRC = "/plans/utah-e11-electrical-plan.jpg";
const PLAN_LABEL = "E1.1 Electrical Plans";
const FIELD_IMAGE_SRC = "/site-media/construction-site-electrical-work.jpg";
const PLAN2FIELD_3D_SRC = "/plan2field3d/auto_plan2field3d.png";
const PLAN2FIELD_MINIMAP_SRC = "/plan2field3d/auto_plan_crop.png";
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

const DEMO_SPATIAL_ISSUES: Issue[] = [
  {
    issue_id: "demo-e11-afci",
    project_id: "demo-plan2field",
    type: "coverage_check",
    discipline: "electrical",
    severity: "major",
    room: "Corridor A",
    status: "review_ready",
    confidence: 0.88,
    title: "AFCI Outlet",
    description: "Low wall outlet not installed at the E1.1 corridor location.",
    recommended_action: "Install AFCI outlet per E1.1 plan specification before rough-in approval.",
    assignee: "Electrical Sub",
    due_date: "2026-07-02",
    subcontractor: "Electrical Sub",
    requirement: {
      text: "Electrical notes require AFCI protected outlets in living and sleeping areas.",
      source: "E1.1"
    },
    observation: {
      text: "Field image and 3D alignment show missing low wall outlet at corridor wall.",
      media_id: "field-photo-01"
    },
    plan_location: { sheet_id: "A-101", code: "E1.1", x: 52, y: 32 },
    rfi_draft: "",
    evidence: [
      {
        evidence_id: "demo-evidence-1",
        evidence_type: "plan_pin",
        ref_id: "A-101",
        r2_key: PLAN2FIELD_3D_SRC,
        page: 1,
        bbox: [0.49, 0.29, 0.55, 0.36],
        frame_ts: 0,
        label: "AFCI outlet plan pin"
      },
      {
        evidence_id: "demo-evidence-2",
        evidence_type: "field_photo",
        ref_id: "field-photo-01",
        r2_key: FIELD_IMAGE_SRC,
        page: 0,
        bbox: [0.31, 0.28, 0.52, 0.58],
        frame_ts: 0,
        label: "missing outlet location"
      },
      {
        evidence_id: "demo-evidence-3",
        evidence_type: "spatial_view",
        ref_id: "plan2field-3d",
        r2_key: PLAN2FIELD_3D_SRC,
        page: 0,
        bbox: [0.45, 0.22, 0.62, 0.43],
        frame_ts: 0,
        label: "3D issue alignment"
      }
    ],
    spatial_context: {
      spatial_evidence_id: "spatial-demo-1",
      room_graph_id: "room-graph-e11",
      design_asset_id: "A-101",
      field_asset_id: "field-photo-01",
      snapshot_uri: PLAN2FIELD_3D_SRC,
      spatial_note: "OCR, symbol extraction, wall union geometry, and issue pin are aligned in the 3D model.",
      alignment_confidence: 0.91,
      geometry_confidence: 0.87,
      geometry_features: { walls: 48, openings: 15, objects: 14 }
    }
  },
  {
    issue_id: "demo-m24-diffuser",
    project_id: "demo-plan2field",
    type: "location_mismatch",
    discipline: "mechanical",
    severity: "major",
    room: "Office 204",
    status: "needs_more_evidence",
    confidence: 0.76,
    title: "Supply Diffuser",
    description: "Diffuser location mismatch against M2.4 ceiling coordination note.",
    recommended_action: "Verify diffuser location against reflected ceiling plan and capture correction evidence.",
    assignee: "Mechanical Sub",
    due_date: "2026-07-03",
    subcontractor: "Mechanical Sub",
    requirement: { text: "Supply diffuser to align with ceiling grid and M2.4 room mark.", source: "M-201" },
    observation: { text: "3D overlay places diffuser outside expected bay.", media_id: "field-photo-02" },
    plan_location: { sheet_id: "M-201", code: "M2.4", x: 67, y: 48 },
    rfi_draft: "",
    evidence: [],
    spatial_context: {
      spatial_evidence_id: "spatial-demo-2",
      room_graph_id: "room-graph-e11",
      snapshot_uri: PLAN2FIELD_3D_SRC,
      spatial_note: "Ceiling object candidate is pinned for PM review.",
      alignment_confidence: 0.82,
      geometry_confidence: 0.86
    }
  },
  {
    issue_id: "demo-a32-door",
    project_id: "demo-plan2field",
    type: "dimension_mismatch",
    discipline: "architectural",
    severity: "blocker",
    room: "Room 101",
    status: "review_ready",
    confidence: 0.81,
    title: "Door Width",
    description: "Actual 810mm door opening is below the 910mm plan requirement.",
    recommended_action: "Confirm framed opening width before drywall close-in.",
    assignee: "GC",
    due_date: "2026-07-01",
    subcontractor: "GC",
    requirement: { text: "Door opening required at 910mm clear width.", source: "A-101" },
    observation: { text: "Measured field opening reads 810mm.", media_id: "field-photo-03" },
    plan_location: { sheet_id: "A-101", code: "A3.2", x: 39, y: 62 },
    rfi_draft: "",
    evidence: [],
    spatial_context: {
      spatial_evidence_id: "spatial-demo-3",
      room_graph_id: "room-graph-e11",
      snapshot_uri: PLAN2FIELD_3D_SRC,
      spatial_note: "Door swing and wall opening are extracted into a lightweight 3D opening object.",
      alignment_confidence: 0.88,
      geometry_confidence: 0.9
    }
  },
  {
    issue_id: "demo-p13-pipe",
    project_id: "demo-plan2field",
    type: "clearance_check",
    discipline: "mechanical",
    severity: "minor",
    room: "Mechanical Room",
    status: "needs_more_evidence",
    confidence: 0.69,
    title: "Pipe Clearance",
    description: "Pipe clearance appears less than 25mm at the utility wall.",
    recommended_action: "Capture one additional field photo with tape reference.",
    assignee: "Field PM",
    due_date: "2026-07-05",
    subcontractor: "Mechanical Sub",
    requirement: { text: "Maintain minimum service clearance at utility wall.", source: "P-101" },
    observation: { text: "Spatial candidate needs PM verification.", media_id: "field-photo-04" },
    plan_location: { sheet_id: "P-101", code: "P1.3", x: 23, y: 74 },
    rfi_draft: "",
    evidence: [],
    spatial_context: {
      spatial_evidence_id: "spatial-demo-4",
      room_graph_id: "room-graph-e11",
      snapshot_uri: PLAN2FIELD_3D_SRC,
      spatial_note: "Pipe clearance issue is generated as a candidate, not a final defect.",
      alignment_confidence: 0.74,
      geometry_confidence: 0.79
    }
  }
];

const DEMO_DOCUMENTS: DocumentAsset[] = [
  {
    doc_id: "demo-doc-a101",
    project_id: "demo-plan2field",
    type: "plan",
    filename: "A-101.pdf",
    mime: "application/pdf",
    r2_key: PLAN2FIELD_MINIMAP_SRC,
    hash: "demo-a101",
    revision: "A",
    parsed_status: "parsed",
    size: 0,
    metadata_json: { sheet_id: "A-101" }
  },
  {
    doc_id: "demo-doc-spec",
    project_id: "demo-plan2field",
    type: "spec",
    filename: "Spec_Electrical.pdf",
    mime: "application/pdf",
    r2_key: PLAN_IMAGE_SRC,
    hash: "demo-spec",
    revision: "A",
    parsed_status: "parsed",
    size: 0,
    metadata_json: { section: "Electrical" }
  }
];

const DEMO_MEDIA_ASSETS: SiteMediaAsset[] = [
  {
    media_id: "field-photo-01",
    project_id: "demo-plan2field",
    filename: "outlet-closeup.jpg",
    mime: "image/jpeg",
    r2_key: FIELD_IMAGE_SRC,
    hash: "demo-media-1",
    metadata_json: { type: "field_evidence" }
  },
  {
    media_id: "field-photo-02",
    project_id: "demo-plan2field",
    filename: "marked-plan.jpg",
    mime: "image/jpeg",
    r2_key: PLAN_IMAGE_SRC,
    hash: "demo-media-2",
    metadata_json: { type: "plan_evidence" }
  },
  {
    media_id: "field-photo-03",
    project_id: "demo-plan2field",
    filename: "spatial-model.jpg",
    mime: "image/png",
    r2_key: PLAN2FIELD_3D_SRC,
    hash: "demo-media-3",
    metadata_json: { type: "spatial_evidence" }
  }
];

const DEMO_OVERLAY: Overlay = {
  project_id: "demo-plan2field",
  sheets: [{ id: "A-101", title: "Plan2Field 3D source sheet" }],
  pins: [
    { id: "demo-e11-afci", label: "E1.1", severity: "major", room: "Corridor A", x: 0.52, y: 0.32, confidence: 0.88 },
    { id: "demo-m24-diffuser", label: "M2.4", severity: "major", room: "Office 204", x: 0.67, y: 0.48, confidence: 0.76 },
    { id: "demo-a32-door", label: "A3.2", severity: "blocker", room: "Room 101", x: 0.39, y: 0.62, confidence: 0.81 },
    { id: "demo-p13-pipe", label: "P1.3", severity: "minor", room: "Mechanical Room", x: 0.23, y: 0.74, confidence: 0.69 }
  ],
  regions: []
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
  const displayIssues = issues.length ? issues : DEMO_SPATIAL_ISSUES;
  const displayDocuments = documents.length ? documents : DEMO_DOCUMENTS;
  const displayMediaAssets = mediaAssets.length ? mediaAssets : DEMO_MEDIA_ASSETS;
  const displayOverlay = overlay ?? DEMO_OVERLAY;

  const selectedIssue = useMemo(
    () => displayIssues.find((issue) => issue.issue_id === selectedIssueId) ?? displayIssues[0],
    [displayIssues, selectedIssueId]
  );

  const actionNeeded = useMemo(
    () => displayIssues.filter((issue) => issue.status === "review_ready" && issue.confidence >= 0.55).length,
    [displayIssues]
  );

  const orderedIssues = useMemo(
    () =>
      [...displayIssues].sort(
        (a, b) =>
          (severityRank[b.severity] ?? 0) - (severityRank[a.severity] ?? 0) ||
          b.confidence - a.confidence
      ),
    [displayIssues]
  );

  const filteredIssues = useMemo(
    () =>
      orderedIssues.filter((issue) => {
        if (issueFilter === "all") return true;
        if (issueFilter === "open") return issue.status === "review_ready";
        if (issueFilter === "review") return issue.status === "needs_more_evidence";
        return issue.status === "approved";
      }),
    [issueFilter, orderedIssues]
  );

  const issueCounts = useMemo(
    () => ({
      all: displayIssues.length,
      open: displayIssues.filter((issue) => issue.status === "review_ready").length,
      review: displayIssues.filter((issue) => issue.status === "needs_more_evidence").length,
      resolved: displayIssues.filter((issue) => issue.status === "approved").length
    }),
    [displayIssues]
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
    if (issueId.startsWith("demo-")) {
      setNotice(`Demo issue marked as ${status.replaceAll("_", " ")}. Connect a project to persist review decisions.`);
      return;
    }
    try {
      const updated = await api.updateIssue(issueId, { status });
      setIssues((current) => current.map((issue) => (issue.issue_id === issueId ? updated : issue)));
      setNotice(`Issue marked as ${status.replaceAll("_", " ")}.`);
    } catch (error) {
      surfaceError(error, "Issue update failed");
    }
  }

  async function generateRfi(issueId: string) {
    const demoIssue = DEMO_SPATIAL_ISSUES.find((item) => item.issue_id === issueId);
    if (demoIssue) {
      setRfi(buildRfiPreview(demoIssue));
      setNotice("Demo RFI draft generated from Plan2Field-3D evidence.");
      setView("reports");
      return;
    }
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
          <Metric label="Issues ready" value={displayIssues.filter((issue) => issue.status === "review_ready").length} />
          <Metric label="Action needed" value={actionNeeded} />
          <Metric label="Evidence score" value={`${Math.round(avg(displayIssues.map((issue) => issue.confidence)) * 100)}%`} />
          <Metric label="Job" value={job ? `${job.progress}%` : issues.length ? "100%" : "demo"} />
        </section>

        {view === "review" && (
          <Plan2FieldReview
            issue={selectedIssue}
            filteredIssues={filteredIssues}
            issueCounts={issueCounts}
            issueFilter={issueFilter}
            overlay={displayOverlay}
            documents={displayDocuments}
            mediaAssets={displayMediaAssets}
            onFilter={setIssueFilter}
            onSelectIssue={setSelectedIssueId}
            onApprove={(id) => patchIssue(id, "approved")}
            onReject={(id) => patchIssue(id, "rejected_false_positive")}
            onNeedMore={(id) => patchIssue(id, "needs_more_evidence")}
            onRfi={generateRfi}
          />
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
                body="Requirement, observation, plan pin, citation, field media, and Plan2Field-3D spatial evidence are bundled for review."
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

function Plan2FieldReview({
  issue,
  filteredIssues,
  issueCounts,
  issueFilter,
  overlay,
  documents,
  mediaAssets,
  onFilter,
  onSelectIssue,
  onApprove,
  onReject,
  onNeedMore,
  onRfi
}: {
  issue?: Issue;
  filteredIssues: Issue[];
  issueCounts: Record<IssueFilter, number>;
  issueFilter: IssueFilter;
  overlay: Overlay | null;
  documents: DocumentAsset[];
  mediaAssets: SiteMediaAsset[];
  onFilter: (filter: IssueFilter) => void;
  onSelectIssue: (id: string) => void;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
  onNeedMore: (id: string) => void;
  onRfi: (id: string) => void;
}) {
  const pins = useMemo(() => buildModelPins(filteredIssues, overlay), [filteredIssues, overlay]);
  const activeIssue = filteredIssues.find((item) => item.issue_id === issue?.issue_id) ?? filteredIssues[0] ?? issue;
  const selectedPin = pins.find((pin) => pin.issueId === activeIssue?.issue_id) ?? pins[0];

  return (
    <section className="spatial-review-shell">
      <aside className="spatial-issue-rail" aria-label="Issues">
        <div className="spatial-rail-head">
          <h2>Issues</h2>
          <span>{filteredIssues.length} shown</span>
        </div>
        <div className="spatial-filter-row" aria-label="Issue filters">
          <FilterChip active={issueFilter === "all"} onClick={() => onFilter("all")}>
            All {issueCounts.all}
          </FilterChip>
          <FilterChip active={issueFilter === "open"} onClick={() => onFilter("open")}>
            Open {issueCounts.open}
          </FilterChip>
          <FilterChip active={issueFilter === "review"} onClick={() => onFilter("review")}>
            In Review {issueCounts.review}
          </FilterChip>
          <FilterChip active={issueFilter === "resolved"} onClick={() => onFilter("resolved")}>
            Resolved {issueCounts.resolved}
          </FilterChip>
        </div>
        <div className="spatial-issue-list">
          {filteredIssues.map((item) => (
            <button
              key={item.issue_id}
              className={item.issue_id === activeIssue?.issue_id ? "spatial-issue-card active" : "spatial-issue-card"}
              type="button"
              onClick={() => onSelectIssue(item.issue_id)}
            >
              <span className={`issue-marker ${item.status === "approved" ? "resolved" : issueTone(item)}`}>
                {item.status === "approved" ? <Check size={15} /> : issueTone(item) === "warning" ? "!" : ""}
              </span>
              <span className="spatial-issue-copy">
                <strong>{issueCode(item)} {issueTitle(item)}</strong>
                <small>{locationLine(item)}</small>
                <span>{item.description || item.recommended_action}</span>
                <em>
                  <FileText size={14} />
                  {String(item.plan_location.sheet_id ?? item.requirement.source ?? PLAN_LABEL)}
                  <FileImage size={14} />
                  {item.evidence.length || 1}
                </em>
              </span>
              <span className={`mini-status ${item.status}`}>{statusLabel(item.status)}</span>
            </button>
          ))}
        </div>
        <button className="add-issue-button" type="button">
          <span>+</span>
          Add Issue
        </button>
      </aside>

      <section className="model-workspace">
        <div className="model-toolbar" aria-label="3D tools">
          <button className="tool-button active" type="button" title="Select">
            <MousePointer2 size={20} />
          </button>
          <button className="tool-button" type="button" title="Pan">
            <Hand size={20} />
          </button>
          <button className="tool-button" type="button" title="Rotate">
            <RotateCcw size={20} />
          </button>
          <button className="tool-button" type="button" title="Fit">
            <Maximize size={20} />
          </button>
          <button className="tool-button" type="button" title="Model">
            <Box size={20} />
          </button>
          <button className="tool-button" type="button" title="Grid">
            <Grid3X3 size={20} />
          </button>
          <button className="tool-button" type="button" title="Measure">
            <Ruler size={20} />
          </button>
        </div>

        <div className="view-toggle" aria-label="Plan view mode">
          <button type="button">2D Plan</button>
          <button className="active" type="button">3D Model</button>
        </div>

        <div className="model-stage">
          <img src={PLAN2FIELD_3D_SRC} alt="Buili Plan2Field 3D model generated from PDF drawing" />
          {pins.map((pin) => (
            <button
              key={pin.issueId}
              className={`model-pin ${pin.tone} ${pin.issueId === activeIssue?.issue_id ? "active" : ""}`}
              style={{ left: `${pin.x}%`, top: `${pin.y}%` }}
              type="button"
              onClick={() => onSelectIssue(pin.issueId)}
              title={pin.title}
            >
              <MapPin size={34} fill="currentColor" />
              <span>{pin.code}</span>
            </button>
          ))}
        </div>

        <div className="floor-key">
          <strong>Floor Plan Key</strong>
          <span><i className="key-dot open" /> Open</span>
          <span><i className="key-dot review" /> In Review</span>
          <span><i className="key-dot resolved" /> Resolved</span>
        </div>

        <div className="mini-map-card">
          <img src={PLAN2FIELD_MINIMAP_SRC} alt="2D plan minimap" />
          {pins.slice(0, 8).map((pin) => (
            <span
              key={pin.issueId}
              className={`mini-map-pin ${pin.tone}`}
              style={{ left: `${pin.minimapX}%`, top: `${pin.minimapY}%` }}
            />
          ))}
        </div>
      </section>

      <section className="spatial-bottom-panel">
        <div className="bottom-evidence">
          <PanelTitle label="Field Evidence" count={Math.max(mediaAssets.length, activeIssue?.evidence.length ?? 0)} />
          <div className="thumb-row">
            <EvidenceThumb src={FIELD_IMAGE_SRC} label="Outlet close-up" />
            <EvidenceThumb src={PLAN_IMAGE_SRC} label="Marked plan area" />
            <EvidenceThumb src={PLAN2FIELD_3D_SRC} label="3D spatial view" />
          </div>
        </div>
        <div className="bottom-documents">
          <PanelTitle label="Documents" count={Math.max(documents.length, 2)} />
          <div className="doc-strip">
            <DocThumb src={PLAN2FIELD_MINIMAP_SRC} name={String(activeIssue?.plan_location.sheet_id ?? "A-101.pdf")} />
            <DocThumb src={PLAN_IMAGE_SRC} name="Spec_Electrical.pdf" />
          </div>
        </div>
        <IssueSummaryPanel
          issue={activeIssue}
          selectedPin={selectedPin}
          onApprove={onApprove}
          onReject={onReject}
          onNeedMore={onNeedMore}
          onRfi={onRfi}
        />
      </section>
    </section>
  );
}

function PanelTitle({ label, count }: { label: string; count: number }) {
  return (
    <div className="panel-title">
      <strong>{label}</strong>
      <span>({count})</span>
    </div>
  );
}

function EvidenceThumb({ src, label }: { src: string; label: string }) {
  return (
    <figure className="evidence-thumb">
      <img src={src} alt={label} />
      <figcaption>{label}</figcaption>
    </figure>
  );
}

function DocThumb({ src, name }: { src: string; name: string }) {
  return (
    <figure className="doc-thumb">
      <img src={src} alt={name} />
      <figcaption>
        <span>{name}</span>
        <b>PDF</b>
      </figcaption>
    </figure>
  );
}

function IssueSummaryPanel({
  issue,
  selectedPin,
  onApprove,
  onReject,
  onNeedMore,
  onRfi
}: {
  issue?: Issue;
  selectedPin?: ReturnType<typeof buildModelPins>[number];
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
  onNeedMore: (id: string) => void;
  onRfi: (id: string) => void;
}) {
  if (!issue) {
    return (
      <aside className="issue-summary-panel">
        <strong>Issue Summary</strong>
        <p>No issue selected.</p>
      </aside>
    );
  }
  return (
    <aside className="issue-summary-panel">
      <strong>Issue Summary</strong>
      <dl>
        <div>
          <dt>Confidence</dt>
          <dd>
            <span className={issue.confidence >= 0.7 ? "confidence-high" : "confidence-medium"}>
              {issue.confidence >= 0.7 ? "High" : "Review"}
            </span>
            {Math.round(issue.confidence * 100)}%
          </dd>
        </div>
        <div>
          <dt>Recommended Action</dt>
          <dd>{issue.recommended_action}</dd>
        </div>
        <div>
          <dt>Assignee</dt>
          <dd>{issue.assignee || issue.subcontractor || "Field PM"}</dd>
        </div>
        <div>
          <dt>Due Date</dt>
          <dd>{issue.due_date || "Open"}</dd>
        </div>
        <div>
          <dt>Spatial Pin</dt>
          <dd>{selectedPin?.code ?? issueCode(issue)}</dd>
        </div>
      </dl>
      <div className="summary-actions">
        <button type="button" onClick={() => onApprove(issue.issue_id)}>
          <Check size={16} />
          Approve
        </button>
        <button type="button" onClick={() => onNeedMore(issue.issue_id)}>
          <MoreHorizontal size={16} />
          More
        </button>
        <button type="button" onClick={() => onRfi(issue.issue_id)}>
          <MessageSquarePlus size={16} />
          RFI
        </button>
        <button type="button" onClick={() => onReject(issue.issue_id)}>
          <X size={16} />
          Reject
        </button>
      </div>
    </aside>
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
      <SpatialEvidenceSummary issue={issue} />
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

function SpatialEvidenceSummary({ issue }: { issue: Issue }) {
  const context = issue.spatial_context;
  const features = context?.geometry_features ?? {};
  if (!context?.spatial_evidence_id) {
    return (
      <section className="spatial-summary muted">
        <span>Plan2Field-3D</span>
        <strong>Spatial evidence pending</strong>
        <p>Run review with spatial enabled to add room graph, alignment, and geometry evidence.</p>
      </section>
    );
  }
  return (
    <section className="spatial-summary">
      <div className="spatial-summary-head">
        <span>Plan2Field-3D</span>
        <strong>{String(context.room_graph_id ?? "room pending")}</strong>
      </div>
      <p>{String(context.spatial_note ?? "Spatial evidence generated for PM review.")}</p>
      <dl>
        <div>
          <dt>Alignment</dt>
          <dd>{percent(Number(context.alignment_confidence ?? features.room_alignment_confidence ?? 0))}</dd>
        </div>
        <div>
          <dt>Geometry</dt>
          <dd>{percent(Number(context.geometry_confidence ?? features.geometry_confidence ?? 0))}</dd>
        </div>
        <div>
          <dt>Coverage</dt>
          <dd>{percent(Number(features.field_coverage_ratio ?? 0))}</dd>
        </div>
        <div>
          <dt>Count</dt>
          <dd>
            {String(features.observed_count ?? 0)} / {String(features.required_count ?? 0)}
          </dd>
        </div>
      </dl>
      <small>
        {String(context.spatial_evidence_id)} · {String(context.snapshot_uri ?? "snapshot pending")}
      </small>
    </section>
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
  const states = [
    "queued",
    "ingesting",
    "indexing",
    "extracting_frames",
    "detecting",
    "spatializing_plan",
    "reconstructing_field",
    "aligning_plan_field",
    "reasoning",
    "review_ready"
  ];
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

function buildModelPins(issues: Issue[], overlay: Overlay | null) {
  const fallback = [
    { x: 34, y: 42, minimapX: 35, minimapY: 42 },
    { x: 56, y: 51, minimapX: 58, minimapY: 55 },
    { x: 67, y: 35, minimapX: 70, minimapY: 36 },
    { x: 45, y: 62, minimapX: 47, minimapY: 64 },
    { x: 75, y: 48, minimapX: 78, minimapY: 50 },
    { x: 29, y: 58, minimapX: 30, minimapY: 60 }
  ];
  return issues.slice(0, 8).map((issue, index) => {
    const pin = overlay?.pins.find((item) => item.id === issue.issue_id);
    const point = fallback[index % fallback.length];
    const tone = issue.status === "approved" ? "resolved" : issueTone(issue);
    return {
      issueId: issue.issue_id,
      code: issueCode(issue),
      title: issue.title,
      tone,
      x: pin ? 16 + pin.x * 68 : point.x,
      y: pin ? 16 + pin.y * 62 : point.y,
      minimapX: pin ? pin.x * 100 : point.minimapX,
      minimapY: pin ? pin.y * 100 : point.minimapY
    };
  });
}

function issueCode(issue: Issue) {
  const explicitCode = String(issue.plan_location.code ?? issue.requirement.code ?? "");
  if (explicitCode) return explicitCode;
  const source = String(issue.requirement.source ?? issue.plan_location.sheet_id ?? "");
  const sheet = source.match(/[A-Z]\d(?:\.\d)?/i)?.[0]?.toUpperCase();
  if (sheet) return sheet;
  const prefix = issue.discipline.startsWith("mech") ? "M" : issue.discipline.startsWith("plumb") ? "P" : "E";
  const compact = issue.issue_id.replace(/[^a-z0-9]/gi, "").slice(-2).toUpperCase() || "1";
  return `${prefix}${compact}`;
}

function issueTitle(issue: Issue) {
  const words = issue.title.split(/\s+/).filter(Boolean);
  return words.length > 5 ? `${words.slice(0, 5).join(" ")}...` : issue.title;
}

function issueTone(issue: Issue) {
  if (issue.status === "needs_more_evidence") return "review";
  if (issue.severity === "minor" || issue.severity === "informational") return "warning";
  return "open";
}

function statusLabel(status: string) {
  if (status === "approved") return "Resolved";
  if (status === "needs_more_evidence") return "In Review";
  if (status === "rejected_false_positive") return "Rejected";
  return "Open";
}

function locationLine(issue: Issue) {
  const level = String(issue.plan_location.level ?? issue.plan_location.sheet_id ?? "Level 1");
  return `${level} / ${issue.room || "Field area"}`;
}

function avg(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percent(value: number) {
  if (!Number.isFinite(value)) return "0%";
  return `${Math.round(value * 100)}%`;
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
