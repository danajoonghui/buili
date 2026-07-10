export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api";

export type BuiliIdentity = {
  role: "project_manager" | "project_engineer" | "superintendent" | "field_user" | "external_reviewer" | "admin" | string;
  actor: string;
};

export type AuthUser = {
  user_id: string;
  email: string;
  name: string;
  role: string;
  organization: {
    org_id: string;
    name: string;
  };
};

export type AuthSession = {
  user: AuthUser;
  projects: Project[];
  expires_at: string;
};

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const DEFAULT_IDENTITY: BuiliIdentity = { role: "project_manager", actor: "Jordan Davis" };

export function getBuiliIdentity(): BuiliIdentity {
  if (typeof window === "undefined") return DEFAULT_IDENTITY;
  try {
    const saved = JSON.parse(window.localStorage.getItem("buili.identity") ?? "null") as Partial<BuiliIdentity> | null;
    return {
      role: saved?.role || DEFAULT_IDENTITY.role,
      actor: saved?.actor || DEFAULT_IDENTITY.actor
    };
  } catch {
    return DEFAULT_IDENTITY;
  }
}

export function setBuiliIdentity(identity: BuiliIdentity) {
  if (typeof window !== "undefined") window.localStorage.setItem("buili.identity", JSON.stringify(identity));
}

export type Project = {
  project_id: string;
  org_id: string;
  name: string;
  address: string;
  project_type: string;
  status: string;
};

export type Evidence = {
  evidence_id: string;
  evidence_type: string;
  ref_id: string;
  r2_key: string;
  page: number;
  bbox: number[];
  frame_ts: number;
  label: string;
};

export type Issue = {
  issue_id: string;
  project_id: string;
  type: string;
  discipline: string;
  severity: "blocker" | "major" | "minor" | "informational" | string;
  room: string;
  status: string;
  confidence: number;
  title: string;
  description: string;
  recommended_action: string;
  assignee: string;
  due_date: string;
  subcontractor: string;
  requirement: Record<string, string | number | boolean>;
  observation: Record<string, string | number | boolean>;
  plan_location: Record<string, string | number | boolean | number[]>;
  rfi_draft: string;
  evidence: Evidence[];
  spatial_context?: {
    spatial_evidence_id?: string;
    room_graph_id?: string;
    design_asset_id?: string;
    field_asset_id?: string;
    snapshot_uri?: string;
    spatial_note?: string;
    alignment_confidence?: number;
    geometry_confidence?: number;
    geometry_features?: Record<string, string | number | boolean>;
  };
};

export type Job = {
  job_id: string;
  project_id: string;
  state: string;
  progress: number;
  retry_count: number;
  input_hash: string;
  error: string;
  events: Array<Record<string, unknown>>;
};

export type DocumentAsset = {
  doc_id: string;
  project_id: string;
  type: string;
  filename: string;
  mime: string;
  r2_key: string;
  hash: string;
  revision: string;
  parsed_status: string;
  size: number;
  metadata_json: Record<string, unknown>;
};

export type SiteMediaAsset = {
  media_id: string;
  project_id: string;
  filename: string;
  mime: string;
  r2_key: string;
  hash: string;
  metadata_json: Record<string, unknown>;
  download_url?: string;
};

export type Observation = {
  observation_id: string;
  media_id: string;
  frame_id: string;
  object_type: string;
  bbox: number[];
  text: string;
  confidence: number;
};

export type TechnologyStatus = {
  key: string;
  label: string;
  status: string;
  evidence_count: number;
  summary: string;
};

export type Overlay = {
  project_id: string;
  sheets: Array<Record<string, string | number>>;
  pins: Array<{
    id: string;
    label: string;
    severity: string;
    room: string;
    x: number;
    y: number;
    confidence: number;
  }>;
  regions: Array<{
    id: string;
    type: string;
    room: string;
    bbox: number[];
    confidence: number;
    source: string;
  }>;
};

export type SpatialEvidence = {
  id: string;
  issue_id: string;
  room_graph_id: string;
  design_asset_id: string;
  field_asset_id: string;
  geometry_features_json: Record<string, string | number | boolean>;
  snapshot_uri: string;
  spatial_note: string;
};

export type SpatialAsset = {
  id: string;
  project_id: string;
  type: string;
  uri: string;
  metadata_json: Record<string, string | number | boolean | unknown[]>;
};

export type ProjectSettings = {
  project_id: string;
  name?: string;
  address?: string;
  client?: string;
  timezone: string;
  unit_system: "imperial" | "metric" | string;
  settings: Record<string, unknown>;
  workflow: Record<string, unknown>;
};

export type DirectoryMember = {
  directory_id: string;
  project_id: string;
  person_name: string;
  email: string;
  company: string;
  role: string;
  trade: string;
  status: "invited" | "active" | "disabled" | string;
  notification?: Record<string, unknown>;
  notification_json?: Record<string, unknown>;
  access_expires_at?: string | null;
  created_at?: string;
  updated_at?: string;
};

export type DrawingRevision = {
  revision_id?: string;
  document_id?: string;
  doc_id?: string;
  project_id?: string;
  logical_key?: string;
  sheet_number?: string;
  revision?: string;
  issue_date?: string;
  discipline?: string;
  state?: "current" | "superseded" | "unclassified" | string;
  supersedes_document_id?: string;
  activated_at?: string | null;
  source_hash?: string;
  upload_actor?: string;
  parse_version?: string;
  filename?: string;
  impacted_issue_count?: number;
};

export type FieldEvidenceRecord = {
  evidence_id: string;
  project_id: string;
  client_capture_id?: string;
  media_id?: string;
  media_type: string;
  filename: string;
  mime: string;
  uri?: string;
  hash?: string;
  captured_at: string;
  author: string;
  location?: Record<string, unknown>;
  location_json?: Record<string, unknown>;
  location_method: string;
  metadata?: Record<string, unknown>;
  metadata_json?: Record<string, unknown>;
  quality?: Record<string, unknown>;
  quality_json?: Record<string, unknown>;
  sufficiency: string;
  status: string;
};

export type ReviewRecord = {
  review_id: string;
  issue_id: string;
  reviewer: string;
  decision: string;
  reason?: string;
  reason_code?: string;
  issue_version?: number;
  created_at?: string;
};

export type ReportRecord = {
  report_id: string;
  project_id?: string;
  report_type: string;
  title?: string;
  status?: string;
  format?: string;
  download_url?: string;
  created_by?: string;
  created_at?: string;
  issue_snapshot?: Array<{
    issue_id?: string;
    status?: string;
    title?: string;
    workflow?: { review_status?: string; source_status?: string };
  }>;
  source_snapshot?: Array<Record<string, unknown>>;
  versions?: ReportVersion[];
};

export type ReportVersion = {
  version_id: string;
  report_id: string;
  version: number;
  format: string;
  status: string;
  download_url?: string;
  created_by?: string;
  reviewer?: string;
  issued_at?: string | null;
  created_at?: string;
  issue_snapshot?: ReportRecord["issue_snapshot"];
  source_snapshot?: Array<Record<string, unknown>>;
};

export type SearchResult = {
  type: "issue" | "drawing" | "document" | "requirement" | "specification" | "evidence" | "rfi" | "submittal" | "person" | "company" | string;
  id: string;
  title: string;
  subtitle?: string;
  snippet?: string;
  revision?: string;
  status?: string;
  location?: string;
  project_id?: string;
  route?: string;
  score?: number;
  metadata?: Record<string, unknown>;
};

export type ProjectNotification = {
  notification_id: string;
  event_type: string;
  title: string;
  body: string;
  entity_type?: string;
  entity_id?: string;
  read_at?: string | null;
  created_at?: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    credentials: "same-origin",
    cache: "no-store"
  });
  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body) as { detail?: string; message?: string };
      detail = parsed.detail || parsed.message || body;
    } catch {
      // Non-JSON upstream errors still retain their useful response text.
    }
    if (response.status === 401 && !path.startsWith("/v1/auth/") && typeof window !== "undefined") {
      window.dispatchEvent(new Event("buili:unauthorized"));
    }
    throw new ApiError(response.status, detail || `${response.status} ${response.statusText}`);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export type UploadPresignResponse = {
  upload_id: string;
  method: string;
  upload_url: string;
  complete_url: string;
  r2_key: string;
  headers: Record<string, string>;
};

export const api = {
  login: (payload: { email: string; password: string; remember_me?: boolean }) =>
    request<AuthSession>("/v1/auth/login", { method: "POST", body: JSON.stringify(payload) }),
  me: () => request<AuthSession>("/v1/auth/me"),
  logout: () => request<void>("/v1/auth/logout", { method: "POST" }),
  listProjects: () => request<Project[]>("/v1/projects"),
  createProject: (payload: { name: string; address: string; project_type: string }) =>
    request<Project>("/v1/projects", { method: "POST", body: JSON.stringify(payload) }),
  presignUpload: (payload: {
    project_id: string;
    filename: string;
    mime: string;
    size: number;
    kind: "document" | "media" | "submittal";
  }) => request<UploadPresignResponse>("/v1/uploads/presign", { method: "POST", body: JSON.stringify(payload) }),
  uploadFile: async (uploadUrl: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(uploadUrl, { method: "POST", credentials: "same-origin", body: form });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`${response.status} ${response.statusText}: ${body}`);
    }
    return (await response.json()) as { upload_id: string; size: number; sha256: string; r2_key: string };
  },
  completeUpload: (completeUrl: string, payload: { document_type: string; revision: string }) =>
    fetch(completeUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload)
    }).then(async (response) => {
      if (!response.ok) {
        const body = await response.text();
        throw new Error(`${response.status} ${response.statusText}: ${body}`);
      }
      return (await response.json()) as { status: string; document_id?: string; media_id?: string };
    }),
  analyzeProject: (projectId: string) =>
    request<Job>(`/v1/projects/${projectId}/analyze`, {
      method: "POST",
      body: JSON.stringify({ priority: "normal", force: false })
    }),
  getJob: (jobId: string) => request<Job>(`/v1/jobs/${jobId}`),
  latestJob: (projectId: string) => request<Job | null>(`/v1/projects/${projectId}/jobs/latest`),
  listDocuments: (projectId: string) => request<DocumentAsset[]>(`/v1/projects/${projectId}/documents`),
  listMedia: (projectId: string) => request<SiteMediaAsset[]>(`/v1/projects/${projectId}/media`),
  listObservations: (projectId: string) => request<Observation[]>(`/v1/projects/${projectId}/observations`),
  technologyStatus: (projectId: string) =>
    request<TechnologyStatus[]>(`/v1/projects/${projectId}/technology-status`),
  listIssues: (projectId: string) => request<Issue[]>(`/v1/projects/${projectId}/issues`),
  updateIssue: (issueId: string, patch: Partial<Issue>) =>
    request<Issue>(`/v1/issues/${issueId}`, { method: "PATCH", body: JSON.stringify(patch) }),
  createRfi: (issueId: string) =>
    request<{ issue_id: string; title: string; markdown: string }>(`/v1/issues/${issueId}/rfi`, {
      method: "POST"
    }),
  createReport: (projectId: string, report_type: "punch" | "co_evidence" | "rfi", format: "pdf" | "csv", issue_ids?: string[]) =>
    request<{ report_id: string; report_type: string; format: string; download_url: string }>(
      `/v1/projects/${projectId}/reports`,
      { method: "POST", body: JSON.stringify({ report_type, format, ...(issue_ids?.length ? { issue_ids } : {}) }) }
    ),
  getOverlay: (projectId: string) => request<Overlay>(`/v1/projects/${projectId}/plan-overlay`),
  createDesign3d: (projectId: string, force = false) =>
    request<SpatialAsset>(`/v1/projects/${projectId}/spatial/design-3d`, { method: "POST", body: JSON.stringify({ force }) }),
  getIssueSpatial: (issueId: string) => request<SpatialEvidence[]>(`/v1/issues/${issueId}/spatial`),
  ragSearch: (projectId: string, query: string) =>
    request<{ returned_context: Array<Record<string, unknown>> }>(
      `/v1/projects/${projectId}/rag/search?q=${encodeURIComponent(query)}`
    ),
  getProject: (projectId: string) => request<Project & Partial<ProjectSettings>>(`/v1/projects/${projectId}`),
  updateProject: (projectId: string, patch: Partial<Project & ProjectSettings>) =>
    request<Project & Partial<ProjectSettings>>(`/v1/projects/${projectId}`, { method: "PATCH", body: JSON.stringify(patch) }),
  getProjectSettings: (projectId: string) => request<ProjectSettings>(`/v1/projects/${projectId}/settings`),
  updateProjectSettings: (projectId: string, patch: Partial<ProjectSettings>) =>
    request<ProjectSettings>(`/v1/projects/${projectId}/settings`, { method: "PATCH", body: JSON.stringify(patch) }),
  listDirectory: (projectId: string) => request<DirectoryMember[]>(`/v1/projects/${projectId}/directory`),
  createDirectoryMember: (projectId: string, payload: Omit<DirectoryMember, "directory_id" | "project_id">) =>
    request<DirectoryMember>(`/v1/projects/${projectId}/directory`, { method: "POST", body: JSON.stringify(payload) }),
  updateDirectoryMember: (directoryId: string, patch: Partial<DirectoryMember>) =>
    request<DirectoryMember>(`/v1/directory/${directoryId}`, { method: "PATCH", body: JSON.stringify(patch) }),
  listDrawingSets: (projectId: string) => request<DrawingRevision[] | { items: DrawingRevision[] }>(`/v1/projects/${projectId}/drawing-sets`).then((response) => Array.isArray(response) ? response : response.items),
  activateRevision: (documentId: string, payload: { logical_key?: string; sheet_number?: string; issue_date?: string; discipline?: string }) =>
    request<DrawingRevision>(`/v1/documents/${documentId}/activate`, { method: "POST", body: JSON.stringify(payload) }),
  listEvidence: (projectId: string) => request<FieldEvidenceRecord[]>(`/v1/projects/${projectId}/evidence`),
  syncEvidence: (projectId: string, payload: {
    client_capture_id: string;
    media_id?: string;
    media_type: "photo" | "video" | "audio" | "measurement";
    filename: string;
    mime: string;
    uri?: string;
    hash?: string;
    sha256?: string;
    content_base64?: string;
    captured_at: string;
    author: string;
    location: Record<string, unknown>;
    location_method: string;
    metadata: Record<string, unknown>;
    observation?: Record<string, unknown> | string;
    quality?: Record<string, unknown>;
    sufficiency?: "unreviewed" | "sufficient" | "insufficient";
  }) => request<FieldEvidenceRecord>(`/v1/projects/${projectId}/evidence/sync`, { method: "POST", body: JSON.stringify({ ...payload, project_id: projectId }) }),
  updateEvidence: (evidenceId: string, patch: Partial<FieldEvidenceRecord>) =>
    request<FieldEvidenceRecord>(`/v1/evidence/${evidenceId}`, { method: "PATCH", body: JSON.stringify(patch) }),
  updateEvidenceLocation: (evidenceId: string, location: Record<string, unknown>, locationMethod = "manual") =>
    request<FieldEvidenceRecord>(`/v1/evidence/${evidenceId}/location`, { method: "PATCH", body: JSON.stringify({ location, location_method: locationMethod }) }),
  linkEvidence: (evidenceId: string, issueId: string, relevance = "supports", annotation = "") =>
    request<Record<string, unknown>>(`/v1/evidence/${evidenceId}/link`, { method: "POST", body: JSON.stringify({ issue_id: issueId, relevance, annotation }) }),
  createIssue: (projectId: string, payload: Record<string, unknown>) =>
    request<Issue>(`/v1/projects/${projectId}/issues`, { method: "POST", body: JSON.stringify({ ...payload, project_id: projectId }) }),
  reviewIssue: (issueId: string, payload: { decision: "approve" | "reject" | "request_evidence"; reviewer: string; reason?: string; reason_code?: string; evidence_gaps?: Array<Record<string, unknown>> }) =>
    request<ReviewRecord>(`/v1/issues/${issueId}/reviews`, { method: "POST", body: JSON.stringify(payload) }),
  listIssueReviews: (issueId: string) => request<ReviewRecord[]>(`/v1/issues/${issueId}/reviews`),
  requestIssueEvidence: (issueId: string, payload: { requested_by: string; reason: string; evidence_gaps?: Array<Record<string, unknown>>; recipient?: string }) =>
    request<Record<string, unknown>>(`/v1/issues/${issueId}/request-evidence`, { method: "POST", body: JSON.stringify(payload) }),
  listReports: (projectId: string) => request<ReportRecord[]>(`/v1/projects/${projectId}/reports`),
  listReportVersions: (reportId: string) => request<ReportVersion[]>(`/v1/reports/${reportId}/versions`),
  exportReport: (reportId: string, payload: { recipients?: string[]; external_id?: string } = {}) =>
    request<ReportVersion | ReportRecord>(`/v1/reports/${reportId}/export`, { method: "POST", body: JSON.stringify(payload) }),
  universalSearch: (projectId: string, query: string, options?: { scope?: "project" | "organization"; historical?: boolean }) => {
    const params = new URLSearchParams({ project_id: projectId, q: query, include_historical: String(Boolean(options?.historical)) });
    return request<SearchResult[] | { results: SearchResult[] }>(`/v1/search?${params.toString()}`).then((response) => Array.isArray(response) ? response : response.results);
  },
  listNotifications: (projectId: string) => request<ProjectNotification[]>(`/v1/projects/${projectId}/notifications`)
};

export const demoIssue: Issue = {
  issue_id: "offline_demo",
  project_id: "offline",
  type: "coverage_check",
  discipline: "electrical",
  severity: "major",
  room: "Main Floor",
  status: "review_ready",
  confidence: 0.74,
  title: "AFCI outlet coverage below E1.1 requirement",
  description: "Candidate issue generated from the Cooper Residence E1.1 electrical plan.",
  recommended_action: "Verify AFCI receptacle coverage and document correction before rough-in approval.",
  assignee: "Field PM",
  due_date: "",
  subcontractor: "",
  requirement: { source: "E1.1", text: "Electrical legend and notes call for AFCI/GFCI outlet coverage by location." },
  observation: { media_id: "field_verification_pending", frame_ts: 0, text: "Field verification is required." },
  plan_location: { sheet_id: "E1.1", x: 0.64, y: 0.63, bbox: [0.58, 0.57, 0.7, 0.69] },
  rfi_draft: "Please confirm AFCI/GFCI outlet requirements for the marked E1.1 location before rough-in signoff.",
  spatial_context: {
    spatial_evidence_id: "offline_spatial",
    room_graph_id: "main_floor",
    snapshot_uri: "spatial/offline/design.glb",
    spatial_note: "Offline sample spatial evidence is display-only.",
    alignment_confidence: 0.74,
    geometry_confidence: 0.69,
    geometry_features: {
      field_coverage_ratio: 0.62,
      required_count: 2,
      observed_count: 1,
      needs_more_evidence: false
    }
  },
  evidence: [
    {
      evidence_id: "offline_evd",
      evidence_type: "sheet",
      ref_id: "e11_electrical_plan",
      r2_key: "asset://plans/utah-e11-electrical-plan.jpg",
      page: 8,
      bbox: [0.58, 0.57, 0.7, 0.69],
      frame_ts: 0,
      label: "E1.1 electrical plan"
    }
  ]
};
