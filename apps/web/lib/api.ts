export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api";

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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    cache: "no-store"
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
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
    const response = await fetch(uploadUrl, { method: "POST", body: form });
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
  createReport: (projectId: string, report_type: "punch" | "co_evidence" | "rfi", format: "pdf" | "csv") =>
    request<{ report_id: string; report_type: string; format: string; download_url: string }>(
      `/v1/projects/${projectId}/reports`,
      { method: "POST", body: JSON.stringify({ report_type, format }) }
    ),
  getOverlay: (projectId: string) => request<Overlay>(`/v1/projects/${projectId}/plan-overlay`),
  ragSearch: (projectId: string, query: string) =>
    request<{ returned_context: Array<Record<string, unknown>> }>(
      `/v1/projects/${projectId}/rag/search?q=${encodeURIComponent(query)}`
    )
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
