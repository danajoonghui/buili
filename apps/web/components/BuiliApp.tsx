"use client";

import {
  AlertTriangle,
  Bell,
  Building2,
  Camera,
  ChevronDown,
  CircleHelp,
  Archive,
  ArrowRight,
  Box,
  Check,
  ClipboardCheck,
  FileDown,
  FileImage,
  FileQuestion,
  FileText,
  FolderPlus,
  Home,
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
  Volume2,
  RefreshCcw,
  RotateCcw,
  Ruler,
  Search,
  Settings,
  ShieldCheck,
  Upload,
  Users,
  Wifi,
  WifiOff,
  Clock3,
  Pencil,
  Plus,
  Trash2,
  ExternalLink,
  Layers,
  LockKeyhole,
  LogOut,
  Eye,
  EyeOff,
  ZoomIn,
  ZoomOut,
  Columns2,
  Save,
  Send,
  UserPlus,
  X
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  API_BASE,
  ApiError,
  AuthSession,
  DirectoryMember,
  DocumentAsset,
  DrawingRevision,
  FieldEvidenceRecord,
  Issue,
  Job,
  Observation,
  Overlay,
  Project,
  ProjectNotification,
  ProjectSettings,
  ReportRecord,
  ReviewRecord,
  SearchResult,
  SiteMediaAsset,
  TechnologyStatus
} from "@/lib/api";
import { InstallPrompt } from "@/components/InstallPrompt";
import { SpatialModelViewer } from "@/components/SpatialModelViewer";
import {
  CaptureDialog,
  IssueEditorDialog,
  IssueEditorValue,
  ProjectWizard,
  ProjectWizardValue,
  ReviewAction,
  ReviewDecisionDialog,
  UploadDialog,
  UploadMetadata
} from "@/components/WorkflowDialogs";
import {
  CaptureMetadata,
  listQueuedCaptures,
  makeCaptureId,
  QueuedCapture,
  removeQueuedCapture,
  saveQueuedCapture,
  updateQueuedCapture
} from "@/lib/offlineQueue";

type View = "overview" | "overlay" | "files" | "evidence" | "issues" | "review" | "reports" | "directory" | "settings";
type IssueFilter = "all" | "open" | "review" | "resolved";
type ReviewAuditEntry = ReviewRecord & { reason?: string; created_at: string };
type SearchScope = "project" | "organization";
type ViewerMode = "2d" | "3d" | "split" | "compare";
type ViewerTool = "select" | "pan" | "rotate" | "fit" | "model" | "grid" | "measure" | "markup";
const PLAN_IMAGE_SRC = "/plans/utah-e11-electrical-plan.jpg";
const PLAN_LABEL = "E1.1 Electrical Plans";
const FIELD_IMAGE_SRC = "/site-media/construction-site-electrical-work.jpg";
const PLAN2FIELD_3D_SRC = "/plan2field3d/auto_plan2field3d.png";
const PLAN2FIELD_MINIMAP_SRC = "/plan2field3d/auto_plan_crop.png";
const DEFAULT_RAG_QUERY = "AFCI GFCI smoke detector outlet electrical plan";
const PILOT_MEDIA_FALLBACKS: Record<string,string> = {
  "garage-east-wall-context.png": "/demo/northstar/evidence/garage-east-wall-context.png",
  "electrical-corridor-context.png": "/demo/northstar/evidence/garage-east-wall-context.png",
  "receptacle-rough-in-detail.png": "/demo/northstar/evidence/receptacle-rough-in-detail.png",
  "box-elevation-measurement.png": "/demo/northstar/evidence/box-elevation-measurement.png",
  "foreman-voice-note.mp3": "/demo/northstar/evidence/foreman-voice-note.mp3"
};

function mediaDownloadUrl(media: SiteMediaAsset) {
  if(media.download_url){
    const path=/^https?:\/\//.test(media.download_url)?new URL(media.download_url).pathname:media.download_url;
    if(path.startsWith(API_BASE))return path;
    return `${API_BASE}${path.startsWith("/")?"":"/"}${path}`;
  }
  return `${API_BASE}/v1/media/${encodeURIComponent(media.media_id)}/download`;
}

function mediaFallbackUrl(media: SiteMediaAsset) {
  return PILOT_MEDIA_FALLBACKS[media.filename] ?? FIELD_IMAGE_SRC;
}

function mediaThumbnailUrl(media: SiteMediaAsset) {
  const filename=media.filename==="electrical-corridor-context.png"?"garage-east-wall-context.png":media.filename;
  return PILOT_MEDIA_FALLBACKS[media.filename]
    ? `/demo/northstar/evidence/${filename.replace(/\.png$/i,"-thumb.webp")}`
    : mediaDownloadUrl(media);
}
const OFFLINE_SESSION_KEY = "buili.offline-session.v1";

function cacheOfflineSession(session: AuthSession) {
  if (typeof window === "undefined") return;
  const minimal: AuthSession = {
    user: session.user,
    projects: session.projects.slice(0, 1),
    expires_at: session.expires_at
  };
  window.localStorage.setItem(OFFLINE_SESSION_KEY, JSON.stringify(minimal));
}

function readOfflineSession() {
  if (typeof window === "undefined" || window.navigator.onLine) return null;
  try {
    const cached = JSON.parse(window.localStorage.getItem(OFFLINE_SESSION_KEY) ?? "null") as AuthSession | null;
    if (!cached?.user?.user_id || !cached.expires_at || new Date(cached.expires_at).getTime() <= Date.now()) return null;
    return cached;
  } catch {
    return null;
  }
}

const viewSegments: Record<View, string> = {
  overview: "overview",
  overlay: "drawings",
  files: "files",
  evidence: "evidence",
  issues: "issues",
  review: "review",
  reports: "reports",
  directory: "directory",
  settings: "settings"
};

function readRoute() {
  if (typeof window === "undefined") return { view: "overview" as View, projectId: "", issueId: "", filter: "all" as IssueFilter, query: "" };
  const parts = window.location.pathname.split("/").filter(Boolean);
  const segment = parts[0] === "projects" ? parts[2] : "";
  const view = (Object.entries(viewSegments).find(([, value]) => value === segment)?.[0] as View | undefined) ?? "overview";
  const params = new URLSearchParams(window.location.search);
  const requestedFilter = params.get("filter") as IssueFilter | null;
  return {
    view,
    projectId: parts[0] === "projects" && parts[1] && parts[1] !== "new" ? decodeURIComponent(parts[1]) : "",
    issueId: params.get("issue") ?? "",
    filter: requestedFilter && ["all", "open", "review", "resolved"].includes(requestedFilter) ? requestedFilter : "all",
    query: params.get("q") ?? ""
  };
}

function makeRoute(projectId: string | undefined, view: View, issueId = "", filter: IssueFilter = "all", query = "") {
  if (!projectId) return "/projects";
  const params = new URLSearchParams();
  if ((view === "issues" || view === "review") && issueId) params.set("issue", issueId);
  if ((view === "issues" || view === "review") && filter !== "all") params.set("filter", filter);
  if (query.trim()) params.set("q", query.trim());
  const suffix = params.size ? `?${params.toString()}` : "";
  return `/projects/${encodeURIComponent(projectId)}/${viewSegments[view]}${suffix}`;
}

const views: Array<{ id: View; label: string; icon: React.ComponentType<{ size?: number }> }> = [
  { id: "overview", label: "Overview", icon: Home },
  { id: "overlay", label: "Drawings & 3D", icon: Map },
  { id: "files", label: "Files & revisions", icon: FileText },
  { id: "evidence", label: "Field evidence", icon: Camera },
  { id: "issues", label: "Issues", icon: ListChecks },
  { id: "review", label: "Review queue", icon: ClipboardCheck },
  { id: "reports", label: "Reports", icon: FileDown },
  { id: "directory", label: "Directory", icon: Users },
  { id: "settings", label: "Project settings", icon: Settings }
];

function selectInitialProject(incoming: Project[]) {
  return (
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
    type: "location_mismatch",
    discipline: "electrical",
    severity: "major",
    room: "Main Floor · Garage · East wall near entry door",
    status: "review_ready",
    confidence: 0.88,
    title: "Garage GFCI box elevation below E1.1 minimum",
    description: "The measured garage GFCI box centerline is 12 inches AFF; current E1.1 Electrical Note 3 requires a minimum of 18 inches AFF.",
    recommended_action: "Confirm design intent before close-in and raise the box or record an accepted deviation.",
    assignee: "Jordan Davis",
    due_date: "2026-07-11",
    subcontractor: "Delta Electrical",
    requirement: {
      text: "E1.1 Electrical Note 3 requires garage outlets to be a minimum of 18 inches AFF.",
      source: "E1.1"
    },
    observation: {
      text: "Tape measurement and foreman voice note record the box centerline at 12 inches AFF.",
      media_id: "field-photo-01"
    },
    plan_location: { sheet_id: "E1.1", code: "E1.1", floor:"Main Floor", room:"Garage", x: 61, y: 43 },
    rfi_draft: "Please confirm whether Delta Electrical should raise the garage east-wall GFCI box from the observed 12-inch AFF centerline to the E1.1 minimum 18-inch AFF before close-in.",
    evidence: [
      {
        evidence_id: "demo-evidence-1",
        evidence_type: "plan_pin",
        ref_id: "E1.1",
        r2_key: PLAN2FIELD_3D_SRC,
        page: 1,
        bbox: [0.49, 0.29, 0.55, 0.36],
        frame_ts: 0,
        label: "E1.1 Electrical Note 3 source pin"
      },
      {
        evidence_id: "demo-evidence-2",
        evidence_type: "field_photo",
        ref_id: "field-photo-01",
        r2_key: "/demo/northstar/evidence/box-elevation-measurement.png",
        page: 0,
        bbox: [0.31, 0.28, 0.52, 0.58],
        frame_ts: 0,
        label: "Garage GFCI box centerline measurement"
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
      spatial_note: "E1.1 source geometry and the garage evidence location are aligned for reviewer navigation.",
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
    doc_id: "demo-doc-e11",
    project_id: "demo-plan2field",
    type: "plan",
    filename: "Cooper-Residence-E1.1-Electrical.pdf",
    mime: "application/pdf",
    r2_key: PLAN2FIELD_MINIMAP_SRC,
    hash: "demo-e11",
    revision: "E1.1",
    parsed_status: "parsed",
    size: 0,
    metadata_json: { sheet_id: "E1.1", state:"current" }
  }
];

const DEMO_MEDIA_ASSETS: SiteMediaAsset[] = [
  {
    media_id: "field-photo-01",
    project_id: "demo-plan2field",
    filename: "garage-east-wall-context.png",
    mime: "image/png",
    r2_key: "/demo/northstar/evidence/garage-east-wall-context.png",
    hash: "demo-media-1",
    metadata_json: { type: "field_evidence", label:"Garage east-wall context near entry door", captured_by:"Mike Torres, Electrical Foreman", location:{floor:"Main Floor",room:"Garage"} }
  },
  {
    media_id: "field-photo-02",
    project_id: "demo-plan2field",
    filename: "receptacle-rough-in-detail.png",
    mime: "image/png",
    r2_key: "/demo/northstar/evidence/receptacle-rough-in-detail.png",
    hash: "demo-media-2",
    metadata_json: { type: "field_evidence", label:"Garage GFCI rough-in detail", captured_by:"Mike Torres, Electrical Foreman", location:{floor:"Main Floor",room:"Garage"} }
  },
  {
    media_id: "field-photo-03",
    project_id: "demo-plan2field",
    filename: "box-elevation-measurement.png",
    mime: "image/png",
    r2_key: "/demo/northstar/evidence/box-elevation-measurement.png",
    hash: "demo-media-3",
    metadata_json: { type: "field_evidence", label:"Garage GFCI box centerline measurement", captured_by:"Mike Torres, Electrical Foreman", location:{floor:"Main Floor",room:"Garage"} }
  },
  {
    media_id: "field-audio-01",
    project_id: "demo-plan2field",
    filename: "foreman-voice-note.mp3",
    mime: "audio/mpeg",
    r2_key: "/demo/northstar/evidence/foreman-voice-note.mp3",
    hash: "demo-media-4",
    metadata_json: { type:"field_evidence", label:"Mike Torres foreman voice note", captured_by:"Mike Torres, Electrical Foreman", duration_seconds:32, captions_uri:"/demo/northstar/evidence/foreman-voice-note.vtt", transcript:"The garage east-wall GFCI box measures 12 inches AFF. E1.1 Electrical Note 3 calls for a minimum of 18 inches in garages. Please confirm whether Delta Electrical should raise it before close-in.", location:{floor:"Main Floor",room:"Garage"} }
  }
];

const DEMO_OVERLAY: Overlay = {
  project_id: "demo-plan2field",
  sheets: [{ id: "E1.1", title: "E1.1 Electrical Plan · Current" }],
  pins: [
    { id: "demo-e11-afci", label: "E1.1", severity: "major", room: "Main Floor · Garage · East wall near entry door", x: 0.61, y: 0.43, confidence: 0.88 }
  ],
  regions: []
};

export function BuiliApp() {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [authState, setAuthState] = useState<"loading" | "anonymous" | "authenticated">("loading");

  useEffect(() => {
    let active = true;
    api.me().then((value) => {
      if (!active) return;
      cacheOfflineSession(value);
      setSession(value);
      setAuthState("authenticated");
    }).catch((error: unknown) => {
      if (!active) return;
      const offline = readOfflineSession();
      if (offline && !(error instanceof ApiError)) {
        setSession(offline);
        setAuthState("authenticated");
        return;
      }
      setSession(null);
      setAuthState(error instanceof ApiError && error.status === 401 ? "anonymous" : "anonymous");
    });
    const expire = () => {
      setSession(null);
      setAuthState("anonymous");
    };
    window.addEventListener("buili:unauthorized", expire);
    return () => {
      active = false;
      window.removeEventListener("buili:unauthorized", expire);
    };
  }, []);

  if (authState === "loading") return <AuthLoading />;
  if (authState === "anonymous" || !session) {
    return <LoginView onAuthenticated={(value) => { setSession(value); setAuthState("authenticated"); }} />;
  }

  return <WorkspaceApp session={session} onSignedOut={() => { setSession(null); setAuthState("anonymous"); }} />;
}

function AuthLoading() {
  return <main className="auth-loading" aria-label="Loading secure workspace"><img src="/brand/buili-mark.png" alt=""/><span/><p>Opening your secure workspace</p></main>;
}

function LoginView({ onAuthenticated }: { onAuthenticated: (session: AuthSession) => void }) {
  const [email, setEmail] = useState(() => typeof window === "undefined" ? "" : window.localStorage.getItem("buili.login.email") ?? "");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function signIn(event: FormEvent) {
    event.preventDefault();
    if (!email.trim() || !password) {
      setError("Enter your work email and password.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const next = await api.login({ email: email.trim().toLowerCase(), password, remember_me: remember });
      cacheOfflineSession(next);
      if (remember) window.localStorage.setItem("buili.login.email", email.trim().toLowerCase());
      else window.localStorage.removeItem("buili.login.email");
      onAuthenticated(next);
    } catch (reason) {
      setError(reason instanceof ApiError && reason.status === 401
        ? "The email or password is incorrect. Try again or contact your workspace administrator."
        : "Secure sign-in is temporarily unavailable. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return <main className="login-shell">
    <section className="login-story" aria-labelledby="login-story-title">
      <a className="login-brand" href="/" aria-label="Buili home"><img src="/brand/buili-mark.png" alt=""/><span><b>BUILI</b><small>Verification intelligence</small></span></a>
      <div className="login-message"><p>FIELD DECISIONS, WITH PROOF</p><h1 id="login-story-title">Turn one drawing into a defensible field decision.</h1><span>Buili connects the current source, the observed condition, and the human decision—across 2D and lightweight 3D context.</span></div>
      <ol className="login-proof">
        <li><span>01</span><div><b>Control the source</b><small>Only the current drawing revision can support a new package.</small></div></li>
        <li><span>02</span><div><b>See the difference</b><small>Move from plan pin to field evidence without losing location.</small></div></li>
        <li><span>03</span><div><b>Issue with confidence</b><small>Human-reviewed RFI and punch outputs preserve their evidence snapshot.</small></div></li>
      </ol>
      <p className="login-footnote">A verification and decision layer for teams that already manage delivery in their system of record.</p>
    </section>
    <section className="login-entry">
      <form aria-label="Sign in to Buili" onSubmit={signIn}>
        <header><p>SECURE WORKSPACE</p><h2>Welcome back</h2><span>Sign in to your company workspace.</span></header>
        <div className="login-field"><label htmlFor="login-email">Work email</label><input id="login-email" autoFocus type="email" autoComplete="username" value={email} onChange={event=>setEmail(event.target.value)} placeholder="name@company.com" required/></div>
        <div className="login-field"><label htmlFor="login-password">Password</label><span className="password-field"><input id="login-password" type={showPassword?"text":"password"} autoComplete="current-password" value={password} onChange={event=>setPassword(event.target.value)} required/><button type="button" onClick={()=>setShowPassword(value=>!value)} aria-label={showPassword?"Hide password":"Show password"}>{showPassword?<EyeOff size={17}/>:<Eye size={17}/>}</button></span></div>
        <div className="login-options"><label><input type="checkbox" checked={remember} onChange={event=>setRemember(event.target.checked)}/> Keep me signed in</label><span>Access is managed by your workspace administrator.</span></div>
        {error?<p className="login-error" role="alert"><AlertTriangle size={16}/>{error}</p>:null}
        <button className="login-submit" type="submit" disabled={busy}>{busy?<Loader2 className="spin" size={18}/>:<LockKeyhole size={18}/>} {busy?"Signing in…":"Sign in securely"}</button>
        <p className="login-security"><ShieldCheck size={16}/> Your password is sent only over the encrypted session and is never stored in this browser.</p>
      </form>
    </section>
  </main>;
}

function WorkspaceApp({ session, onSignedOut }: { session: AuthSession; onSignedOut: () => void }) {
  // URL state is applied after hydration so the first server/client render is
  // deterministic even for a directly opened project route.
  const initialRoute = useMemo(() => ({
    view: "overview" as View,
    projectId: "",
    issueId: "",
    filter: "all" as IssueFilter,
    query: ""
  }), []);
  const [view, setView] = useState<View>(initialRoute.view);
  const [issueFilter, setIssueFilter] = useState<IssueFilter>(initialRoute.filter);
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [issues, setIssues] = useState<Issue[]>([]);
  const [demoIssues, setDemoIssues] = useState<Issue[]>([DEMO_SPATIAL_ISSUES[0]]);
  const [documents, setDocuments] = useState<DocumentAsset[]>([]);
  const [drawingRevisions, setDrawingRevisions] = useState<DrawingRevision[]>([]);
  const [mediaAssets, setMediaAssets] = useState<SiteMediaAsset[]>([]);
  const [fieldEvidence, setFieldEvidence] = useState<FieldEvidenceRecord[]>([]);
  const [observations, setObservations] = useState<Observation[]>([]);
  const [technologyStatus, setTechnologyStatus] = useState<TechnologyStatus[]>([]);
  const [directory, setDirectory] = useState<DirectoryMember[]>([]);
  const [projectSettings, setProjectSettings] = useState<ProjectSettings | null>(null);
  const [selectedIssueId, setSelectedIssueId] = useState<string>(initialRoute.issueId);
  const [job, setJob] = useState<Job | null>(null);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [query, setQuery] = useState(DEFAULT_RAG_QUERY);
  const [ragResults, setRagResults] = useState<Array<Record<string, unknown>>>([]);
  const [rfi, setRfi] = useState("");
  const [reportUrl, setReportUrl] = useState("");
  const [reports, setReports] = useState<ReportRecord[]>([]);
  const [reviewHistory, setReviewHistory] = useState<ReviewAuditEntry[]>([]);
  const [notifications, setNotifications] = useState<ProjectNotification[]>([]);
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(Boolean(initialRoute.query));
  const [globalQuery, setGlobalQuery] = useState(initialRoute.query);
  const [searchScope, setSearchScope] = useState<SearchScope>("project");
  const [includeHistorical, setIncludeHistorical] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchBusy, setSearchBusy] = useState(false);
  const [projectWizardOpen, setProjectWizardOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [captureOpen, setCaptureOpen] = useState(false);
  const [issueEditor, setIssueEditor] = useState<{ open: boolean; issueId?: string }>({ open: false });
  const [reviewDecision, setReviewDecision] = useState<{ action: ReviewAction; issueId: string } | null>(null);
  const [queuedCaptures, setQueuedCaptures] = useState<QueuedCapture[]>([]);
  const [online, setOnline] = useState(true);
  const [syncingQueue, setSyncingQueue] = useState(false);
  const [busy, setBusy] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [apiError, setApiError] = useState("");
  const [notice, setNotice] = useState("");
  const navigationSource = useRef<"ui" | "popstate" | "initial">("initial");
  const wasOnline = useRef(online);
  const demoMode = !project || apiError.startsWith("API offline");
  const displayIssues = issues.length ? issues : demoMode ? demoIssues : [];
  const displayDocuments = documents.length ? documents : demoMode ? DEMO_DOCUMENTS : [];
  const displayMediaAssets = mediaAssets.length ? mediaAssets : demoMode ? DEMO_MEDIA_ASSETS : [];
  const displayOverlay = overlay ?? (demoMode ? DEMO_OVERLAY : null);

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
      issueResult,
      overlayResult,
      jobResult,
      documentResult,
      mediaResult,
      observationResult,
      technologyResult,
      ragResult,
      revisionResult,
      evidenceResult,
      directoryResult,
      settingsResult,
      reportResult,
      notificationResult
    ] = await Promise.allSettled([
      api.listIssues(projectId),
      api.getOverlay(projectId),
      api.latestJob(projectId),
      api.listDocuments(projectId),
      api.listMedia(projectId),
      api.listObservations(projectId),
      api.technologyStatus(projectId),
      api.ragSearch(projectId, DEFAULT_RAG_QUERY),
      api.listDrawingSets(projectId),
      api.listEvidence(projectId),
      api.listDirectory(projectId),
      api.getProjectSettings(projectId),
      api.listReports(projectId),
      api.listNotifications(projectId)
    ]);
    const nextIssues = issueResult.status === "fulfilled" ? issueResult.value : [];
    setIssues(nextIssues);
    setDocuments(documentResult.status === "fulfilled" ? documentResult.value : []);
    setMediaAssets(mediaResult.status === "fulfilled" ? mediaResult.value : []);
    setObservations(observationResult.status === "fulfilled" ? observationResult.value : []);
    setTechnologyStatus(technologyResult.status === "fulfilled" ? technologyResult.value : []);
    setRagResults(ragResult.status === "fulfilled" ? ragResult.value.returned_context : []);
    setJob(jobResult.status === "fulfilled" ? jobResult.value : null);
    setDrawingRevisions(revisionResult.status === "fulfilled" ? revisionResult.value : []);
    setFieldEvidence(evidenceResult.status === "fulfilled" ? evidenceResult.value : []);
    setDirectory(directoryResult.status === "fulfilled" ? directoryResult.value : []);
    setProjectSettings(settingsResult.status === "fulfilled" ? settingsResult.value : null);
    setReports(reportResult.status === "fulfilled" ? reportResult.value : []);
    setNotifications(notificationResult.status === "fulfilled" ? notificationResult.value : []);
    setSelectedIssueId((current) =>
      nextIssues.some((issue) => issue.issue_id === current) ? current : nextIssues[0]?.issue_id || ""
    );
    setOverlay(overlayResult.status === "fulfilled" ? overlayResult.value : null);
    try {
      const saved = JSON.parse(window.localStorage.getItem(`buili.review-history.${projectId}`) ?? "[]") as ReviewAuditEntry[];
      const serverResults = await Promise.allSettled(nextIssues.slice(0, 50).map((item) => api.listIssueReviews(item.issue_id)));
      const server = serverResults.flatMap((result) => result.status === "fulfilled" ? result.value : []).map((item) => ({ ...item, created_at: item.created_at ?? new Date().toISOString() }));
      const merged = [...server, ...(Array.isArray(saved) ? saved : [])].filter((item,index,array)=>array.findIndex(candidate=>candidate.review_id===item.review_id)===index).sort((a,b)=>b.created_at.localeCompare(a.created_at));
      setReviewHistory(merged);
    } catch {
      setReviewHistory([]);
    }
    try {
      setQueuedCaptures(await listQueuedCaptures(projectId));
    } catch {
      setQueuedCaptures([]);
    }
    if (issueResult.status === "rejected" && documentResult.status === "rejected") throw issueResult.reason;
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
    const requestedRoute = readRoute();
    setView(requestedRoute.view);
    setIssueFilter(requestedRoute.filter);
    setSelectedIssueId(requestedRoute.issueId);
    setGlobalQuery(requestedRoute.query);
    setSearchOpen(Boolean(requestedRoute.query));
    (session.projects.length ? Promise.resolve(session.projects) : api.listProjects())
      .then(async (incoming) => {
        setProjects(incoming);
        const initial = incoming.find((item) => item.project_id === requestedRoute.projectId) ?? selectInitialProject(incoming);
        setProject(initial);
        if (initial) {
          window.localStorage.setItem("buili.last-project", JSON.stringify(initial));
          await loadProjectData(initial.project_id);
        }
      })
      .catch(async (error: Error) => {
        setApiError(`API offline: ${error.message}`);
        setIssues([]);
        try {
          const cached = JSON.parse(window.localStorage.getItem("buili.last-project") ?? "null") as Project | null;
          if (cached?.project_id) {
            setProjects([cached]);
            setProject(cached);
            setQueuedCaptures(await listQueuedCaptures(cached.project_id));
          }
        } catch {
          // The app shell remains usable even when no recent project is cached.
        }
      });
  }, [loadProjectData, session.projects]);

  const navigate = useCallback((nextView: View, issueId?: string) => {
    const nextIssueId = issueId ?? selectedIssueId;
    setView(nextView);
    if (issueId !== undefined) setSelectedIssueId(issueId);
    if (typeof window !== "undefined") {
      const nextRoute = makeRoute(project?.project_id, nextView, nextIssueId, issueFilter, globalQuery);
      const current = `${window.location.pathname}${window.location.search}`;
      if (current !== nextRoute) window.history.pushState({ view: nextView }, "", nextRoute);
    }
  }, [globalQuery, issueFilter, project?.project_id, selectedIssueId]);

  const chooseProject = useCallback(async (next: Project | null, nextView: View = view) => {
    setProject(next);
    setApiError("");
    if (next) {
      window.localStorage.setItem("buili.last-project", JSON.stringify(next));
      await loadProjectData(next.project_id);
      if (typeof window !== "undefined") window.history.pushState({}, "", makeRoute(next.project_id, nextView));
    } else if (typeof window !== "undefined") {
      window.history.pushState({}, "", "/projects");
    }
  }, [loadProjectData, view]);

  const applyIssueFilter = useCallback((nextFilter: IssueFilter) => {
    setIssueFilter(nextFilter);
    if (typeof window !== "undefined") {
      window.history.replaceState({}, "", makeRoute(project?.project_id, view, selectedIssueId, nextFilter, globalQuery));
    }
  }, [globalQuery, project?.project_id, selectedIssueId, view]);

  useEffect(() => {
    const onPopState = () => {
      const route = readRoute();
      navigationSource.current = "popstate";
      setView(route.view);
      setIssueFilter(route.filter);
      setSelectedIssueId(route.issueId);
      setGlobalQuery(route.query);
      setSearchOpen(Boolean(route.query));
      const requested = projects.find((item) => item.project_id === route.projectId);
      if (requested && requested.project_id !== project?.project_id) {
        setProject(requested);
        loadProjectData(requested.project_id).catch((error) => surfaceError(error, "Could not open project"));
      }
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [loadProjectData, project?.project_id, projects, surfaceError]);

  useEffect(() => {
    if (!project) return;
    const nextRoute = makeRoute(project.project_id, view, selectedIssueId, issueFilter, searchOpen ? globalQuery : "");
    const current = `${window.location.pathname}${window.location.search}`;
    if (current !== nextRoute && navigationSource.current === "initial") window.history.replaceState({}, "", nextRoute);
    navigationSource.current = "initial";
  }, [globalQuery, issueFilter, project, searchOpen, selectedIssueId, view]);

  useEffect(() => {
    const refreshOnlineState = () => setOnline(navigator.onLine);
    refreshOnlineState();
    window.addEventListener("online", refreshOnlineState);
    window.addEventListener("offline", refreshOnlineState);
    return () => {
      window.removeEventListener("online", refreshOnlineState);
      window.removeEventListener("offline", refreshOnlineState);
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setSearchOpen(true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

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

  async function createProject(value: ProjectWizardValue) {
    setBusy(true);
    setApiError("");
    try {
      const created = await api.createProject({
        name: value.name.trim(),
        address: value.address.trim(),
        project_type: value.projectType
      });
      await Promise.allSettled([
        api.updateProject(created.project_id, {
          client: value.client,
          timezone: value.timezone,
          unit_system: value.units,
          status: "active"
        }),
        api.updateProjectSettings(created.project_id, {
          timezone: value.timezone,
          unit_system: value.units,
          settings: {
            building: value.building,
            floors: value.floors.split("\n").map((item) => item.trim()).filter(Boolean),
            grid: value.grid,
            spatial: { scale: value.scale, north: value.north, floor_height: value.floorHeight, origin: value.origin }
          },
          workflow: { review_route: value.reviewRoute, report_template: value.reportTemplate, second_reviewer_high_risk: true }
        }),
        api.createDirectoryMember(created.project_id, {
          person_name: value.approverName,
          email: value.approverEmail,
          company: value.company,
          role: "project_manager",
          trade: value.trade,
          status: "active",
          notification: { in_app: true, email: true },
          access_expires_at: null
        })
      ]);
      setProjects((current) => [created, ...current]);
      setProject(created);
      setIssues([]);
      setOverlay(null);
      setSelectedIssueId("");
      if (value.files.length) {
        await uploadFiles(value.files, {
          documentType: "plan",
          revision: "A",
          issueDate: new Date().toISOString().slice(0, 10),
          discipline: "general",
          setStatus: "current"
        }, created);
      }
      setProjectWizardOpen(false);
      setNotice(`Project activated: ${created.name}. ${value.files.length ? `${value.files.length} intake file(s) received.` : "Upload the current drawing set to begin verification."}`);
      await loadProjectData(created.project_id);
      window.history.pushState({}, "", makeRoute(created.project_id, "overview"));
    } catch (error) {
      surfaceError(error, "Project creation failed");
      throw error;
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

  function openReviewDecision(issueId: string, action: ReviewAction) {
    setReviewDecision({ issueId, action });
  }

  async function confirmReviewDecision(reasonCode: string, note: string) {
    if (!reviewDecision) return;
    const issue = displayIssues.find((item) => item.issue_id === reviewDecision.issueId);
    if (!issue) return;
    setBusy(true);
    const status = reviewDecision.action === "approve" ? "approved" : reviewDecision.action === "reject" ? "rejected_false_positive" : "needs_more_evidence";
    try {
      let record: ReviewAuditEntry;
      if (issue.issue_id.startsWith("demo-")) {
        setDemoIssues((current) => current.map((item) => item.issue_id === issue.issue_id ? { ...item, status } : item));
        record = {
          review_id: `local-review-${Date.now()}`,
          issue_id: issue.issue_id,
          reviewer: session.user.name,
          decision: reviewDecision.action,
          reason_code: reasonCode,
          reason: note,
          created_at: new Date().toISOString()
        };
      } else {
        const saved = await api.reviewIssue(issue.issue_id, {
          decision: reviewDecision.action,
          reviewer: session.user.name,
          reason_code: reasonCode,
          reason: note,
          evidence_gaps: reviewDecision.action === "request_evidence" ? [{ type: reasonCode, instruction: note }] : []
        });
        record = { ...saved, reason: saved.reason ?? note, created_at: saved.created_at ?? new Date().toISOString() };
        await loadProjectData(issue.project_id);
      }
      const nextHistory = [record, ...reviewHistory];
      setReviewHistory(nextHistory);
      if (project) window.localStorage.setItem(`buili.review-history.${project.project_id}`, JSON.stringify(nextHistory.slice(0, 100)));
      setReviewDecision(null);
      setNotice(reviewDecision.action === "approve" ? "Issue package approved and reviewer snapshot locked." : reviewDecision.action === "reject" ? "Draft rejected with an auditable reason." : "Evidence request sent to the assignee and added to review history.");
    } catch (error) {
      surfaceError(error, "Review decision failed");
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function saveIssue(value: IssueEditorValue) {
    setBusy(true);
    setApiError("");
    const existing = issueEditor.issueId ? displayIssues.find((item) => item.issue_id === issueEditor.issueId) : undefined;
    const payload = {
      title: value.title,
      type: value.type,
      discipline: value.discipline,
      severity: value.severity,
      room: value.room,
      description: value.description,
      recommended_action: value.recommendedAction,
      assignee: value.assignee,
      due_date: value.dueDate,
      requirement: { text: value.expected, source: value.source, route: value.route },
      observation: existing?.observation ?? { text: value.description, source: "manual issue" },
      plan_location: { ...(existing?.plan_location ?? {}), level: value.floor, sheet_id: value.source },
      expected_condition: value.expected,
      difference: value.description,
      recommended_route: value.route,
      priority: value.severity === "blocker" ? "critical" : value.severity === "major" ? "high" : value.severity === "minor" ? "medium" : "low",
      source_references: (() => {
        const needle = value.source.toLowerCase().replace(/\s+/g, "");
        const sourceDocument = documents.find((item) => {
          const candidate = `${item.filename} ${item.revision} ${String(item.metadata_json.sheet_id ?? "")}`.toLowerCase().replace(/\s+/g, "");
          return needle && (candidate.includes(needle) || needle.includes(item.filename.replace(/\.[^.]+$/, "").toLowerCase()));
        });
        return sourceDocument ? [{ document_id: sourceDocument.doc_id, page: 1, bbox: [], label: value.source }] : [];
      })(),
      evidence_gaps: value.route === "more_evidence" ? [{ type: "field_verification", status: "open" }] : []
    };
    try {
      if (existing?.issue_id.startsWith("demo-")) {
        setDemoIssues((current) => current.map((item) => item.issue_id === existing.issue_id ? { ...item, ...payload } as Issue : item));
      } else if (existing) {
        const updated = await api.updateIssue(existing.issue_id, payload as Partial<Issue>);
        setIssues((current) => current.map((item) => item.issue_id === existing.issue_id ? updated : item));
      } else if (project) {
        const created = await api.createIssue(project.project_id, payload);
        setIssues((current) => [created, ...current]);
        setSelectedIssueId(created.issue_id);
      } else {
        const created: Issue = {
          issue_id: `demo-manual-${Date.now()}`,
          project_id: "demo-plan2field",
          confidence: 1,
          status: "review_ready",
          subcontractor: "",
          rfi_draft: "",
          evidence: [],
          ...payload
        } as Issue;
        setDemoIssues((current) => [created, ...current]);
        setSelectedIssueId(created.issue_id);
      }
      setIssueEditor({ open: false });
      setNotice(existing ? "Issue fields and source context saved." : "Issue draft created and routed to review.");
    } catch (error) {
      surfaceError(error, "Issue save failed");
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function generateRfi(issueId: string) {
    const demoIssue = DEMO_SPATIAL_ISSUES.find((item) => item.issue_id === issueId);
    if (demoIssue) {
      setRfi(buildRfiPreview(demoIssue));
      setNotice("Demo RFI draft generated from Plan2Field-3D evidence.");
      navigate("reports", issueId);
      return;
    }
    try {
      const draft = await api.createRfi(issueId);
      setRfi(draft.markdown);
      setNotice("RFI draft generated.");
      navigate("reports", issueId);
    } catch (error) {
      surfaceError(error, "RFI generation failed");
    }
  }

  async function generateReport(type: "punch" | "co_evidence" | "rfi", format: "pdf" | "csv", issueIds: string[] = []) {
    if (!project) return;
    setBusy(true);
    try {
      const report = await api.createReport(project.project_id, type, format, issueIds);
      setReportUrl(report.download_url);
      const refreshed = await api.listReports(project.project_id).catch(() => reports);
      setReports(refreshed);
      setNotice(`${type.replace("_", " ")} ${format.toUpperCase()} report generated.`);
    } catch (error) {
      surfaceError(error, "Report generation failed");
    } finally {
      setBusy(false);
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

  async function uploadFiles(files: File[], metadata: UploadMetadata, projectOverride?: Project) {
    const targetProject = projectOverride ?? project;
    if (!targetProject || !files.length) return;
    setIsUploading(true);
    setApiError("");
    setNotice(`Uploading ${files.length} file${files.length === 1 ? "" : "s"}…`);
    try {
      for (const file of files) {
        const kind = metadata.documentType === "media" || file.type.startsWith("image/") || file.type.startsWith("video/") || file.type.startsWith("audio/") ? "media" : metadata.documentType === "submittal" ? "submittal" : "document";
        const presigned = await api.presignUpload({ project_id: targetProject.project_id, filename: file.name, mime: file.type || "application/octet-stream", size: file.size, kind });
        await api.uploadFile(presigned.upload_url, file);
        const completed = await api.completeUpload(presigned.complete_url, {
          document_type: metadata.documentType === "addendum" ? "other" : metadata.documentType === "media" ? "media" : metadata.documentType,
          revision: metadata.revision || "unclassified"
        });
        if (completed.document_id && metadata.setStatus === "current") {
          await api.activateRevision(completed.document_id, {
            logical_key: file.name.replace(/\s*rev(?:ision)?\s*[a-z0-9_-]+/i, "").toLowerCase(),
            sheet_number: file.name.replace(/\.[^.]+$/, ""),
            issue_date: metadata.issueDate,
            discipline: metadata.discipline
          }).catch(() => undefined);
        }
      }
      setUploadOpen(false);
      setNotice(`${files.length} file${files.length === 1 ? "" : "s"} uploaded and classified. Open revisions to verify the current set.`);
      await loadProjectData(targetProject.project_id);
    } catch (error) {
      surfaceError(error, "Upload failed");
      throw error;
    } finally {
      setIsUploading(false);
    }
  }

  async function syncCapture(capture: QueuedCapture) {
    if (!project || capture.projectId !== project.project_id || !navigator.onLine) return false;
    const syncing = await updateQueuedCapture(capture, { state: "syncing", attempts: capture.attempts + 1, error: "" });
    setQueuedCaptures((current) => current.map((item) => item.id === capture.id ? syncing : item));
    try {
      const bytes = await capture.blob.arrayBuffer();
      const [contentBase64, digest] = await Promise.all([arrayBufferToBase64(bytes), sha256Hex(bytes)]);
      const evidence = await api.syncEvidence(project.project_id, {
        client_capture_id: capture.id,
        media_type: capture.metadata.mediaType === "voice" ? "audio" : capture.metadata.mediaType,
        filename: capture.filename,
        mime: capture.mime,
        content_base64: contentBase64,
        sha256: digest,
        captured_at: capture.createdAt,
        author: session.user.name,
        location: { floor: capture.metadata.floor, room: capture.metadata.room, source: capture.metadata.source },
        location_method: capture.metadata.locationMethod,
        metadata: { note: capture.metadata.note, trade: capture.metadata.trade, intent: capture.metadata.intent, measurement: capture.metadata.measurement },
        observation: { text: capture.metadata.note, measurement: capture.metadata.measurement },
        quality: { original_preserved: true, context_present: capture.metadata.mediaType !== "measurement", source_linked: Boolean(capture.metadata.source) },
        sufficiency: capture.metadata.room && (capture.metadata.source || capture.metadata.intent === "observation") ? "sufficient" : "insufficient"
      });
      if (capture.metadata.intent === "issue") {
        const created = await api.createIssue(project.project_id, {
          title: capture.metadata.note || `${capture.metadata.trade} field observation`,
          type: "field_observation",
          discipline: capture.metadata.trade.toLowerCase(),
          severity: "minor",
          room: capture.metadata.room || "Unlinked",
          description: capture.metadata.note || "Field capture requires review.",
          recommended_action: "Review the original evidence and link a current contract requirement.",
          requirement: { source: capture.metadata.source, text: "Source confirmation required." },
          observation: { text: capture.metadata.note, evidence_id: evidence.evidence_id },
          plan_location: { level: capture.metadata.floor },
          recommended_route: "more_evidence",
          evidence_ids: [evidence.evidence_id],
          evidence_gaps: capture.metadata.source ? [] : [{ type: "current_source", status: "open" }]
        });
        setIssues((current) => [created, ...current]);
      }
      await removeQueuedCapture(capture.id);
      setQueuedCaptures((current) => current.filter((item) => item.id !== capture.id));
      setFieldEvidence((current) => [evidence, ...current.filter((item) => item.evidence_id !== evidence.evidence_id)]);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Capture sync failed";
      const failed = await updateQueuedCapture(syncing, { state: "failed", error: message });
      setQueuedCaptures((current) => current.map((item) => item.id === capture.id ? failed : item));
      return false;
    }
  }

  async function syncOfflineQueue() {
    if (!project || syncingQueue || !navigator.onLine) return;
    setSyncingQueue(true);
    const snapshot = await listQueuedCaptures(project.project_id).catch(() => queuedCaptures);
    let synced = 0;
    for (const capture of snapshot) if (await syncCapture(capture)) synced += 1;
    setSyncingQueue(false);
    setNotice(synced ? `${synced} locally saved capture${synced === 1 ? "" : "s"} synced successfully.` : snapshot.length ? "Capture queue is still local. Review the failed item and retry." : "Offline queue is already clear.");
  }

  async function saveCapture(file: File, metadata: CaptureMetadata) {
    if (!project) throw new Error("Choose a project before capturing evidence.");
    setBusy(true);
    const capture: QueuedCapture = { id: makeCaptureId(), projectId: project.project_id, filename: file.name, mime: file.type || "application/octet-stream", size: file.size, blob: file, metadata, createdAt: new Date().toISOString(), attempts: 0, state: "queued" };
    try {
      await saveQueuedCapture(capture);
      setQueuedCaptures((current) => [capture, ...current]);
      setCaptureOpen(false);
      navigate("evidence");
      setNotice("Saved locally. Buili will keep the original queued until the server confirms sync.");
      if (navigator.onLine) {
        setSyncingQueue(true);
        const synced = await syncCapture(capture);
        setSyncingQueue(false);
        setNotice(synced ? `${metadata.intent === "issue" ? "Issue draft and evidence" : "Evidence"} synced successfully.` : "Saved locally; sync will retry when the connection is stable.");
      }
    } catch (error) {
      surfaceError(error, "Local capture save failed");
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function searchUniversal(event?: FormEvent) {
    event?.preventDefault();
    if (!project || globalQuery.trim().length < 2) return;
    setSearchBusy(true);
    try {
      const results = await api.universalSearch(project.project_id, globalQuery.trim(), { scope: searchScope, historical: includeHistorical });
      setSearchResults(results);
      window.history.replaceState({}, "", makeRoute(project.project_id, view, selectedIssueId, issueFilter, globalQuery));
    } catch (error) {
      surfaceError(error, "Search failed");
    } finally {
      setSearchBusy(false);
    }
  }

  async function activateDocument(documentId: string, metadata: { sheetNumber: string; issueDate: string; discipline: string }) {
    if (!project) return;
    setBusy(true);
    try {
      await api.activateRevision(documentId, { logical_key: metadata.sheetNumber.toLowerCase(), sheet_number: metadata.sheetNumber, issue_date: metadata.issueDate, discipline: metadata.discipline });
      await loadProjectData(project.project_id);
      setNotice(`Revision ${metadata.sheetNumber} activated. Referencing open issues were routed to stale-source review.`);
    } catch (error) {
      surfaceError(error, "Revision activation failed");
    } finally { setBusy(false); }
  }

  async function inviteDirectoryMember(payload: Omit<DirectoryMember, "directory_id" | "project_id">) {
    if (!project) return;
    const member = await api.createDirectoryMember(project.project_id, payload);
    setDirectory((current) => [member, ...current]);
    setNotice(`Invitation created for ${member.person_name}.`);
  }

  async function changeDirectoryMember(id: string, patch: Partial<DirectoryMember>) {
    const member = await api.updateDirectoryMember(id, patch);
    setDirectory((current) => current.map((item) => item.directory_id === id ? member : item));
    setNotice(`${member.person_name}'s access was updated.`);
  }

  async function saveSettings(next: Partial<ProjectSettings>) {
    if (!project) return;
    const saved = await api.updateProjectSettings(project.project_id, next);
    setProjectSettings(saved);
    setNotice("Project settings and review identity saved.");
  }

  async function exportExistingReport(reportId: string) {
    setBusy(true);
    try {
      await api.exportReport(reportId);
      if (project) setReports(await api.listReports(project.project_id));
      setNotice("Report version issued with its source snapshot locked.");
    } catch (error) { surfaceError(error, "Report export failed"); }
    finally { setBusy(false); }
  }

  async function signOut() {
    setAccountOpen(false);
    try {
      await api.logout();
    } catch {
      // A local sign-out still removes access to cached identity if the network
      // is unavailable; the server session will expire independently.
    }
    window.localStorage.removeItem(OFFLINE_SESSION_KEY);
    onSignedOut();
    window.history.replaceState({}, "", "/");
  }

  useEffect(() => {
    const reconnected = online && !wasOnline.current;
    wasOnline.current = online;
    if (reconnected && queuedCaptures.length) {
      setNotice(`${queuedCaptures.length} locally saved capture${queuedCaptures.length === 1 ? " is" : "s are"} ready to sync.`);
    }
  }, [online]);

  return (
    <main className="app-shell">
      <aside className="side-nav" aria-label="Primary">
        <button className="brand-mark" type="button" aria-label="Go to overview" onClick={() => navigate("overview")}>
          <img src="/brand/buili-mark.png" alt="" />
          <span><strong>BUILI</strong><small>Verification intelligence</small></span>
        </button>
        <button className="sidebar-project" type="button" onClick={() => navigate("overview")} aria-label="Open current project overview"><span>{project?.name.slice(0,2).toUpperCase() ?? "BU"}</span><b>{project?.name ?? "Choose a project"}</b><ChevronDown size={15} /></button>
        <p className="nav-label">Workspace</p>
        <nav>
          {views.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "nav-item active" : "nav-item"}
              type="button"
              onClick={() => navigate(item.id)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot"><button type="button" onClick={() => setNotice("Shortcuts: ⌘/Ctrl K search · Esc closes a dialog. Captures save locally before sync.")}><CircleHelp size={18}/> Help center</button><button type="button" onClick={() => navigate("directory")}><Building2 size={18}/> {session.user.organization.name}</button><button className="user-chip" type="button" aria-label="Open account menu" aria-expanded={accountOpen} onClick={()=>setAccountOpen(value=>!value)}><span>{session.user.name.split(" ").map(part=>part[0]).join("").slice(0,2)}</span><div><b>{session.user.name}</b><small>{session.user.role.replaceAll("_"," ")}</small></div><ChevronDown size={14}/></button>{accountOpen?<div className="account-menu"><span>{session.user.email}</span><small>{session.user.organization.name}</small><button type="button" onClick={()=>void signOut()}><LogOut size={15}/> Sign out</button></div>:null}</div>
      </aside>

      <section className="workbench">
        <header className="top-bar">
          <div className="breadcrumbs"><span>Projects</span><b>/</b><span>{project?.name ?? "Project workspace"}</span><b>/</b><strong>{views.find(v => v.id === view)?.label}</strong></div>
          <div className="top-actions">
            <form className="global-search" onSubmit={(event)=>{setSearchOpen(true);void searchUniversal(event)}}><Search size={17}/><input aria-label="Universal search" value={globalQuery} onFocus={()=>setSearchOpen(true)} onChange={(event)=>setGlobalQuery(event.target.value)} placeholder="Search sheets, issues, evidence…"/><kbd>⌘ K</kbd></form>
            <span className={`connectivity ${online?"online":"offline"}`}>{online?<Wifi size={16}/>:<WifiOff size={16}/>}<span>{online?"Online":"Offline"}</span></span>
            <button className="top-icon" aria-label="Help" onClick={() => setNotice("Use universal search to jump directly to issues, drawings, evidence, and people.")}><CircleHelp size={19}/></button><button className={notifications.some(item=>!item.read_at)?"top-icon has-dot":"top-icon"} aria-label="Notifications" aria-expanded={notificationOpen} onClick={()=>setNotificationOpen(current=>!current)}><Bell size={19}/></button>
            {notificationOpen ? <div className="notification-popover" role="region" aria-label="Notifications"><header><b>Notifications</b><button type="button" onClick={()=>setNotificationOpen(false)} aria-label="Close notifications"><X size={16}/></button></header>{notifications.length?notifications.slice(0,6).map(item=><button type="button" key={item.notification_id} onClick={()=>{if(item.entity_type==="issue"&&item.entity_id)navigate("review",item.entity_id);setNotificationOpen(false)}}><span>{item.title}</span><small>{item.body}</small></button>):<p>No new notifications.</p>}</div>:null}
          </div>
        </header>

        <section className="project-strip utility-strip">
          <div className="project-picker">
            <label htmlFor="project">Project</label>
            <select
              id="project"
              value={project?.project_id ?? ""}
              onChange={async (event) => {
                const next = projects.find((item) => item.project_id === event.target.value) ?? null;
                await chooseProject(next);
              }}
            >
              {!projects.length ? <option value="">No projects yet</option> : null}
              {projects.map((item) => (
                <option key={item.project_id} value={item.project_id}>
                  {item.name}
                </option>
              ))}
            </select>
          </div>
          <button className="icon-button ghost new-project-trigger" type="button" onClick={()=>setProjectWizardOpen(true)}><FolderPlus size={18}/><span>New project</span></button>
          <button
            className="icon-button ghost"
            type="button"
            title="Upload"
            disabled={!project || isUploading}
            onClick={() => setUploadOpen(true)}
          >
            {isUploading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
            <span>{isUploading ? "Uploading" : "Upload"}</span>
          </button>
          <button className="icon-button primary" type="button" onClick={runAnalysis} disabled={!project || busy}>{busy ? <Loader2 className="spin" size={18}/> : <RefreshCcw size={18}/>}<span>Run verification</span></button>
        </section>

        {apiError ? <p className="api-error">{apiError}</p> : null}
        {notice ? <p className="status-note">{notice}</p> : null}

        {view === "overview" && <OverviewView userName={session.user.name} organizationName={session.user.organization.name} issues={displayIssues} documents={displayDocuments} job={job} onReview={() => navigate("review")} onCapture={() => setCaptureOpen(true)} onDrawings={() => navigate("overlay")} />}

        {view !== "overview" && <section className="kpi-band compact-kpis" aria-label="Project metrics">
          <Metric label="Issues ready" value={displayIssues.filter((issue) => issue.status === "review_ready").length} />
          <Metric label="Action needed" value={actionNeeded} />
          <Metric label="Evidence score" value={`${Math.round(avg(displayIssues.map((issue) => issue.confidence)) * 100)}%`} />
          <Metric label="Job" value={job ? `${job.progress}%` : issues.length ? "100%" : "Idle"} />
        </section>}

        {view === "files" && <FilesView documents={displayDocuments} revisions={drawingRevisions} staleIssueCount={displayIssues.filter(item=>item.status==="stale_source_review").length} busy={busy} onUpload={() => setUploadOpen(true)} onActivate={activateDocument} />}

        {view === "issues" && (
          <IssuesView issues={filteredIssues} selectedId={selectedIssue?.issue_id} filter={issueFilter} counts={issueCounts} onFilter={applyIssueFilter} onSelect={(id) => navigate("review", id)} onCreate={() => setIssueEditor({open:true})} />
        )}

        {view === "review" && (
          <Plan2FieldReview
            issue={selectedIssue}
            filteredIssues={filteredIssues}
            issueCounts={issueCounts}
            issueFilter={issueFilter}
            overlay={displayOverlay}
            documents={displayDocuments}
            mediaAssets={displayMediaAssets}
            reviewHistory={reviewHistory}
            onFilter={applyIssueFilter}
            onSelectIssue={(id)=>navigate("review",id)}
            onCreateIssue={() => setIssueEditor({open:true})}
            onEditIssue={(id)=>setIssueEditor({open:true,issueId:id})}
            onApprove={(id) => openReviewDecision(id, "approve")}
            onReject={(id) => openReviewDecision(id, "reject")}
            onNeedMore={(id) => openReviewDecision(id, "request_evidence")}
            onRfi={generateRfi}
          />
        )}

        {view === "evidence" && (
          <section className="standard-page evidence-page"><PageHeading eyebrow="FIELD CAPTURE" title="Field evidence" copy="Original media, confirmed location, sufficiency, and issue relationships." action={<button className="page-primary" type="button" onClick={()=>setCaptureOpen(true)}><Camera size={17}/> Capture evidence</button>}/><OfflineQueue captures={queuedCaptures} online={online} syncing={syncingQueue} onSync={syncOfflineQueue} onRemove={async(id)=>{await removeQueuedCapture(id);setQueuedCaptures(current=>current.filter(item=>item.id!==id))}}/><div className="content-grid evidence-grid">
            <EvidenceViewer
              issue={selectedIssue}
              documents={displayDocuments}
              mediaAssets={displayMediaAssets}
              fieldEvidence={fieldEvidence}
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
          </div></section>
        )}

        {view === "overlay" && <OverlayView projectId={project?.project_id??selectedIssue?.project_id??""} overlay={displayOverlay} issues={displayIssues} selectedIssueId={selectedIssueId} onSelectIssue={(id)=>{setSelectedIssueId(id)}} onOpenIssue={(id)=>navigate("review",id)} />}

        {view === "reports" && (
          <ReportsView issues={displayIssues} documents={displayDocuments} mediaAssets={displayMediaAssets} selectedIssue={selectedIssue} reports={reports} rfi={rfi} reportUrl={reportUrl} busy={busy} onGenerate={generateReport} onExport={exportExistingReport} onSelectIssue={(id)=>navigate("review",id)} />
        )}

        {view === "directory" && <DirectoryView members={directory} onInvite={inviteDirectoryMember} onChange={changeDirectoryMember} />}
        {view === "settings" && <SettingsView user={session.user} project={project} settings={projectSettings} technologyStatus={technologyStatus} onSave={saveSettings} />}
        <ProjectWizard open={projectWizardOpen} projects={projects} busy={busy} onClose={()=>setProjectWizardOpen(false)} onCreate={createProject}/>
        <UploadDialog open={uploadOpen} busy={isUploading} onClose={()=>setUploadOpen(false)} onUpload={uploadFiles}/>
        <CaptureDialog open={captureOpen} projectName={project?.name??"No project selected"} online={online} busy={busy} onClose={()=>setCaptureOpen(false)} onSave={saveCapture}/>
        <IssueEditorDialog open={issueEditor.open} issue={issueEditor.issueId?displayIssues.find(item=>item.issue_id===issueEditor.issueId):undefined} busy={busy} onClose={()=>setIssueEditor({open:false})} onSave={saveIssue}/>
        <ReviewDecisionDialog open={Boolean(reviewDecision)} issue={reviewDecision?displayIssues.find(item=>item.issue_id===reviewDecision.issueId):undefined} action={reviewDecision?.action??"approve"} busy={busy} onClose={()=>setReviewDecision(null)} onConfirm={confirmReviewDecision}/>
        {searchOpen ? <SearchPalette query={globalQuery} scope={searchScope} historical={includeHistorical} results={searchResults} busy={searchBusy} onQuery={setGlobalQuery} onScope={setSearchScope} onHistorical={setIncludeHistorical} onSearch={searchUniversal} onClose={()=>{setSearchOpen(false);setSearchResults([]);if(project)window.history.replaceState({},"",makeRoute(project.project_id,view,selectedIssueId,issueFilter))}} onOpen={(result)=>{setSearchOpen(false);if(result.type==="issue")navigate("review",result.id);else if(["drawing","document","requirement","specification"].includes(result.type))navigate("overlay");else if(result.type==="evidence")navigate("evidence");else if(result.type==="person"||result.type==="company")navigate("directory")}}/>:null}
      </section>

      {mobileMenuOpen?<div className="mobile-workspace-menu" role="dialog" aria-modal="true" aria-label="Workspace menu"><header><div><b>Workspace</b><small>{project?.name??"Current project"}</small></div><button type="button" onClick={()=>setMobileMenuOpen(false)} aria-label="Close workspace menu"><X size={18}/></button></header><nav>{views.filter(item=>!["overview","issues"].includes(item.id)).map(item=><button type="button" key={item.id} className={view===item.id?"active":""} onClick={()=>{navigate(item.id);setMobileMenuOpen(false)}}><item.icon size={18}/><span>{item.label}</span><ArrowRight size={15}/></button>)}</nav><button className="mobile-signout" type="button" onClick={()=>void signOut()}><LogOut size={16}/> Sign out</button></div>:null}
      <nav className="bottom-nav" aria-label="Mobile">
        <button className={view==="overview"?"bottom-item active":"bottom-item"} type="button" onClick={()=>navigate("overview")}><Home size={19}/><span>Home</span></button>
        <button className={captureOpen?"bottom-item capture active":"bottom-item capture"} type="button" onClick={()=>setCaptureOpen(true)}><Camera size={19}/><span>Capture</span></button>
        <button className={view==="issues"||view==="review"?"bottom-item active":"bottom-item"} type="button" onClick={()=>navigate("issues")}><ListChecks size={19}/><span>Issues</span></button>
        <button className={mobileMenuOpen||["overlay","files","evidence","reports","directory","settings"].includes(view)?"bottom-item active":"bottom-item"} type="button" aria-label="Open workspace menu" aria-expanded={mobileMenuOpen} onClick={()=>setMobileMenuOpen(value=>!value)}><MoreHorizontal size={19}/><span>More</span></button>
      </nav>
    </main>
  );
}

function PageHeading({ eyebrow, title, copy, action }: { eyebrow: string; title: string; copy: string; action?: React.ReactNode }) {
  return <header className="page-heading"><div><p>{eyebrow}</p><h1>{title}</h1><span>{copy}</span></div>{action}</header>;
}

function OverviewView({ userName, organizationName, issues, documents, job, onReview, onCapture, onDrawings }: { userName:string; organizationName:string; issues: Issue[]; documents: DocumentAsset[]; job: Job | null; onReview: () => void; onCapture: () => void; onDrawings: () => void }) {
  const ready = issues.filter(i => i.status === "review_ready").length;
  const gaps = issues.filter(i => i.status === "needs_more_evidence").length;
  return <section className="overview-page">
    <PageHeading eyebrow={organizationName.toUpperCase()} title={`Good morning, ${userName.split(" ")[0]}`} copy="Here are the field decisions that need a human review today." action={<div className="heading-actions"><button onClick={onCapture}><Camera size={17}/> Capture evidence</button><button className="primary" onClick={onReview}><ClipboardCheck size={17}/> Review queue <b>{ready}</b></button></div>} />
    <div className="signal-strip">
      <article><span>Review-ready</span><strong>{ready}</strong><small><i className="good"/> Human decision queue</small></article>
      <article><span>Evidence gaps</span><strong>{gaps}</strong><small><i className="warn"/> Field action needed</small></article>
      <article><span>Open issues</span><strong>{issues.filter(item=>!["approved","rejected_false_positive"].includes(item.status)).length}</strong><small>Current project only</small></article>
      <article><span>Drawing coverage</span><strong>{job?.progress ?? 0}%</strong><small>{documents.length} current source file(s)</small></article>
    </div>
    <div className="overview-grid">
      <section className="priority-list"><div className="section-line"><div><p>PRIORITY</p><h2>Needs your decision</h2></div><button onClick={onReview}>View queue →</button></div>
        {issues.slice(0,4).map((issue, index) => <button className="priority-row" key={issue.issue_id} onClick={onReview}><span className={`priority-index tone-${index}`}>{String(index+1).padStart(2,"0")}</span><div><strong>{issue.title}</strong><p>{issue.room} · {String(issue.requirement.source ?? issue.plan_location.sheet_id ?? "Source unresolved")}</p></div><span className={`plain-status ${issue.status}`}>{statusLabel(issue.status)}</span><time>{issue.due_date || "Today"}</time></button>)}
      </section>
      <aside className="project-progress"><p>VERIFICATION PROGRESS</p><h2>Current drawing set</h2><div className="ring" style={{"--progress": `${job?.progress ?? 0}%`} as React.CSSProperties}><strong>{job?.progress ?? 0}%</strong><span>verified</span></div><dl><div><dt>{documents.length}</dt><dd>Files indexed</dd></div><div><dt>{issues.filter(item=>item.room).length}</dt><dd>Located issues</dd></div><div><dt>{gaps}</dt><dd>Need evidence</dd></div></dl><button type="button" onClick={onDrawings}>Open drawing workspace →</button></aside>
    </div>
    <section className="activity-flow"><div className="section-line"><div><p>ACTIVITY</p><h2>Latest on site</h2></div></div><div className="timeline"><span/><article><b>Jordan Davis</b> prepared the garage receptacle RFI for review <time>18 min ago</time></article><span/><article><b>Buili verification</b> linked the current E1.1 source to the field measurement <time>42 min ago</time></article><span/><article><b>Mike Torres</b> added context, detail, measurement, and a foreman voice note <time>1 hr ago</time></article></div></section>
    <section className="verification-layer"><div className="section-line"><div><p>BUILT FOR THE EVIDENCE GAP</p><h2>From field signal to system-of-record output</h2></div></div><p className="verification-intro">Buili is the verification and decision layer between capture and project controls. It keeps every recommendation anchored to the current source, visible evidence, and an accountable human decision.</p><ol><li><span>01</span><div><b>Current-source control</b><small>Superseded sheets cannot silently support a new issue.</small></div></li><li><span>02</span><div><b>2D → 3D spatial index</b><small>Use a drawing even when a coordinated BIM model does not exist.</small></div></li><li><span>03</span><div><b>Evidence readiness</b><small>Missing location, media, or source context blocks issuance.</small></div></li><li><span>04</span><div><b>Human-issued package</b><small>Export an RFI or punch record with its review and source snapshot intact.</small></div></li></ol></section>
  </section>;
}

function FilesView({ documents, revisions, staleIssueCount, busy, onUpload, onActivate }: { documents: DocumentAsset[]; revisions: DrawingRevision[]; staleIssueCount:number; busy: boolean; onUpload: () => void; onActivate: (documentId:string, metadata:{sheetNumber:string;issueDate:string;discipline:string})=>Promise<void> }) {
  const [tab,setTab]=useState("all"); const [search,setSearch]=useState(""); const [showLog,setShowLog]=useState(false); const [editing,setEditing]=useState<DocumentAsset|null>(null); const [sheetNumber,setSheetNumber]=useState(""); const [issueDate,setIssueDate]=useState(""); const [discipline,setDiscipline]=useState("architectural");
  const rows=documents.filter(doc=>(tab==="all"||(tab==="drawings"&&doc.type==="plan")||(tab==="specs"&&doc.type==="spec")||(tab==="records"&&["rfi","submittal"].includes(doc.type)))&&`${doc.filename} ${doc.revision} ${doc.type}`.toLowerCase().includes(search.toLowerCase()));
  const revisionFor=(doc:DocumentAsset)=>revisions.find(item=>(item.document_id??item.doc_id)===doc.doc_id);
  const stateFor=(doc:DocumentAsset)=>revisionFor(doc)?.state??String(doc.metadata_json.revision_state??"unclassified");
  function edit(doc:DocumentAsset){const revision=revisionFor(doc);setEditing(doc);setSheetNumber(revision?.sheet_number??String(doc.metadata_json.sheet_id??doc.filename.replace(/\.[^.]+$/,"")));setIssueDate(revision?.issue_date??"");setDiscipline(revision?.discipline??String(doc.metadata_json.discipline??"architectural"))}
  return <section className="standard-page"><PageHeading eyebrow="PROJECT DATA" title="Files & revisions" copy="Control the current set before evidence or AI drafts can cite it." action={<button className="page-primary" onClick={onUpload}><Upload size={17}/> Upload files</button>}/>
    <div className="revision-alert"><ShieldCheck size={19}/><div><strong>Current-set integrity is enforced</strong><span>{revisions.filter(item=>item.state==="superseded").length} superseded · {revisions.filter(item=>item.state==="unclassified").length} unclassified · {staleIssueCount} stale-source issue{staleIssueCount===1?"":"s"} awaiting review.</span></div><button type="button" onClick={()=>setShowLog(current=>!current)}>{showLog?"Hide":"View"} revision log</button></div>
    {showLog?<div className="revision-log" aria-label="Revision log">{revisions.length?revisions.map(item=><div key={item.revision_id??item.document_id}><Clock3 size={15}/><b>{item.sheet_number||item.logical_key||"Unclassified"}</b><span>Rev {item.revision||"—"}</span><em className={`revision-state ${item.state}`}>{item.state}</em><small>{item.issue_date||"Issue date missing"} · {item.upload_actor||"system"}</small></div>):<p>No revision events yet.</p>}</div>:null}
    <div className="table-tools"><div className="tabs">{[["all","All files"],["drawings","Drawings"],["specs","Specifications"],["records","RFIs & submittals"]].map(([key,label])=><button type="button" className={tab===key?"active":""} key={key} onClick={()=>setTab(key)}>{label} {key==="all"?<b>{documents.length}</b>:null}</button>)}</div><label><Search size={16}/><input value={search} onChange={event=>setSearch(event.target.value)} placeholder="Filter files"/></label></div>
    <div className="data-table file-table"><div className="table-head"><span>Name</span><span>Revision</span><span>Status</span><span>Discipline</span><span>Processed</span><span>Action</span></div>
      {rows.map(doc=>{const revision=revisionFor(doc);const state=stateFor(doc);return <div className="table-row" key={doc.doc_id}><span className="file-name"><FileText size={19}/><div><b>{doc.filename}</b><small>{doc.mime} · {doc.size?`${Math.max(1,Math.round(doc.size/1024))} KB`:"size unavailable"}</small></div></span><span><b>{doc.revision||"Missing"}</b></span><span><i className={state}/><em className={`revision-state ${state}`}>{state.replaceAll("_"," ")}</em>{Number(revision?.impacted_issue_count)>0?<small className="stale-impact">{revision?.impacted_issue_count} stale</small>:null}</span><span>{revision?.discipline||String(doc.metadata_json.discipline??"Unclassified")}</span><span>{doc.parsed_status==="parsed"?"Indexed":doc.parsed_status}</span><button type="button" onClick={()=>edit(doc)} aria-label={`Classify or activate ${doc.filename}`}>{state==="current"?"Details":"Classify"}</button></div>})}
      {!rows.length?<p className="table-empty">No files match this view.</p>:null}
    </div>
    {editing?<form className="inline-editor" onSubmit={async event=>{event.preventDefault();await onActivate(editing.doc_id,{sheetNumber,issueDate,discipline});setEditing(null)}}><header><div><b>Classify and activate</b><span>{editing.filename}</span></div><button type="button" onClick={()=>setEditing(null)} aria-label="Close file editor"><X size={17}/></button></header><label>Sheet / logical key<input value={sheetNumber} onChange={event=>setSheetNumber(event.target.value)} required/></label><label>Issue date<input type="date" value={issueDate} onChange={event=>setIssueDate(event.target.value)} required/></label><label>Discipline<select value={discipline} onChange={event=>setDiscipline(event.target.value)}><option value="architectural">Architectural</option><option value="structural">Structural</option><option value="mechanical">Mechanical</option><option value="electrical">Electrical</option><option value="plumbing">Plumbing</option></select></label><button className="page-primary" type="submit" disabled={busy}><Check size={16}/>{busy?"Activating…":"Make current"}</button></form>:null}
  </section>;
}

function IssuesView({ issues, selectedId, filter, counts, onFilter, onSelect, onCreate }: { issues: Issue[]; selectedId?: string; filter: IssueFilter; counts: Record<IssueFilter,number>; onFilter:(f:IssueFilter)=>void; onSelect:(id:string)=>void; onCreate:()=>void }) {
  const [search,setSearch]=useState(""); const rows=issues.filter(issue=>`${issue.issue_id} ${issue.title} ${issue.room} ${issue.assignee} ${issue.discipline}`.toLowerCase().includes(search.toLowerCase()));
  return <section className="standard-page"><PageHeading eyebrow="FIELD TO ACTION" title="Issues" copy="Every observation, source and decision in one auditable record." action={<button className="page-primary" type="button" onClick={onCreate}><MessageSquarePlus size={17}/> New issue</button>}/>
    <div className="table-tools"><div className="tabs">{(["all","open","review","resolved"] as IssueFilter[]).map(f=><button className={filter===f?"active":""} key={f} onClick={()=>onFilter(f)}>{f === "review" ? "Needs evidence" : f} <b>{counts[f]}</b></button>)}</div><label><Search size={16}/><input value={search} onChange={event=>setSearch(event.target.value)} placeholder="Search issues"/></label></div>
    <div className="data-table issue-table"><div className="table-head"><span>Issue</span><span>Location</span><span>Source</span><span>Assignee</span><span>Due</span><span>Status</span></div>{rows.map(issue=><button className={selectedId===issue.issue_id?"table-row selected":"table-row"} key={issue.issue_id} onClick={()=>onSelect(issue.issue_id)}><span><i className={`severity-dot ${issue.severity}`}/><div><b>{issue.title}</b><small>{issue.discipline} · {issueCode(issue)}</small></div></span><span>{issue.room}</span><span className="source-link">{String(issue.requirement.source ?? "Unresolved")}</span><span>{issue.assignee || "Unassigned"}</span><span>{issue.due_date || "—"}</span><span><em className={`plain-status ${issue.status}`}>{statusLabel(issue.status)}</em></span></button>)}{!rows.length?<p className="table-empty">No issues match this view.</p>:null}</div>
  </section>;
}

function DirectoryView({ members, onInvite, onChange }: { members: DirectoryMember[]; onInvite:(payload:Omit<DirectoryMember,"directory_id"|"project_id">)=>Promise<void>; onChange:(id:string,patch:Partial<DirectoryMember>)=>Promise<void> }) {
  const [inviting,setInviting]=useState(false); const [busy,setBusy]=useState(false); const [error,setError]=useState("");
  async function submit(event:FormEvent<HTMLFormElement>){event.preventDefault();const form=new FormData(event.currentTarget);const person_name=String(form.get("person_name")??"").trim();const email=String(form.get("email")??"").trim();if(!person_name||!/^\S+@\S+\.\S+$/.test(email)){setError("Name and a valid email are required.");return}setBusy(true);setError("");try{await onInvite({person_name,email,company:String(form.get("company")??""),role:String(form.get("role")??"field_user"),trade:String(form.get("trade")??""),status:"invited",notification:{in_app:true,email:true},access_expires_at:String(form.get("expires")??"")||null});setInviting(false)}catch(err){setError(err instanceof Error?err.message:"Invite failed")}finally{setBusy(false)}}
  return <section className="standard-page"><PageHeading eyebrow="PROJECT TEAM" title="Directory" copy="Company, role, trade, access period, and notifications for every project participant." action={<button className="page-primary" type="button" onClick={()=>setInviting(current=>!current)}><UserPlus size={17}/>{inviting?"Close invite":"Invite member"}</button>}/>
    {inviting?<form className="invite-form" onSubmit={submit}><label>Name<input name="person_name" autoFocus/></label><label>Email<input name="email" type="email"/></label><label>Company<input name="company"/></label><label>Role<select name="role"><option value="field_user">Field user</option><option value="project_engineer">Project engineer</option><option value="project_manager">Project manager / approver</option><option value="external_reviewer">External reviewer</option></select></label><label>Trade<input name="trade"/></label><label>Access expires<input name="expires" type="date"/></label>{error?<p className="form-error" role="alert">{error}</p>:null}<button className="page-primary" type="submit" disabled={busy}><Send size={16}/>{busy?"Inviting…":"Send invitation"}</button></form>:null}
    <div className="people-list">{members.map((member,i)=><article key={member.directory_id}><span className={`person-avatar a${i%5}`}>{member.person_name.split(" ").map(part=>part[0]).join("").slice(0,2)}</span><div><b>{member.person_name}</b><small>{member.role.replaceAll("_"," ")} · {member.trade||"No trade"}</small></div><span>{member.company||member.email}</span><em className={member.status}>{member.status}</em><button type="button" aria-label={`${member.status==="disabled"?"Enable":"Disable"} ${member.person_name}`} onClick={()=>void onChange(member.directory_id,{status:member.status==="disabled"?"active":"disabled"})}>{member.status==="disabled"?<Check size={17}/>:<X size={17}/>}</button></article>)}{!members.length?<p className="table-empty">No directory members yet. Invite the first project approver.</p>:null}</div>
  </section>
}

function SettingsView({ user, project, settings, technologyStatus, onSave }: { user:AuthSession["user"]; project:Project|null; settings:ProjectSettings|null; technologyStatus:TechnologyStatus[]; onSave:(next:Partial<ProjectSettings>)=>Promise<void> }) {
  const [tab,setTab]=useState("general"); const [busy,setBusy]=useState(false); const [error,setError]=useState(""); const [unit,setUnit]=useState(settings?.unit_system??"imperial"); const [timezone,setTimezone]=useState(settings?.timezone??"America/Los_Angeles"); const [threshold,setThreshold]=useState(String(settings?.settings?.confidence_threshold??"0.65")); const [sla,setSla]=useState(String(settings?.workflow?.review_sla_hours??"24")); const [secondReviewer,setSecondReviewer]=useState(Boolean(settings?.workflow?.second_reviewer_high_risk??true)); const [issueTypes,setIssueTypes]=useState<Record<string,boolean>>({punch:true,rfi:true,pce:true,observation:true}); const [serviceFlags,setServiceFlags]=useState<Record<string,boolean>>({}); const [integrations,setIntegrations]=useState<Record<string,boolean>>({email:true,storage:false,webhook:false,procore:false});
  useEffect(()=>{setUnit(settings?.unit_system??"imperial");setTimezone(settings?.timezone??"America/Los_Angeles");setThreshold(String(settings?.settings?.confidence_threshold??"0.65"));setSla(String(settings?.workflow?.review_sla_hours??"24"));setSecondReviewer(Boolean(settings?.workflow?.second_reviewer_high_risk??true));setIssueTypes({punch:true,rfi:true,pce:true,observation:true,...(settings?.settings?.issue_types as Record<string,boolean>|undefined)});setIntegrations({email:true,storage:false,webhook:false,procore:false,...(settings?.settings?.integrations as Record<string,boolean>|undefined)});setServiceFlags(Object.fromEntries(technologyStatus.map(item=>[item.key,item.status!=="disabled"])))},[settings,technologyStatus]);
  async function submit(event:FormEvent){event.preventDefault();setBusy(true);setError("");try{await onSave({timezone,unit_system:unit,settings:{...(settings?.settings??{}),confidence_threshold:Number(threshold),issue_types:issueTypes,integrations,verification_services:serviceFlags},workflow:{...(settings?.workflow??{}),review_sla_hours:Number(sla),second_reviewer_high_risk:secondReviewer}})}catch(cause){setError(cause instanceof Error?cause.message:"Settings save failed.")}finally{setBusy(false)}}
  const tabs=[["general","General"],["workflow","Workflow & approvals"],["types","Issue types"],["ai","AI policy"],["integrations","Integrations"],["audit","Access & audit"]];
  return <section className="standard-page"><PageHeading eyebrow="CONFIGURATION" title="Project settings" copy="Units, workflow gates, AI policy, integrations, and authenticated access."/><div className="settings-layout"><nav>{tabs.map(([key,label])=><button type="button" className={tab===key?"active":""} key={key} onClick={()=>setTab(key)}>{label}</button>)}</nav><form className="settings-form" onSubmit={submit}>
    {tab==="general"?<><h2>General configuration</h2><label>Project name<input value={project?.name??""} readOnly/></label><div className="field-pair"><label>Measurement system<select value={unit} onChange={event=>setUnit(event.target.value)}><option value="imperial">Imperial</option><option value="metric">Metric</option></select></label><label>Timezone<select value={timezone} onChange={event=>setTimezone(event.target.value)}><option value="America/Los_Angeles">Pacific Time (US)</option><option value="America/New_York">Eastern Time (US)</option><option value="Asia/Seoul">Korea Standard Time</option><option value="UTC">UTC</option></select></label></div></>:null}
    {tab==="workflow"?<><h2>Workflow & approvals</h2><label>Review SLA (hours)<input type="number" min="1" value={sla} onChange={event=>setSla(event.target.value)}/></label><label className="toggle-row"><span><b>Second reviewer for high risk</b><small>Required for stale sources, cost impact, or low location confidence.</small></span><input type="checkbox" checked={secondReviewer} onChange={event=>setSecondReviewer(event.target.checked)}/></label></>:null}
    {tab==="types"?<><h2>Issue types</h2>{[["punch","Punch item"],["rfi","Request for information"],["pce","Potential change event"],["observation","Daily observation"]].map(([key,label])=><label className="toggle-row" key={key}><span><b>{label}</b><small>Available as a recommended route and report output.</small></span><input type="checkbox" checked={Boolean(issueTypes[key])} onChange={event=>setIssueTypes(current=>({...current,[key]:event.target.checked}))}/></label>)}</>:null}
    {tab==="ai"?<><h2>AI policy</h2><label>Minimum review confidence<input type="number" min="0" max="1" step="0.05" value={threshold} onChange={event=>setThreshold(event.target.value)}/></label>{technologyStatus.slice(0,4).map(item=><label className="toggle-row" key={item.key}><span><b>{item.label}</b><small>{item.summary}</small></span><input type="checkbox" checked={Boolean(serviceFlags[item.key])} onChange={event=>setServiceFlags(current=>({...current,[item.key]:event.target.checked}))}/></label>)}<div className="policy-note"><AlertTriangle size={18}/><span>Official actions always require an authorized human reviewer. AI confidence is never exported as a contractual assertion.</span></div></>:null}
    {tab==="integrations"?<><h2>Integrations</h2>{[["email","Email delivery"],["storage","Cloud storage export"],["webhook","Webhook notifications"],["procore","Procore / ACC handoff"]].map(([key,label])=><label className="toggle-row" key={key}><span><b>{label}</b><small>Controlled at project scope; exports retain their source snapshots.</small></span><input type="checkbox" checked={Boolean(integrations[key])} onChange={event=>setIntegrations(current=>({...current,[key]:event.target.checked}))}/></label>)}</>:null}
    {tab==="audit"?<><h2>Authenticated review identity</h2><label>Signed-in user<input value={user.name} readOnly/></label><label>Work email<input value={user.email} readOnly/></label><label>Authorized role<input value={user.role.replaceAll("_"," ")} readOnly/></label><div className="policy-note"><ShieldCheck size={18}/><span>Review identity and role are derived from the signed server session. Browser-supplied identity headers are discarded before every gated mutation and audit event.</span></div></>:null}
    {error?<p className="form-error" role="alert">{error}</p>:null}<button className="page-primary" type="submit" disabled={busy}><Save size={16}/>{busy?"Saving…":"Save changes"}</button></form></div></section>
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
  reviewHistory,
  onFilter,
  onSelectIssue,
  onCreateIssue,
  onEditIssue,
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
  reviewHistory: ReviewAuditEntry[];
  onFilter: (filter: IssueFilter) => void;
  onSelectIssue: (id: string) => void;
  onCreateIssue: () => void;
  onEditIssue: (id: string) => void;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
  onNeedMore: (id: string) => void;
  onRfi: (id: string) => void;
}) {
  const pins = useMemo(() => buildModelPins(filteredIssues, overlay), [filteredIssues, overlay]);
  const activeIssue = filteredIssues.find((item) => item.issue_id === issue?.issue_id) ?? filteredIssues[0] ?? issue;
  const selectedPin = pins.find((pin) => pin.issueId === activeIssue?.issue_id) ?? pins[0];
  const [reviewMode, setReviewMode] = useState<"2d" | "3d" | "split">("split");
  const [viewerZoom, setViewerZoom] = useState(1);
  const [reviewStage, setReviewStage] = useState<"locate" | "compare" | "decide">("compare");
  const hasSource = Boolean(activeIssue && (activeIssue.requirement.source || activeIssue.plan_location.sheet_id));
  const hasObservation = Boolean(activeIssue && (activeIssue.evidence.length || activeIssue.observation.media_id));

  function moveReview(stage: "locate" | "compare" | "decide") {
    setReviewStage(stage);
    if (stage === "locate") setReviewMode("2d");
    if (stage === "compare") setReviewMode("split");
  }

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
        <button className="add-issue-button" type="button" onClick={onCreateIssue}>
          <span>+</span>
          Add Issue
        </button>
      </aside>

      <section className="model-workspace verification-workspace">
        <nav className="verification-steps" aria-label="Issue verification steps">
          <button className={reviewStage==="locate"?"active":""} type="button" onClick={()=>moveReview("locate")}><span>1</span><div><b>Locate</b><small>Confirm the current source</small></div>{hasSource?<Check size={15}/>:<AlertTriangle size={15}/>}</button>
          <button className={reviewStage==="compare"?"active":""} type="button" onClick={()=>moveReview("compare")}><span>2</span><div><b>Compare</b><small>Expected vs observed</small></div>{hasObservation?<Check size={15}/>:<AlertTriangle size={15}/>}</button>
          <button className={reviewStage==="decide"?"active":""} type="button" onClick={()=>moveReview("decide")}><span>3</span><div><b>Decide</b><small>Approve, route, or reject</small></div><ArrowRight size={15}/></button>
        </nav>
        <div className="review-viewbar">
          <div className="view-toggle" aria-label="Plan view mode">
            {(["2d","3d","split"] as const).map(mode=><button className={reviewMode===mode?"active":""} type="button" key={mode} onClick={()=>setReviewMode(mode)}>{mode==="2d"?"2D source":mode==="3d"?"3D context":"2D + 3D"}</button>)}
          </div>
          <div className="model-toolbar" aria-label="Spatial review tools">
            <span className="review-control-hint">{reviewMode==="3d"?"Drag to orbit · shift-drag to pan · scroll to zoom":"2D zoom · drag 3D to orbit · shift-drag to pan"}</span>
            {reviewMode!=="3d"?<><button className="tool-button" type="button" title="Zoom out" aria-label="Zoom out" onClick={()=>setViewerZoom(value=>Math.max(.7,value-.15))}><ZoomOut size={18}/></button><button className="tool-button" type="button" title="Zoom in" aria-label="Zoom in" onClick={()=>setViewerZoom(value=>Math.min(1.8,value+.15))}><ZoomIn size={18}/></button><button className="tool-button" type="button" title="Fit 2D source" aria-label="Fit 2D source" onClick={()=>setViewerZoom(1)}><Maximize size={18}/></button></>:null}
          </div>
        </div>

        <div className={`review-spatial-stage review-mode-${reviewMode}`}>
          {reviewMode!=="3d"?<div className="review-pane plan-pane"><span className="pane-label"><FileText size={14}/> 2D · current drawing</span><div className="review-media" style={{transform:`scale(${viewerZoom})`}}><img src={PLAN_IMAGE_SRC} alt="Current 2D drawing with synchronized issue pins"/>{(overlay?.pins??[]).map(pin=><button key={pin.id} type="button" aria-label={`Select ${pin.label} in 2D`} className={`plan-review-pin ${pin.severity} ${pin.id===activeIssue?.issue_id?"active":""}`} style={{left:`${pin.x*100}%`,top:`${pin.y*100}%`}} onClick={()=>onSelectIssue(pin.id)}><span>{pin.label}</span></button>)}</div></div>:null}
          {reviewMode!=="2d"?<div className="review-pane model-pane"><span className="pane-label"><Box size={14}/> 3D · interactive spatial index</span><SpatialModelViewer compact projectId={activeIssue?.project_id??""} initialAssetId={activeIssue?.spatial_context?.design_asset_id} pins={pins} selectedIssueId={activeIssue?.issue_id} geometryConfidence={Number(activeIssue?.spatial_context?.geometry_confidence??0)} onSelectIssue={onSelectIssue} fallbackImage={PLAN2FIELD_3D_SRC}/></div>:null}
          <div className="spatial-sync-note"><Layers size={15}/><span>Selection and source coordinates stay synchronized between views.</span><b>{Math.round(viewerZoom*100)}%</b></div>
        </div>

        <div className="floor-key">
          <strong>Floor Plan Key</strong>
          <span><i className="key-dot open" /> Open</span>
          <span><i className="key-dot review" /> In Review</span>
          <span><i className="key-dot resolved" /> Resolved</span>
        </div>

        <div className="mini-map-card" aria-label="Source location minimap">
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
            {mediaAssets.filter(item=>item.mime.startsWith("image/")).slice(0,1).map(item=><EvidenceThumb key={item.media_id} src={mediaThumbnailUrl(item)} fallback={mediaFallbackUrl(item)} label={String(item.metadata_json.label??item.filename)} />)}
            {!mediaAssets.some(item=>item.mime.startsWith("image/"))?<><EvidenceThumb src={FIELD_IMAGE_SRC} label="Field context"/><EvidenceThumb src={PLAN_IMAGE_SRC} label="Current source"/><EvidenceThumb src={PLAN2FIELD_3D_SRC} label="3D spatial view"/></>:null}
          </div>
          {mediaAssets.find(item=>item.mime.startsWith("audio/"))?<CompactAudioEvidence media={mediaAssets.find(item=>item.mime.startsWith("audio/"))!}/>:null}
        </div>
        <div className="bottom-documents">
          <PanelTitle label="Current drawing" count={Math.max(documents.filter(item=>item.type==="plan").length, 1)} />
          <div className="doc-strip">
            <DocThumb src={PLAN2FIELD_MINIMAP_SRC} name={documents.find(item=>item.type==="plan")?.filename??String(activeIssue?.plan_location.sheet_id ?? "E1.1.pdf")} />
          </div>
        </div>
        <IssueSummaryPanel
          issue={activeIssue}
          selectedPin={selectedPin}
          history={reviewHistory.filter(item=>item.issue_id===activeIssue?.issue_id)}
          onEdit={onEditIssue}
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

function EvidenceThumb({ src, label, fallback }: { src: string; label: string; fallback?:string }) {
  return (
    <figure className="evidence-thumb">
      <img src={src} alt={label} loading="lazy" decoding="async" onError={event=>{if(fallback&&event.currentTarget.src!==new URL(fallback,window.location.href).href)event.currentTarget.src=fallback}} />
      <figcaption>{label}</figcaption>
    </figure>
  );
}

function CompactAudioEvidence({ media }: { media:SiteMediaAsset }) {
  const transcript=String(media.metadata_json.transcript??"");
  const captions=String(media.metadata_json.captions_uri??"");
  return <div className="compact-audio-evidence"><span><Volume2 size={16}/><b>{String(media.metadata_json.label??media.filename)}</b></span><audio controls preload="metadata"><source src={mediaDownloadUrl(media)} type={media.mime}/><source src={mediaFallbackUrl(media)} type={media.mime}/>{captions?<track kind="captions" src={captions} srcLang="en" label="English" default/>:null}</audio>{transcript?<details><summary>Read transcript</summary><p>{transcript}</p></details>:null}</div>;
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
  history,
  onEdit,
  onApprove,
  onReject,
  onNeedMore,
  onRfi
}: {
  issue?: Issue;
  selectedPin?: ReturnType<typeof buildModelPins>[number];
  history: ReviewAuditEntry[];
  onEdit: (id: string) => void;
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
  const readiness = [
    { label: "Current source", ready: Boolean(issue.requirement.source || issue.plan_location.sheet_id) },
    { label: "Confirmed location", ready: Boolean(issue.room && issue.room.toLowerCase() !== "unknown") },
    { label: "Field observation", ready: Boolean(issue.evidence.length || issue.observation.media_id) }
  ];
  const readyToDecide = readiness.every(item=>item.ready);
  return (
    <aside className="issue-summary-panel">
      <div className="issue-summary-heading"><strong>Decision</strong><span className={readyToDecide?"ready":"gap"}>{readyToDecide?<><Check size={14}/> Ready for review</>:<><AlertTriangle size={14}/> Evidence gap</>}</span></div>
      <div className="readiness-checks" aria-label="Issue readiness">{readiness.map(item=><span className={item.ready?"ready":"gap"} key={item.label}>{item.ready?<Check size={13}/>:<X size={13}/>} {item.label}</span>)}</div>
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
        <button type="button" onClick={() => onEdit(issue.issue_id)}>
          <Pencil size={16} />
          Edit
        </button>
        <button type="button" onClick={() => onApprove(issue.issue_id)} disabled={!readyToDecide} title={readyToDecide?"Approve this evidence package":"Resolve source, location, and evidence gaps first"}>
          <Check size={16} />
          Approve
        </button>
        <button type="button" onClick={() => onNeedMore(issue.issue_id)}>
          <MoreHorizontal size={16} />
          Request evidence
        </button>
        <button type="button" onClick={() => onRfi(issue.issue_id)} disabled={!readiness[0].ready || !readiness[1].ready} title="Create a source-cited RFI draft">
          <MessageSquarePlus size={16} />
          RFI
        </button>
        <button type="button" onClick={() => onReject(issue.issue_id)}>
          <X size={16} />
          Reject
        </button>
      </div>
      <section className="review-history" role="region" aria-label="Review history">
        <strong>Review history</strong>
        {history.length ? history.slice(0,4).map(item=><div key={item.review_id}><Clock3 size={14}/><span><b>{item.decision.replaceAll("_"," ")}</b><small>{item.reviewer} · {item.reason_code?.replaceAll("_"," ")||"reviewed"}</small>{item.reason?<em>{item.reason}</em>:null}</span></div>) : <p>No review decisions yet.</p>}
      </section>
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

function EvidenceMediaGallery({ assets }: { assets:SiteMediaAsset[] }) {
  const [active,setActive]=useState<SiteMediaAsset|null>(null);
  const images=assets.filter(item=>item.mime.startsWith("image/"));
  const audio=assets.filter(item=>item.mime.startsWith("audio/"));
  const [selectedImageId,setSelectedImageId]=useState("");
  const selectedImage=images.find(item=>item.media_id===selectedImageId)??images[0];
  if(!assets.length){
    return <div className="crop-stage" aria-label="Offline field crop preview"><img className="field-image" src={FIELD_IMAGE_SRC} alt="Offline construction evidence sample"/><span className="crop-marker"/></div>;
  }
  return <>
    <div className="evidence-media-grid" aria-label="Field evidence media">
      <nav className="evidence-media-selector" aria-label="Evidence photos">{images.map((item,index)=>{const location=(item.metadata_json.location??{}) as Record<string,unknown>;return <button className={item.media_id===selectedImage?.media_id?"active":""} type="button" key={item.media_id} onClick={()=>setSelectedImageId(item.media_id)}><img src={mediaThumbnailUrl(item)} alt="" loading="lazy" decoding="async"/><span>{String(index+1).padStart(2,"0")}</span><div><b>{String(item.metadata_json.label??item.filename)}</b><small>{String(location.floor??"")} · {String(location.room??"Location pending")}</small></div>{item.media_id===selectedImage?.media_id?<Check size={14}/>:<ArrowRight size={14}/>}</button>})}</nav>
      {selectedImage?<button className="evidence-media-focus" type="button" onClick={()=>setActive(selectedImage)}><img src={mediaThumbnailUrl(selectedImage)} alt={String(selectedImage.metadata_json.label??selectedImage.filename)} decoding="async" onError={event=>{const fallback=mediaFallbackUrl(selectedImage);if(event.currentTarget.src!==new URL(fallback,window.location.href).href)event.currentTarget.src=fallback}}/><span><b>{String(selectedImage.metadata_json.label??selectedImage.filename)}</b><small>Optimized preview · open the preserved original to inspect</small></span><em>Open original</em></button>:null}
    </div>
    {audio.map(item=>{const transcript=String(item.metadata_json.transcript??"");const captions=String(item.metadata_json.captions_uri??"");return <article className="evidence-audio-record" key={item.media_id}><header><span><Volume2 size={18}/></span><div><b>{String(item.metadata_json.label??item.filename)}</b><small>{String(item.metadata_json.captured_by??"Field contributor")} · {Math.round(Number(item.metadata_json.duration_seconds??0))} sec</small></div></header><audio controls preload="metadata"><source src={mediaDownloadUrl(item)} type={item.mime}/><source src={mediaFallbackUrl(item)} type={item.mime}/>{captions?<track kind="captions" src={captions} srcLang="en" label="English" default/>:null}</audio>{transcript?<details open><summary>Transcript</summary><p>{transcript}</p></details>:null}</article>})}
    {active?<div className="evidence-lightbox" role="presentation" onMouseDown={event=>event.currentTarget===event.target&&setActive(null)}><section role="dialog" aria-modal="true" aria-label={String(active.metadata_json.label??active.filename)}><button type="button" onClick={()=>setActive(null)} aria-label="Close evidence preview"><X size={19}/></button><img src={mediaDownloadUrl(active)} alt={String(active.metadata_json.label??active.filename)} decoding="async" onError={event=>{event.currentTarget.src=mediaFallbackUrl(active)}}/><footer><div><b>{String(active.metadata_json.label??active.filename)}</b><small>Original preserved · {String(active.metadata_json.captured_by??"Field contributor")}</small></div><a href={mediaDownloadUrl(active)} target="_blank" rel="noreferrer">Open original <ExternalLink size={14}/></a></footer></section></div>:null}
  </>;
}

function EvidenceViewer({
  issue,
  documents,
  mediaAssets,
  fieldEvidence,
  observations
}: {
  issue?: Issue;
  documents: DocumentAsset[];
  mediaAssets: SiteMediaAsset[];
  fieldEvidence: FieldEvidenceRecord[];
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
        <span>{documents.length} docs · {mediaAssets.length + fieldEvidence.length} evidence records</span>
      </div>
      <EvidenceMediaGallery assets={mediaAssets}/>
      {fieldEvidence.length ? <div className="field-evidence-list" aria-label="Synced field evidence">{fieldEvidence.slice(0,8).map(item=>{const location=item.location??item.location_json??{};return <article key={item.evidence_id}><span className={`evidence-state ${item.sufficiency}`}>{item.sufficiency}</span><div><b>{item.filename||item.media_type}</b><small>{String(location.floor??"")} / {String(location.room??"Unlinked")} · {item.author||"Unknown author"}</small></div><time>{new Date(item.captured_at).toLocaleString()}</time></article>})}</div>:null}
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
        <img className="plan-image" src={PLAN_IMAGE_SRC} alt="Current E1.1 electrical drawing" />
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

function OverlayView({ projectId, overlay, issues, selectedIssueId, onSelectIssue, onOpenIssue }: { projectId:string; overlay:Overlay|null; issues:Issue[]; selectedIssueId:string; onSelectIssue:(id:string)=>void; onOpenIssue:(id:string)=>void }) {
  const [mode,setMode]=useState<ViewerMode>("2d"); const [tool,setTool]=useState<ViewerTool>("select"); const [zoom,setZoom]=useState(1); const [showGrid,setShowGrid]=useState(false); const [compare,setCompare]=useState(50);
  const [showPins,setShowPins]=useState(true); const [showDirections,setShowDirections]=useState(true);
  const [planPan,setPlanPan]=useState({x:0,y:0}); const panStart=useRef<{x:number;y:number;originX:number;originY:number}|null>(null);
  const [measurePoints,setMeasurePoints]=useState<Array<{x:number;y:number}>>([]); const [markups,setMarkups]=useState<Array<{id:string;x:number;y:number}>>([]);
  const selected=issues.find(item=>item.issue_id===selectedIssueId)??issues[0];
  const modelPins=useMemo(()=>buildModelPins(issues,overlay),[issues,overlay]);
  const tools:[ViewerTool,React.ReactNode,string][]=[["select",<MousePointer2 size={18}/>,"Select"],["pan",<Hand size={18}/>,"Pan 2D"],["measure",<Ruler size={18}/>,"Measure on 2D"],["markup",<Pencil size={18}/>,"Add correction markup"]];
  const measured=measurePoints.length===2?Math.hypot(measurePoints[1].x-measurePoints[0].x,measurePoints[1].y-measurePoints[0].y):0;
  function planCoordinate(event:React.MouseEvent<HTMLDivElement>){const rect=event.currentTarget.getBoundingClientRect();const x=((event.clientX-rect.left-rect.width/2-planPan.x)/zoom+rect.width/2)/rect.width*100;const y=((event.clientY-rect.top-rect.height/2-planPan.y)/zoom+rect.height/2)/rect.height*100;return{x:Math.max(0,Math.min(100,x)),y:Math.max(0,Math.min(100,y))}}
  function actOnPlan(event:React.MouseEvent<HTMLDivElement>){if((event.target as HTMLElement).closest("button"))return;if(tool==="measure"){const point=planCoordinate(event);setMeasurePoints(current=>current.length===1?[current[0],point]:[point])}else if(tool==="markup"){const point=planCoordinate(event);setMarkups(current=>[...current,{id:`markup-${Date.now()}`, ...point}])}}
  const plan=<div className={`interactive-plan ${showGrid?"show-grid":""} ${tool==="pan"?"is-pannable":""}`} onClick={actOnPlan} onPointerDown={event=>{if(tool!=="pan")return;panStart.current={x:event.clientX,y:event.clientY,originX:planPan.x,originY:planPan.y};event.currentTarget.setPointerCapture(event.pointerId)}} onPointerMove={event=>{if(!panStart.current)return;setPlanPan({x:panStart.current.originX+event.clientX-panStart.current.x,y:panStart.current.originY+event.clientY-panStart.current.y})}} onPointerUp={event=>{panStart.current=null;if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId)}}><div className="plan-content" style={{transform:`translate(${planPan.x}px,${planPan.y}px) scale(${zoom})`}}><img className="plan-image" src={PLAN_IMAGE_SRC} alt="Current drawing source"/>{(overlay?.regions??[]).slice(0,28).map(region=><span key={region.id} className="region" style={{left:`${region.bbox[0]*100}%`,top:`${region.bbox[1]*100}%`,width:`${Math.max((region.bbox[2]-region.bbox[0])*100,2)}%`,height:`${Math.max((region.bbox[3]-region.bbox[1])*100,2)}%`}}/>)}{showDirections?<svg className="evidence-direction-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Evidence direction cues">{(overlay?.pins??[]).map((pin,index)=><line key={pin.id} x1={pin.x*100} y1={pin.y*100} x2={Math.min(97,pin.x*100+5+(index%3)*2)} y2={Math.max(3,pin.y*100-7-(index%2)*2)}/>)}</svg>:null}{measurePoints.length?<svg className="plan-measure-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label={measurePoints.length===2?`Relative plan measurement ${measured.toFixed(1)} percent of sheet width`:"Measurement start point selected"}>{measurePoints.length===2?<line x1={measurePoints[0].x} y1={measurePoints[0].y} x2={measurePoints[1].x} y2={measurePoints[1].y}/>:null}{measurePoints.map((point,index)=><circle key={index} cx={point.x} cy={point.y} r=".8"/>)}</svg>:null}{markups.map((markup,index)=><button type="button" className="correction-markup" key={markup.id} style={{left:`${markup.x}%`,top:`${markup.y}%`}} onClick={()=>setMarkups(current=>current.filter(item=>item.id!==markup.id))} aria-label={`Remove correction markup ${index+1}`}><Pencil size={11}/><span>Correction {index+1}</span></button>)}{showPins?(overlay?.pins??[]).map(pin=><button key={pin.id} className={`pin ${pin.severity} ${pin.id===selected?.issue_id?"active":""}`} style={{left:`${pin.x*100}%`,top:`${pin.y*100}%`}} onClick={()=>onSelectIssue(pin.id)} type="button" aria-label={`Select ${pin.label} in ${pin.room}`}><AlertTriangle size={15}/><span>{pin.label}</span></button>):null}</div></div>;
  // This pilot intentionally has one current drawing and no superseded revision,
  // so do not present an unrelated page as a revision comparison.
  const availableModes:ViewerMode[]=["2d","3d","split"];
  return <section className="drawing-workbench"><PageHeading eyebrow="SPATIAL INDEX" title="Drawings & 3D" copy="Verify source locations, issue pins, evidence, and revision context in synchronized views."/><div className="drawing-modebar"><div>{availableModes.map(item=><button type="button" className={mode===item?"active":""} onClick={()=>{setMode(item);if(item==="3d")setTool("select")}} key={item}>{item==="2d"?"2D Plan":item==="3d"?"3D Context":item==="split"?"Split View":"Revision Compare"}</button>)}</div><span>{overlay?.pins.length??0} issue pins · {mode==="3d"?"orbit / pan / zoom model":`${tool} tool`}</span></div><div className="drawing-layout">
    <aside className="sheet-rail"><strong>Current drawing set</strong><button className="active" type="button" disabled aria-current="page"><FileText size={16}/><span>{String(overlay?.sheets[0]?.title??overlay?.sheets[0]?.id??"No sheet loaded")}</span><em>Current</em></button><hr/><label><input type="checkbox" checked={showGrid} onChange={event=>setShowGrid(event.target.checked)}/> Confidence grid</label><label><input type="checkbox" checked={showPins} onChange={event=>setShowPins(event.target.checked)}/> Issue pins</label><label><input type="checkbox" checked={showDirections} onChange={event=>setShowDirections(event.target.checked)}/> Evidence direction cues</label></aside>
    <div className="viewer-shell"><div className="canvas-tools" aria-label={mode==="3d"?"3D navigation help":"Drawing tools"}>{mode==="3d"?<span className="three-control-hint">Drag to orbit · shift-drag to pan · scroll to zoom · click a marker to select</span>:<>{tools.map(([key,icon,label])=><button type="button" className={tool===key?"active":""} onClick={()=>setTool(key)} aria-label={label} title={label} key={key}>{icon}</button>)}<button type="button" onClick={()=>setZoom(value=>Math.min(2,value+.15))} aria-label="Zoom in"><ZoomIn size={18}/></button><button type="button" onClick={()=>setZoom(value=>Math.max(.6,value-.15))} aria-label="Zoom out"><ZoomOut size={18}/></button><button type="button" onClick={()=>{setZoom(1);setPlanPan({x:0,y:0});setTool("fit")}} aria-label="Fit view"><Maximize size={18}/></button>{measurePoints.length||markups.length?<button type="button" onClick={()=>{setMeasurePoints([]);setMarkups([])}} aria-label="Clear measurements and correction markups"><Trash2 size={18}/></button>:null}</>}</div>
      <div className={`viewer-canvas mode-${mode}`}>{mode==="2d"?plan:null}{mode==="3d"?<SpatialModelViewer projectId={projectId} initialAssetId={selected?.spatial_context?.design_asset_id} pins={showPins?modelPins:[]} selectedIssueId={selected?.issue_id} geometryConfidence={Number(selected?.spatial_context?.geometry_confidence??0)} onSelectIssue={onSelectIssue} fallbackImage={PLAN2FIELD_3D_SRC}/>:null}{mode==="split"?<><div className="split-pane">{plan}<span className="split-label">Current 2D source</span></div><div className="split-pane"><SpatialModelViewer compact projectId={projectId} initialAssetId={selected?.spatial_context?.design_asset_id} pins={showPins?modelPins:[]} selectedIssueId={selected?.issue_id} geometryConfidence={Number(selected?.spatial_context?.geometry_confidence??0)} onSelectIssue={onSelectIssue} fallbackImage={PLAN2FIELD_3D_SRC}/><span className="split-label">Generated 3D context</span></div></>:null}{mode==="compare"?<div className="compare-stage">{plan}<div className="compare-layer" style={{clipPath:`inset(0 ${100-compare}% 0 0)`}}><img src={PLAN2FIELD_MINIMAP_SRC} alt="Comparison revision"/></div><input aria-label="Revision comparison slider" type="range" min="0" max="100" value={compare} onChange={event=>setCompare(Number(event.target.value))}/></div>:null}</div>
      <footer className="viewer-status"><span>Mode: {mode.replaceAll("_"," ")}</span><span>{mode==="3d"?"Interactive GLB":`Zoom ${Math.round(zoom*100)}%`}</span><span>{mode==="3d"?"Model navigation is independent from the source image.":tool==="measure"?(measurePoints.length===2?`Relative measure ${measured.toFixed(1)}% of sheet width · calibrate drawing scale before contractual use.`:`${measurePoints.length?"Select the endpoint.":"Select two points on the current source."}`):tool==="markup"?`${markups.length} correction draft${markups.length===1?"":"s"} · click a markup to remove it.`:"Source coordinates preserved."}</span></footer></div>
    <aside className="drawing-inspector"><strong>Selection</strong>{selected?<><h3>{selected.title}</h3><span>{selected.issue_id} · {selected.room}</span><div className="view-verification-path"><span className={selected.requirement.source||selected.plan_location.sheet_id?"ready":"gap"}>{selected.requirement.source||selected.plan_location.sheet_id?<Check size={13}/>:<X size={13}/>} Source</span><span className={selected.evidence.length||selected.observation.media_id?"ready":"gap"}>{selected.evidence.length||selected.observation.media_id?<Check size={13}/>:<X size={13}/>} Evidence</span><span className={selected.status==="approved"?"ready":"pending"}>{selected.status==="approved"?<Check size={13}/>:<Clock3 size={13}/>} Decision</span></div><dl><div><dt>Source</dt><dd>{String(selected.requirement.source??"Unresolved")}</dd></div><div><dt>Revision</dt><dd>{String(selected.plan_location.revision??"Current")}</dd></div><div><dt>Evidence</dt><dd>{selected.evidence.length} linked</dd></div><div><dt>Location confidence</dt><dd>{Math.round(selected.confidence*100)}%</dd></div><div><dt>Geometry confidence</dt><dd>{Math.round(Number(selected.spatial_context?.geometry_confidence??0)*100)}% · derived</dd></div></dl><p>{selected.description}</p><p className="derived-geometry-note"><AlertTriangle size={14}/> 3D geometry is a navigational index derived from the current 2D source. Confirm dimensions against the cited drawing before issuing.</p><div className="inspector-actions"><button type="button" onClick={()=>setMode("2d")}><ExternalLink size={15}/> Jump to source</button><button type="button" onClick={()=>{setMode("2d");setTool("markup")}}><Pencil size={15}/> Correction mode</button><button className="primary" type="button" onClick={()=>onOpenIssue(selected.issue_id)}>Verify issue</button></div></>:<p>Select an issue pin to inspect its source and evidence.</p>}</aside>
  </div></section>
}

function OfflineQueue({ captures, online, syncing, onSync, onRemove }: { captures:QueuedCapture[]; online:boolean; syncing:boolean; onSync:()=>Promise<void>; onRemove:(id:string)=>Promise<void> }) {
  return <section className="offline-queue" aria-label="Offline queue"><header><div><span className={online?"online":"offline"}>{online?<Wifi size={17}/>:<WifiOff size={17}/>}</span><div><b>Offline queue</b><small>{captures.length?`${captures.length} original capture${captures.length===1?"":"s"} stored on this device.`:"All locally saved captures are synced."}</small></div></div><button type="button" onClick={()=>void onSync()} disabled={!online||syncing||!captures.length}>{syncing?<Loader2 className="spin" size={16}/>:<RefreshCcw size={16}/>} Sync now</button></header>{captures.length?<div className="queue-list">{captures.map(item=><article key={item.id}><span className={`queue-state ${item.state}`}>{item.state}</span><div><b>{item.filename}</b><small>{item.metadata.floor} / {item.metadata.room||"Unlinked"} · {item.metadata.intent}</small>{item.error?<em>{item.error}</em>:null}</div><time>{new Date(item.createdAt).toLocaleString()}</time><button type="button" aria-label={`Remove ${item.filename} from offline queue`} onClick={()=>void onRemove(item.id)}><Trash2 size={16}/></button></article>)}</div>:null}</section>
}

function ReportsView({ issues, documents, mediaAssets, selectedIssue, reports, rfi, reportUrl, busy, onGenerate, onExport, onSelectIssue }: { issues:Issue[]; documents:DocumentAsset[]; mediaAssets:SiteMediaAsset[]; selectedIssue?:Issue; reports:ReportRecord[]; rfi:string; reportUrl:string; busy:boolean; onGenerate:(type:"punch"|"co_evidence"|"rfi",format:"pdf"|"csv",issueIds:string[])=>Promise<void>; onExport:(id:string)=>Promise<void>; onSelectIssue:(id:string)=>void }) {
  const [type,setType]=useState<"punch"|"co_evidence"|"rfi">("punch");
  const [format,setFormat]=useState<"pdf"|"csv">("pdf");
  const [selectedIds,setSelectedIds]=useState<string[]>(()=>selectedIssue?[selectedIssue.issue_id]:[]);

  useEffect(()=>{
    setSelectedIds(current=>{
      const valid=current.filter(id=>issues.some(issue=>issue.issue_id===id));
      if(valid.length) return type==="rfi"?[valid[0]]:valid;
      return selectedIssue?[selectedIssue.issue_id]:issues[0]?[issues[0].issue_id]:[];
    });
  },[issues,selectedIssue?.issue_id,type]);

  const selected=issues.filter(issue=>selectedIds.includes(issue.issue_id));
  const active=selected[0]??selectedIssue;
  const requirements=selected.flatMap(issue=>[
    {id:`${issue.issue_id}-source`,label:`${issueCode(issue)} current source`,ready:Boolean(issue.requirement.source||issue.plan_location.sheet_id),blocking:true},
    {id:`${issue.issue_id}-location`,label:`${issueCode(issue)} confirmed location`,ready:Boolean(issue.room&&issue.room.toLowerCase()!=="unknown"),blocking:true},
    {id:`${issue.issue_id}-evidence`,label:`${issueCode(issue)} field evidence`,ready:Boolean(issue.evidence.length||issue.observation.media_id),blocking:true},
    ...(type==="punch"?[{id:`${issue.issue_id}-owner`,label:`${issueCode(issue)} responsible party`,ready:Boolean(issue.assignee||issue.subcontractor),blocking:true},{id:`${issue.issue_id}-due`,label:`${issueCode(issue)} due date`,ready:Boolean(issue.due_date),blocking:true}]:[]),
    {id:`${issue.issue_id}-review`,label:`${issueCode(issue)} human approval`,ready:issue.status==="approved",blocking:false}
  ]);
  const blocked=!selected.length||requirements.some(item=>item.blocking&&!item.ready);
  const preview=type==="rfi"?(rfi||(active?buildRfiPreview(active):"")):selected.map(issue=>`${issueCode(issue)} · ${issue.title} · ${issue.room}`).join("\n");
  const typeLabel=type==="punch"?"Punch list":type==="rfi"?"RFI":"Potential change evidence";

  function toggleIssue(id:string){
    setSelectedIds(current=>type==="rfi"?[id]:current.includes(id)?current.filter(item=>item!==id):[...current,id]);
  }

  return <section className="standard-page reports-page">
    <PageHeading eyebrow="SOURCE-BACKED OUTPUT" title="Reports" copy="Build a reviewable package first. Issuance stays blocked until its human and source gates are complete." action={<button className="page-primary" type="button" disabled={busy||blocked} onClick={()=>void onGenerate(type,format,selectedIds)}><FileDown size={17}/>{busy?"Generating…":"Generate report"}</button>}/>
    <div className="report-builder-controls"><label>Output<select value={type} onChange={event=>setType(event.target.value as typeof type)}><option value="punch">Punch list</option><option value="rfi">RFI</option><option value="co_evidence">Potential change evidence</option></select></label><label>Format<select value={format} onChange={event=>setFormat(event.target.value as typeof format)}><option value="pdf">PDF package</option><option value="csv">CSV issue log</option></select></label><span><ShieldCheck size={17}/> Current source and evidence are snapshotted at generation.</span></div>
    <div className="report-composer">
      <section className="report-issue-picker"><header><div><p>1 · SELECT RECORDS</p><h2>{type==="rfi"?"Choose one issue":"Choose issue rows"}</h2></div><span>{selected.length} selected</span></header><div>{issues.map(issue=><label className={selectedIds.includes(issue.issue_id)?"selected":""} key={issue.issue_id}><input type={type==="rfi"?"radio":"checkbox"} name={type==="rfi"?"rfi-issue":undefined} checked={selectedIds.includes(issue.issue_id)} onChange={()=>toggleIssue(issue.issue_id)}/><span className={`severity-dot ${issue.severity}`}/><div><b>{issueCode(issue)} · {issue.title}</b><small>{issue.room} · {String(issue.requirement.source??issue.plan_location.sheet_id??"Source missing")}</small></div><em className={issue.status}>{statusLabel(issue.status)}</em></label>)}</div></section>
      <aside className="report-readiness"><header><p>2 · VALIDATE</p><h2>Package readiness</h2></header>{selected.length?<div className="report-checks">{requirements.map(item=><span className={item.ready?"ready":item.blocking?"blocked":"pending"} key={item.id}>{item.ready?<Check size={14}/>:item.blocking?<X size={14}/>:<Clock3 size={14}/>}<b>{item.label}</b>{!item.blocking&&!item.ready?<small>Required to issue</small>:null}</span>)}</div>:<p className="empty-report-selection">Select at least one issue to validate.</p>}<div className={`report-gate ${blocked?"blocked":"ready"}`}>{blocked?<AlertTriangle size={17}/>:<ShieldCheck size={17}/>}<span><b>{blocked?"Draft blocked":"Ready to generate"}</b><small>{blocked?"Resolve the red source, location, ownership, or evidence gaps.":"The generated draft will retain this exact evidence snapshot."}</small></span></div></aside>
    </div>
    <section className="report-output-preview"><header><div><p>3 · REVIEW OUTPUT</p><h2>{typeLabel} preview</h2></div><span>{documents.length} source file{documents.length===1?"":"s"} · {mediaAssets.length} media record{mediaAssets.length===1?"":"s"}</span></header><pre>{preview||"Select an issue to preview the package."}</pre>{active?<button type="button" onClick={()=>onSelectIssue(active.issue_id)}>Open issue and evidence <ExternalLink size={14}/></button>:null}</section>
    {reportUrl?<a className="download-link" href={reportUrl} target="_blank" rel="noreferrer"><FileDown size={18}/>Download latest generated report</a>:null}
    <section className="report-history-section"><header><p>VERSION HISTORY</p><h2>Generated and issued packages</h2></header><div className="report-history" aria-label="Report history">{reports.length?reports.map(report=>{const snapshot=report.issue_snapshot??[];const issueBlocked=!snapshot.length||snapshot.some(frozen=>{const current=issues.find(issue=>issue.issue_id===frozen.issue_id);return current?current.status!=="approved"||!Boolean(current.requirement.source||current.plan_location.sheet_id):(frozen.workflow?.review_status??frozen.status)!=="approved"||Boolean(frozen.workflow?.source_status&&frozen.workflow.source_status!=="current")});return <article key={report.report_id}><span className={`report-state ${report.status??"draft"}`}>{report.status??"draft"}</span><div><b>{report.title||report.report_type.replaceAll("_"," ")}</b><small>{report.report_id} · {report.format??"versioned output"} · {snapshot.length} locked issue{snapshot.length===1?"":"s"} · {report.created_by??"system"}</small></div>{report.download_url?<a href={report.download_url} target="_blank" rel="noreferrer" aria-label={`Download ${report.title??report.report_type}`}><FileDown size={16}/></a>:null}<button type="button" onClick={()=>void onExport(report.report_id)} disabled={busy||issueBlocked||["issued","synced"].includes(report.status??"")} title={issueBlocked?"Every issue in this locked report scope requires a current source and human approval":"Issue this locked report version"}><Send size={15}/> Issue / export</button></article>}):<p className="table-empty">No generated reports yet.</p>}</div></section>
  </section>;
}

function SearchPalette({ query, scope, historical, results, busy, onQuery, onScope, onHistorical, onSearch, onClose, onOpen }: { query:string; scope:SearchScope; historical:boolean; results:SearchResult[]; busy:boolean; onQuery:(value:string)=>void; onScope:(value:SearchScope)=>void; onHistorical:(value:boolean)=>void; onSearch:(event?:FormEvent)=>Promise<void>; onClose:()=>void; onOpen:(result:SearchResult)=>void }) {
  useEffect(()=>{const close=(event:KeyboardEvent)=>{if(event.key==="Escape")onClose()};window.addEventListener("keydown",close);return()=>window.removeEventListener("keydown",close)},[onClose]);
  return <div className="search-backdrop" role="presentation" onMouseDown={event=>event.target===event.currentTarget&&onClose()}><section className="search-palette" role="dialog" aria-modal="true" aria-label="Universal search"><form onSubmit={onSearch}><Search size={20}/><input autoFocus value={query} onChange={event=>onQuery(event.target.value)} placeholder="Search current sources, issues, evidence, and people" aria-label="Search query"/><button type="submit" disabled={busy||query.trim().length<2}>{busy?<Loader2 className="spin" size={17}/>:"Search"}</button><button type="button" onClick={onClose} aria-label="Close search"><X size={18}/></button></form><div className="search-filters"><button type="button" className={scope==="project"?"active":""} onClick={()=>onScope("project")}>Current project</button><button type="button" className={scope==="organization"?"active":""} onClick={()=>onScope("organization")}>Organization</button><label><input type="checkbox" checked={historical} onChange={event=>onHistorical(event.target.checked)}/> Include historical sources</label></div><div className="search-results">{results.map(result=><button type="button" key={`${result.type}-${result.id}`} onClick={()=>onOpen(result)}><span className="result-type">{result.type}</span><div><b>{result.title}</b><small>{result.subtitle||result.location||result.revision||result.status}</small><p>{result.snippet}</p></div><ArrowRight size={17}/></button>)}{!busy&&!results.length?<p>{query.trim().length<2?"Enter at least two characters.":"Run search to see current-project results."}</p>:null}</div></section></div>
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

function arrayBufferToBase64(buffer: ArrayBuffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, Math.min(offset + chunkSize, bytes.length)));
  }
  return btoa(binary);
}

async function sha256Hex(buffer: ArrayBuffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
