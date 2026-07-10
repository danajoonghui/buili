"use client";

import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Camera,
  Check,
  FileText,
  MapPin,
  Mic,
  Ruler,
  Upload,
  Video,
  X
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { CaptureIntent, CaptureMetadata } from "@/lib/offlineQueue";
import { Issue, Project } from "@/lib/api";

export function Dialog({
  open,
  title,
  description,
  children,
  onClose,
  size = "medium"
}: {
  open: boolean;
  title: string;
  description?: string;
  children: ReactNode;
  onClose: () => void;
  size?: "small" | "medium" | "large";
}) {
  const titleRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const timer = window.setTimeout(() => titleRef.current?.focus(), 0);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.clearTimeout(timer);
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [onClose, open]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className={`dialog-panel ${size}`} role="dialog" aria-modal="true" aria-labelledby="dialog-title" aria-describedby={description ? "dialog-description" : undefined}>
        <header className="dialog-header">
          <div>
            <h2 id="dialog-title" ref={titleRef} tabIndex={-1}>{title}</h2>
            {description ? <p id="dialog-description">{description}</p> : null}
          </div>
          <button className="dialog-close" type="button" onClick={onClose} aria-label={`Close ${title}`}>
            <X size={20} />
          </button>
        </header>
        {children}
      </section>
    </div>
  );
}

export type ProjectWizardValue = {
  name: string;
  address: string;
  client: string;
  projectType: string;
  timezone: string;
  building: string;
  floors: string;
  units: "imperial" | "metric";
  grid: string;
  company: string;
  approverName: string;
  approverEmail: string;
  trade: string;
  files: File[];
  scale: string;
  north: string;
  floorHeight: string;
  origin: string;
  reviewRoute: string;
  reportTemplate: string;
};

const WIZARD_STEPS = ["Basics", "Building", "Team", "Files", "Spatial setup", "Workflow", "Review"];

const INITIAL_PROJECT: ProjectWizardValue = {
  name: "",
  address: "",
  client: "",
  projectType: "tenant_improvement",
  timezone: "America/Los_Angeles",
  building: "Main Building",
  floors: "Level 01",
  units: "imperial",
  grid: "",
  company: "",
  approverName: "",
  approverEmail: "",
  trade: "General Contractor",
  files: [],
  scale: "Assumed from drawing metadata",
  north: "0",
  floorHeight: "10 ft",
  origin: "Southwest corner",
  reviewRoute: "Project Manager approval",
  reportTemplate: "BUILI standard"
};

export function ProjectWizard({
  open,
  projects,
  busy,
  onClose,
  onCreate
}: {
  open: boolean;
  projects: Project[];
  busy: boolean;
  onClose: () => void;
  onCreate: (value: ProjectWizardValue) => Promise<void>;
}) {
  const [step, setStep] = useState(0);
  const [value, setValue] = useState<ProjectWizardValue>(INITIAL_PROJECT);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setStep(0);
    setValue(INITIAL_PROJECT);
    setError("");
  }, [open]);

  function update<K extends keyof ProjectWizardValue>(key: K, next: ProjectWizardValue[K]) {
    setValue((current) => ({ ...current, [key]: next }));
  }

  function validate(index: number) {
    if (index === 0) {
      if (!value.name.trim() || !value.address.trim() || !value.client.trim()) return "Project name, address, and client are required.";
      if (projects.some((item) => item.name.trim().toLowerCase() === value.name.trim().toLowerCase())) return "A project with this name already exists.";
    }
    if (index === 1 && (!value.building.trim() || !value.floors.trim())) return "Add a building and at least one floor or zone.";
    if (index === 2 && (!value.company.trim() || !value.approverName.trim() || !/^\S+@\S+\.\S+$/.test(value.approverEmail))) return "Assign one approver with a valid email address.";
    if (index === 4 && (!value.scale.trim() || !value.floorHeight.trim() || !value.origin.trim())) return "Scale, floor height, and coordinate origin are required.";
    return "";
  }

  function next() {
    const validation = validate(step);
    if (validation) {
      setError(validation);
      return;
    }
    setError("");
    setStep((current) => Math.min(WIZARD_STEPS.length - 1, current + 1));
  }

  async function submit() {
    const validation = WIZARD_STEPS.map((_, index) => validate(index)).find(Boolean);
    if (validation) {
      setError(validation);
      return;
    }
    try {
      await onCreate(value);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Project creation failed.");
    }
  }

  return (
    <Dialog open={open} title="Create project" description="Set up a review-ready workspace. You can refine every setting later." onClose={onClose} size="large">
      <div className="wizard-layout">
        <ol className="wizard-steps" aria-label="Project setup progress">
          {WIZARD_STEPS.map((label, index) => (
            <li key={label} className={index === step ? "active" : index < step ? "complete" : ""}>
              <button type="button" onClick={() => index <= step && setStep(index)} aria-current={index === step ? "step" : undefined}>
                <span>{index < step ? <Check size={13} /> : index + 1}</span>{label}
              </button>
            </li>
          ))}
        </ol>
        <div className="wizard-content">
          <p className="form-eyebrow">STEP {step + 1} OF {WIZARD_STEPS.length}</p>
          <h3>{WIZARD_STEPS[step]}</h3>
          {step === 0 ? <div className="form-grid two">
            <label className="span-two">Project name<input autoFocus value={value.name} onChange={(event) => update("name", event.target.value)} required /></label>
            <label className="span-two">Site address<input value={value.address} onChange={(event) => update("address", event.target.value)} required /></label>
            <label>Client<input value={value.client} onChange={(event) => update("client", event.target.value)} required /></label>
            <label>Project type<select value={value.projectType} onChange={(event) => update("projectType", event.target.value)}><option value="tenant_improvement">Tenant improvement</option><option value="ground_up">Ground-up</option><option value="renovation">Renovation</option><option value="closeout">Closeout</option></select></label>
            <label className="span-two">Timezone<select value={value.timezone} onChange={(event) => update("timezone", event.target.value)}><option value="America/Los_Angeles">Pacific Time (US)</option><option value="America/Denver">Mountain Time (US)</option><option value="America/Chicago">Central Time (US)</option><option value="America/New_York">Eastern Time (US)</option><option value="Asia/Seoul">Korea Standard Time</option></select></label>
          </div> : null}
          {step === 1 ? <div className="form-grid two">
            <label>Building name<input autoFocus value={value.building} onChange={(event) => update("building", event.target.value)} /></label>
            <label>Units<select value={value.units} onChange={(event) => update("units", event.target.value as ProjectWizardValue["units"])}><option value="imperial">Imperial</option><option value="metric">Metric</option></select></label>
            <label className="span-two">Floors / zones<textarea value={value.floors} onChange={(event) => update("floors", event.target.value)} placeholder="One floor or zone per line" /></label>
            <label className="span-two">Grid (optional)<input value={value.grid} onChange={(event) => update("grid", event.target.value)} placeholder="A–F / 1–8" /></label>
          </div> : null}
          {step === 2 ? <div className="form-grid two">
            <label>Company<input autoFocus value={value.company} onChange={(event) => update("company", event.target.value)} /></label>
            <label>Trade<select value={value.trade} onChange={(event) => update("trade", event.target.value)}><option>General Contractor</option><option>Architect</option><option>Electrical</option><option>Mechanical</option><option>Plumbing</option></select></label>
            <label>Approver name<input value={value.approverName} onChange={(event) => update("approverName", event.target.value)} /></label>
            <label>Approver email<input type="email" value={value.approverEmail} onChange={(event) => update("approverEmail", event.target.value)} /></label>
            <p className="inline-note span-two"><Check size={16} /> The approver is the human gate for official reports and external actions.</p>
          </div> : null}
          {step === 3 ? <div className="wizard-upload">
            <label className="drop-field"><Upload size={24}/><b>Add current drawings, specifications, or historical RFIs</b><span>PDF, DOCX, XLSX, CSV, images, video, or audio</span><input autoFocus type="file" multiple accept=".pdf,.docx,.txt,.csv,.xlsx,image/*,video/*,audio/*" onChange={(event) => update("files", Array.from(event.target.files ?? []))}/></label>
            {value.files.length ? <ul className="selected-files">{value.files.map((file) => <li key={`${file.name}-${file.size}`}><FileText size={16}/><span>{file.name}</span><small>{formatBytes(file.size)}</small></li>)}</ul> : <p className="empty-inline">Files are optional during setup. Uploading now starts intake immediately after activation.</p>}
          </div> : null}
          {step === 4 ? <div className="form-grid two">
            <label className="span-two">Drawing scale<select autoFocus value={value.scale} onChange={(event) => update("scale", event.target.value)}><option>Assumed from drawing metadata</option><option>Verified manually</option><option>Not known — correction required</option></select></label>
            <label>North rotation (degrees)<input type="number" value={value.north} onChange={(event) => update("north", event.target.value)} /></label>
            <label>Typical floor height<input value={value.floorHeight} onChange={(event) => update("floorHeight", event.target.value)} /></label>
            <label className="span-two">Coordinate origin<input value={value.origin} onChange={(event) => update("origin", event.target.value)} /></label>
            <p className="inline-note warning span-two"><AlertTriangle size={16}/> Assumed geometry stays visibly distinct until a reviewer verifies it.</p>
          </div> : null}
          {step === 5 ? <div className="form-grid">
            <label>Review route<select autoFocus value={value.reviewRoute} onChange={(event) => update("reviewRoute", event.target.value)}><option>Project Manager approval</option><option>Project Engineer, then Project Manager</option><option>Discipline lead, then Project Manager</option></select></label>
            <label>Default report template<select value={value.reportTemplate} onChange={(event) => update("reportTemplate", event.target.value)}><option>BUILI standard</option><option>Contractor punch</option><option>Owner evidence package</option></select></label>
            <label className="check-field"><input type="checkbox" defaultChecked/> Require a second reviewer for stale sources, cost impact, or low location confidence.</label>
          </div> : null}
          {step === 6 ? <div className="review-summary">
            <dl><div><dt>Project</dt><dd>{value.name}</dd></div><div><dt>Client</dt><dd>{value.client}</dd></div><div><dt>Building</dt><dd>{value.building} · {value.floors.split("\n").filter(Boolean).length} floor(s)</dd></div><div><dt>Approver</dt><dd>{value.approverName} · {value.approverEmail}</dd></div><div><dt>Files</dt><dd>{value.files.length} selected</dd></div><div><dt>Spatial basis</dt><dd>{value.scale} · {value.origin}</dd></div><div><dt>Workflow</dt><dd>{value.reviewRoute}</dd></div></dl>
            {!value.files.length ? <p className="inline-note warning"><AlertTriangle size={16}/> No files selected. The project can activate, but verification remains blocked until a current drawing set is uploaded.</p> : null}
          </div> : null}
          {error ? <p className="form-error" role="alert">{error}</p> : null}
          <footer className="dialog-actions">
            <button type="button" className="text-button" onClick={onClose}>Cancel</button>
            <span />
            {step > 0 ? <button type="button" onClick={() => { setError(""); setStep((current) => current - 1); }}><ArrowLeft size={16}/> Back</button> : null}
            {step < WIZARD_STEPS.length - 1 ? <button type="button" className="primary" onClick={next}>Continue <ArrowRight size={16}/></button> : <button type="button" className="primary" onClick={submit} disabled={busy}>{busy ? "Creating…" : "Activate project"}</button>}
          </footer>
        </div>
      </div>
    </Dialog>
  );
}

export type UploadMetadata = {
  documentType: "plan" | "spec" | "rfi" | "submittal" | "addendum" | "other" | "media";
  revision: string;
  issueDate: string;
  discipline: string;
  setStatus: "current" | "historical";
};

export function UploadDialog({
  open,
  busy,
  onClose,
  onUpload
}: {
  open: boolean;
  busy: boolean;
  onClose: () => void;
  onUpload: (files: File[], metadata: UploadMetadata) => Promise<void>;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [metadata, setMetadata] = useState<UploadMetadata>({ documentType: "plan", revision: "", issueDate: "", discipline: "architectural", setStatus: "current" });
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setFiles([]);
    setMetadata({ documentType: "plan", revision: "", issueDate: "", discipline: "architectural", setStatus: "current" });
    setError("");
  }, [open]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!files.length) return setError("Choose at least one file.");
    const documentsOnly = files.some((file) => !file.type.startsWith("image/") && !file.type.startsWith("video/") && !file.type.startsWith("audio/"));
    if (documentsOnly && (!metadata.revision.trim() || !metadata.issueDate)) return setError("Revision and issue date are required for documents. Otherwise, classify them as Unclassified after upload.");
    setError("");
    try {
      await onUpload(files, metadata);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed.");
    }
  }

  return <Dialog open={open} title="Upload project files" description="Classify the intake so current and historical sources never mix silently." onClose={onClose} size="medium">
    <form className="dialog-form" onSubmit={submit}>
      <label className="drop-field"><Upload size={24}/><b>Choose files</b><span>Multiple files up to the server upload limit</span><input autoFocus type="file" multiple accept=".pdf,.docx,.txt,.csv,.xlsx,image/*,video/*,audio/*" onChange={(event) => setFiles(Array.from(event.target.files ?? []))}/></label>
      {files.length ? <ul className="selected-files compact">{files.map((file) => <li key={`${file.name}-${file.size}`}><FileText size={16}/><span>{file.name}</span><small>{formatBytes(file.size)}</small></li>)}</ul> : null}
      <div className="form-grid two">
        <label>Document type<select value={metadata.documentType} onChange={(event) => setMetadata((current) => ({ ...current, documentType: event.target.value as UploadMetadata["documentType"] }))}><option value="plan">Drawing / plan</option><option value="spec">Specification</option><option value="rfi">RFI</option><option value="submittal">Submittal</option><option value="addendum">Addendum</option><option value="other">Other document</option><option value="media">Field media</option></select></label>
        <label>Discipline<select value={metadata.discipline} onChange={(event) => setMetadata((current) => ({ ...current, discipline: event.target.value }))}><option value="architectural">Architectural</option><option value="structural">Structural</option><option value="mechanical">Mechanical</option><option value="electrical">Electrical</option><option value="plumbing">Plumbing</option><option value="civil">Civil</option><option value="general">General</option></select></label>
        <label>Revision<input value={metadata.revision} onChange={(event) => setMetadata((current) => ({ ...current, revision: event.target.value }))} placeholder="A, 03, IFC…" /></label>
        <label>Issue date<input type="date" value={metadata.issueDate} onChange={(event) => setMetadata((current) => ({ ...current, issueDate: event.target.value }))}/></label>
        <label className="span-two">Set status<select value={metadata.setStatus} onChange={(event) => setMetadata((current) => ({ ...current, setStatus: event.target.value as UploadMetadata["setStatus"] }))}><option value="current">Current set — may supersede an earlier revision</option><option value="historical">Historical — searchable only when explicitly included</option></select></label>
      </div>
      {error ? <p className="form-error" role="alert">{error}</p> : null}
      <footer className="dialog-actions"><button type="button" className="text-button" onClick={onClose}>Cancel</button><span/><button className="primary" type="submit" disabled={busy}>{busy ? "Uploading…" : `Upload ${files.length || ""} file${files.length === 1 ? "" : "s"}`}</button></footer>
    </form>
  </Dialog>;
}

const CAPTURE_STEPS = ["Location", "Evidence", "Review"];

export function CaptureDialog({
  open,
  projectName,
  online,
  busy,
  onClose,
  onSave
}: {
  open: boolean;
  projectName: string;
  online: boolean;
  busy: boolean;
  onClose: () => void;
  onSave: (file: File, metadata: CaptureMetadata) => Promise<void>;
}) {
  const [step, setStep] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [intent, setIntent] = useState<CaptureIntent>("observation");
  const [metadata, setMetadata] = useState<Omit<CaptureMetadata, "intent">>({ floor: "Main Floor", room: "", trade: "General", note: "", mediaType: "photo", measurement: "", source: "", locationMethod: "confirmed" });
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    setStep(0); setFile(null); setPreview(""); setIntent("observation"); setError("");
    setMetadata({ floor: "Main Floor", room: "", trade: "General", note: "", mediaType: "photo", measurement: "", source: "", locationMethod: "confirmed" });
  }, [open]);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  function chooseFile(next: File | null) {
    if (preview) URL.revokeObjectURL(preview);
    setFile(next);
    setPreview(next && next.type.startsWith("image/") ? URL.createObjectURL(next) : "");
  }

  function next() {
    if (step === 0 && metadata.locationMethod !== "unlinked" && (!metadata.floor.trim() || !metadata.room.trim())) return setError("Confirm a floor and room, or save this capture as unlinked evidence.");
    if (step === 1 && !file && !metadata.measurement?.trim()) return setError("Capture a photo, video, voice note, or measurement.");
    setError(""); setStep((current) => Math.min(2, current + 1));
  }

  async function save() {
    let uploadFile = file;
    if (!uploadFile) {
      const body = JSON.stringify({ measurement: metadata.measurement, note: metadata.note, captured_at: new Date().toISOString() });
      uploadFile = new File([body], `measurement-${Date.now()}.json`, { type: "application/json" });
    }
    try {
      await onSave(uploadFile, { ...metadata, intent });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The capture could not be saved locally.");
    }
  }

  const sufficiency = useMemo(() => [
    { label: "Location", ready: metadata.locationMethod === "unlinked" || Boolean(metadata.floor && metadata.room), message: "Confirm floor and room." },
    { label: "Context", ready: Boolean(file), message: "Add one wide-angle or narrated media capture." },
    { label: "Detail", ready: Boolean(file || metadata.measurement), message: "Add a clear detail or measurement." },
    { label: "Source", ready: Boolean(metadata.source), message: "Link a current drawing or specification when available." }
  ], [file, metadata]);

  return <Dialog open={open} title="Field capture" description={`${projectName} · ${online ? "Online" : "Offline — local save is available"}`} onClose={onClose} size="medium">
    <div className="capture-flow">
      <ol className="capture-progress" aria-label="Capture progress">{CAPTURE_STEPS.map((label,index)=><li className={index===step?"active":index<step?"complete":""} key={label}><span>{index<step?<Check size={13}/>:index+1}</span>{label}</li>)}</ol>
      {step === 0 ? <div className="capture-step">
        <h3><MapPin size={20}/> Confirm location</h3>
        <label>Project<input value={projectName} readOnly /></label>
        <div className="form-grid two"><label>Floor<select autoFocus value={metadata.floor} onChange={(event)=>setMetadata(current=>({...current,floor:event.target.value}))}><option>Main Floor</option><option>Upper Floor</option><option>Basement</option><option>Roof</option></select></label><label>Room / zone<input value={metadata.room} onChange={(event)=>setMetadata(current=>({...current,room:event.target.value}))} placeholder="Garage · East wall" /></label></div>
        <div className="choice-row"><button type="button" className={metadata.locationMethod==="recent"?"active":""} onClick={()=>setMetadata(current=>({...current,locationMethod:"recent",room:current.room||"Garage · East wall near entry door"}))}>Use recent area</button><button type="button" className={metadata.locationMethod==="qr"?"active":""} onClick={()=>setMetadata(current=>({...current,locationMethod:"qr",room:current.room||"QR-verified zone"}))}>Scan result</button><button type="button" className={metadata.locationMethod==="unlinked"?"active":""} onClick={()=>setMetadata(current=>({...current,locationMethod:"unlinked"}))}>Save unlinked</button></div>
      </div> : null}
      {step === 1 ? <div className="capture-step">
        <h3><Camera size={20}/> Add evidence</h3>
        <div className="media-choice" role="group" aria-label="Evidence type">{([{key:"photo",icon:Camera,label:"Photo"},{key:"video",icon:Video,label:"Video"},{key:"voice",icon:Mic,label:"Voice"},{key:"measurement",icon:Ruler,label:"Measure"}] as const).map(item=><button type="button" className={metadata.mediaType===item.key?"active":""} onClick={()=>setMetadata(current=>({...current,mediaType:item.key}))} key={item.key}><item.icon size={21}/>{item.label}</button>)}</div>
        {metadata.mediaType !== "measurement" ? <label className="capture-file"><input autoFocus type="file" capture={metadata.mediaType === "photo" ? "environment" : undefined} accept={metadata.mediaType === "photo" ? "image/*" : metadata.mediaType === "video" ? "video/*" : "audio/*"} onChange={(event)=>chooseFile(event.target.files?.[0]??null)}/>{preview?<img src={preview} alt="Capture preview"/>:<span><Upload size={24}/><b>{file?.name ?? `Choose or capture ${metadata.mediaType}`}</b></span>}</label> : <label>Measurement<input autoFocus value={metadata.measurement} onChange={(event)=>setMetadata(current=>({...current,measurement:event.target.value}))} placeholder="e.g. 31 7/8 in clear opening" /></label>}
        <div className="form-grid two"><label>Trade<select value={metadata.trade} onChange={(event)=>setMetadata(current=>({...current,trade:event.target.value}))}><option>General</option><option>Architectural</option><option>Electrical</option><option>Mechanical</option><option>Plumbing</option><option>Structural</option></select></label><label>Current source (optional)<input value={metadata.source} onChange={(event)=>setMetadata(current=>({...current,source:event.target.value}))} placeholder="E-204 Rev 3" /></label></div>
        <label>Short note<textarea value={metadata.note} onChange={(event)=>setMetadata(current=>({...current,note:event.target.value}))} placeholder="Describe only what is visible or measured." /></label>
      </div> : null}
      {step === 2 ? <div className="capture-step review-capture">
        <h3><Check size={20}/> Evidence sufficiency</h3>
        <ul className="sufficiency-list">{sufficiency.map(item=><li className={item.ready?"ready":"gap"} key={item.label}><span>{item.ready?<Check size={15}/>:<AlertTriangle size={15}/>}</span><div><b>{item.label}</b><small>{item.ready?"Ready":item.message}</small></div></li>)}</ul>
        <label>Save as<select autoFocus value={intent} onChange={(event)=>setIntent(event.target.value as CaptureIntent)}><option value="observation">Observation — keep in evidence library</option><option value="issue">Issue draft — route to review</option></select></label>
        <p className="local-save-note"><Check size={17}/><span><b>Local save happens first.</b> {online ? "BUILI will sync the original immediately and keep it queued until confirmed." : "The original stays on this device and syncs when connectivity returns."}</span></p>
      </div> : null}
      {error ? <p className="form-error" role="alert">{error}</p> : null}
      <footer className="dialog-actions capture-actions"><button type="button" className="text-button" onClick={onClose}>Cancel</button><span/>{step>0?<button type="button" onClick={()=>{setError("");setStep(current=>current-1)}}><ArrowLeft size={16}/> Back</button>:null}{step<2?<button type="button" className="primary" onClick={next}>Continue <ArrowRight size={16}/></button>:<button type="button" className="primary" onClick={save} disabled={busy}>{busy?"Saving locally…":intent==="issue"?"Create issue draft":"Save observation"}</button>}</footer>
    </div>
  </Dialog>;
}

export type IssueEditorValue = {
  title: string;
  type: string;
  discipline: string;
  severity: string;
  floor: string;
  room: string;
  description: string;
  expected: string;
  source: string;
  recommendedAction: string;
  assignee: string;
  dueDate: string;
  route: string;
};

function issueToEditor(issue?: Issue): IssueEditorValue {
  return {
    title: issue?.title ?? "",
    type: issue?.type ?? "field_observation",
    discipline: issue?.discipline ?? "general",
    severity: issue?.severity ?? "minor",
    floor: String(issue?.plan_location.level ?? issue?.plan_location.floor ?? "Main Floor"),
    room: issue?.room ?? "",
    description: issue?.description ?? "",
    expected: String(issue?.requirement.text ?? ""),
    source: String(issue?.requirement.source ?? ""),
    recommendedAction: issue?.recommended_action ?? "Verify condition and route after source review.",
    assignee: issue?.assignee ?? "",
    dueDate: issue?.due_date ?? "",
    route: String(issue?.requirement.route ?? "more_evidence")
  };
}

export function IssueEditorDialog({ open, issue, busy, onClose, onSave }: { open: boolean; issue?: Issue; busy: boolean; onClose: () => void; onSave: (value: IssueEditorValue) => Promise<void> }) {
  const [value, setValue] = useState<IssueEditorValue>(() => issueToEditor(issue));
  const [error, setError] = useState("");
  useEffect(() => { if (open) { setValue(issueToEditor(issue)); setError(""); } }, [issue, open]);
  function update<K extends keyof IssueEditorValue>(key: K, next: IssueEditorValue[K]) { setValue((current)=>({...current,[key]:next})); }
  async function submit(event: FormEvent) { event.preventDefault(); if (!value.title.trim() || !value.room.trim() || !value.description.trim()) return setError("Title, room, and observed condition are required."); if (value.route !== "observation" && (!value.expected.trim() || !value.source.trim())) return setError("An actionable issue needs an expected condition and current source citation."); setError(""); try { await onSave(value); } catch (cause) { setError(cause instanceof Error ? cause.message : "Issue save failed."); } }
  return <Dialog open={open} title={issue ? "Edit issue" : "New issue"} description="Separate observed facts from requirements and recommended action." onClose={onClose} size="large"><form className="dialog-form" onSubmit={submit}><div className="form-grid two">
    <label className="span-two">Issue title<input autoFocus value={value.title} onChange={(event)=>update("title",event.target.value)} placeholder="Observed condition, not an assumed solution" /></label>
    <label>Type<select value={value.type} onChange={(event)=>update("type",event.target.value)}><option value="field_observation">Field observation</option><option value="coverage_check">Coverage check</option><option value="dimension_mismatch">Dimension mismatch</option><option value="location_mismatch">Location mismatch</option><option value="clearance_check">Clearance check</option></select></label>
    <label>Discipline<select value={value.discipline} onChange={(event)=>update("discipline",event.target.value)}><option value="general">General</option><option value="architectural">Architectural</option><option value="structural">Structural</option><option value="mechanical">Mechanical</option><option value="electrical">Electrical</option><option value="plumbing">Plumbing</option></select></label>
    <label>Floor<input value={value.floor} onChange={(event)=>update("floor",event.target.value)} /></label><label>Room / zone<input value={value.room} onChange={(event)=>update("room",event.target.value)} /></label>
    <label>Priority<select value={value.severity} onChange={(event)=>update("severity",event.target.value)}><option value="blocker">Blocker</option><option value="major">Major</option><option value="minor">Minor</option><option value="informational">Informational</option></select></label><label>Recommended route<select value={value.route} onChange={(event)=>update("route",event.target.value)}><option value="more_evidence">More evidence</option><option value="punch">Punch item</option><option value="rfi">RFI</option><option value="pce">Potential change</option><option value="observation">Observation only</option></select></label>
    <label className="span-two">Observed condition<textarea value={value.description} onChange={(event)=>update("description",event.target.value)} placeholder="What can be verified in the attached evidence?" /></label>
    <label className="span-two">Expected condition<textarea value={value.expected} onChange={(event)=>update("expected",event.target.value)} placeholder="Exact requirement from the source" /></label>
    <label>Source citation<input value={value.source} onChange={(event)=>update("source",event.target.value)} placeholder="A-101 Rev 3, Detail 5" /></label><label>Assignee<input value={value.assignee} onChange={(event)=>update("assignee",event.target.value)} placeholder="Person or trade queue" /></label>
    <label>Due date<input type="date" value={value.dueDate} onChange={(event)=>update("dueDate",event.target.value)} /></label><label>Recommended action<input value={value.recommendedAction} onChange={(event)=>update("recommendedAction",event.target.value)} /></label>
  </div>{error?<p className="form-error" role="alert">{error}</p>:null}<footer className="dialog-actions"><button type="button" className="text-button" onClick={onClose}>Cancel</button><span/><button className="primary" type="submit" disabled={busy}>{busy?"Saving…":issue?"Save issue":"Create issue draft"}</button></footer></form></Dialog>;
}

export type ReviewAction = "approve" | "reject" | "request_evidence";

export function ReviewDecisionDialog({ open, issue, action, busy, onClose, onConfirm }: { open: boolean; issue?: Issue; action: ReviewAction; busy: boolean; onClose: () => void; onConfirm: (reason: string, note: string) => Promise<void> }) {
  const [reason,setReason]=useState(""); const [note,setNote]=useState(""); const [error,setError]=useState("");
  useEffect(()=>{if(open){setReason(action==="approve"?"source_and_evidence_sufficient":"");setNote("");setError("")}},[action,open]);
  const labels = action === "approve" ? { title:"Approve issue package", button:"Approve package", copy:"Confirm the source snapshot and field evidence are sufficient for an official action." } : action === "reject" ? { title:"Reject issue draft", button:"Reject draft", copy:"Record a structured reason so the decision is auditable and useful for future drafts." } : { title:"Request evidence", button:"Send evidence request", copy:"Describe exactly what the field team must capture before this package can return to review." };
  const reasons = action === "approve" ? [["source_and_evidence_sufficient","Source and evidence sufficient"],["approved_with_note","Approved with reviewer note"]] : action === "reject" ? [["wrong_location","Wrong location"],["wrong_source","Wrong source"],["insufficient_evidence","Insufficient evidence"],["wrong_route","Wrong route"],["duplicate","Duplicate"],["not_an_issue","Not an issue"]] : [["context_photo","Context photo needed"],["detail_photo","Detail photo needed"],["measurement","Measurement needed"],["location_confirmation","Location confirmation needed"],["current_source","Current source needed"],["field_reverification","Field re-verification needed"]];
  async function submit(event:FormEvent){event.preventDefault();if(!reason)return setError("Choose a decision reason.");if(action!=="approve"&&!note.trim())return setError("Add instructions or reviewer notes.");setError("");try{await onConfirm(reason,note)}catch(cause){setError(cause instanceof Error?cause.message:"Review decision failed.")}}
  return <Dialog open={open} title={labels.title} description={labels.copy} onClose={onClose} size="small"><form className="dialog-form" onSubmit={submit}><div className="decision-context"><span>{issue?.issue_id}</span><b>{issue?.title}</b><small>{issue?.room} · {String(issue?.requirement.source??"Source unresolved")}</small></div><label>Reason<select autoFocus value={reason} onChange={(event)=>setReason(event.target.value)}><option value="">Choose a reason</option>{reasons.map(([value,label])=><option value={value} key={value}>{label}</option>)}</select></label><label>{action==="request_evidence"?"Instructions to field team":"Reviewer note"}<textarea value={note} onChange={(event)=>setNote(event.target.value)} placeholder={action==="request_evidence"?"Capture a wide-angle image and a perpendicular detail with tape reference.":"Optional context for the audit record."}/></label>{error?<p className="form-error" role="alert">{error}</p>:null}<footer className="dialog-actions"><button type="button" className="text-button" onClick={onClose}>Cancel</button><span/><button className={`primary ${action}`} type="submit" disabled={busy}>{busy?"Saving decision…":labels.button}</button></footer></form></Dialog>;
}

export function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}
