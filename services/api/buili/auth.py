from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import delete, event, select
from sqlalchemy.orm import Session, with_loader_criteria

from .config import get_settings
from .database import SessionLocal, get_session
from .models import (
    AuditEvent,
    DirectoryMember,
    Document,
    DocumentRevision,
    EvidenceLink,
    FieldEvidence,
    FieldPoseFrame,
    Frame,
    Issue,
    IssueEvidence,
    IssueWorkflow,
    Job,
    LoginSession,
    Membership,
    Notification,
    Observation,
    Organization,
    PlanEntity,
    PlanGraph,
    Project,
    ProjectProfile,
    ReportRecord,
    ReportScope,
    ReportVersion,
    ReviewRecord,
    Sheet,
    SiteMedia,
    SpatialAlignment,
    SpatialAsset,
    SpatialEvidence,
    SpecChunk,
    UploadIntent,
    User,
    UserCredential,
    new_id,
)
from .schemas import LoginRequest

COOKIE_NAME = "buili_session"
PBKDF2_ITERATIONS = 600_000
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@dataclass(frozen=True)
class Principal:
    user_id: str
    email: str
    name: str
    org_id: str
    org_name: str
    role: str
    session_id: str
    expires_at: datetime


_principal_context: ContextVar[Principal | None] = ContextVar(
    "buili_principal", default=None
)


def current_principal() -> Principal | None:
    return _principal_context.get()


def bind_principal(principal: Principal | None) -> Token[Principal | None]:
    return _principal_context.set(principal)


def reset_principal(token: Token[Principal | None]) -> None:
    _principal_context.reset(token)


DIRECT_PROJECT_MODELS = (
    Document,
    SiteMedia,
    Issue,
    PlanGraph,
    SpatialAsset,
    SpatialAlignment,
    Job,
    UploadIntent,
    ProjectProfile,
    DirectoryMember,
    DocumentRevision,
    FieldEvidence,
    ReviewRecord,
    ReportRecord,
    Notification,
    AuditEvent,
)


@event.listens_for(Session, "do_orm_execute")
def _scope_authenticated_selects(execute_state) -> None:  # type: ignore[no-untyped-def]
    """Apply organization scope to every direct project entity query.

    This is defense-in-depth behind route checks.  It also covers lookups by
    opaque issue/report/job IDs, preventing identifier guessing from crossing an
    organization boundary.
    """

    principal = current_principal()
    if (
        not principal
        or not execute_state.is_select
        or execute_state.execution_options.get("include_all_tenants")
    ):
        return
    project_ids = select(Project.project_id).where(Project.org_id == principal.org_id)
    document_ids = select(Document.doc_id).where(Document.project_id.in_(project_ids))
    sheet_ids = select(Sheet.sheet_id).where(Sheet.doc_id.in_(document_ids))
    media_ids = select(SiteMedia.media_id).where(SiteMedia.project_id.in_(project_ids))
    issue_ids = select(Issue.issue_id).where(Issue.project_id.in_(project_ids))
    report_ids = select(ReportRecord.report_id).where(ReportRecord.project_id.in_(project_ids))
    statement = execute_state.statement.options(
        with_loader_criteria(
            Project,
            Project.org_id == principal.org_id,
            include_aliases=True,
        )
    )
    for model in DIRECT_PROJECT_MODELS:
        statement = statement.options(
            with_loader_criteria(
                model,
                model.project_id.in_(project_ids),
                include_aliases=True,
            )
        )
    subordinate_criteria = (
        (Sheet, Sheet.doc_id.in_(document_ids)),
        (SpecChunk, SpecChunk.doc_id.in_(document_ids)),
        (PlanEntity, PlanEntity.sheet_id.in_(sheet_ids)),
        (Frame, Frame.media_id.in_(media_ids)),
        (Observation, Observation.media_id.in_(media_ids)),
        (FieldPoseFrame, FieldPoseFrame.media_id.in_(media_ids)),
        (IssueEvidence, IssueEvidence.issue_id.in_(issue_ids)),
        (EvidenceLink, EvidenceLink.issue_id.in_(issue_ids)),
        (IssueWorkflow, IssueWorkflow.issue_id.in_(issue_ids)),
        (SpatialEvidence, SpatialEvidence.issue_id.in_(issue_ids)),
        (ReportVersion, ReportVersion.report_id.in_(report_ids)),
        (ReportScope, ReportScope.report_id.in_(report_ids)),
    )
    for model, criterion in subordinate_criteria:
        statement = statement.options(
            with_loader_criteria(model, criterion, include_aliases=True)
        )
    execute_state.statement = statement


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def _decode_b64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_raw, salt_raw, digest_raw = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        if iterations < 100_000 or iterations > 2_000_000:
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), _decode_b64(salt_raw), iterations
        )
        return hmac.compare_digest(candidate, _decode_b64(digest_raw))
    except (ValueError, TypeError):
        return False


# Always perform an expensive comparison for unknown users to reduce account
# enumeration through timing.  This hash is intentionally not a valid credential.
_DUMMY_HASH = hash_password("not-a-real-buili-password")


def _keyed_hash(value: str) -> str:
    secret = get_settings().session_secret.encode()
    return hmac.new(secret, value.encode(), hashlib.sha256).hexdigest()


def _client_fingerprint(request: Request) -> tuple[str, str]:
    user_agent = request.headers.get("user-agent", "")[:1024]
    client_ip = request.client.host if request.client else ""
    return _keyed_hash(user_agent), _keyed_hash(client_ip)


def _parse_cookie(raw: str) -> tuple[str, str] | None:
    if not raw or "." not in raw:
        return None
    session_id, token = raw.split(".", 1)
    if not session_id.startswith("ses_") or len(token) < 32:
        return None
    return session_id, token


def _principal_for_cookie(raw: str, request: Request) -> Principal | None:
    parsed = _parse_cookie(raw)
    if not parsed:
        return None
    session_id, token = parsed
    now = _utcnow()
    with SessionLocal() as session:
        login_session = session.get(LoginSession, session_id)
        if (
            not login_session
            or login_session.revoked_at is not None
            or login_session.expires_at <= now
            or not hmac.compare_digest(login_session.token_hash, _keyed_hash(token))
        ):
            return None
        user = session.get(User, login_session.user_id)
        org = session.get(Organization, login_session.org_id)
        credential = session.scalar(
            select(UserCredential).where(UserCredential.user_id == login_session.user_id)
        )
        if not user or not org or not credential or credential.disabled_at is not None:
            return None
        membership = session.scalar(
            select(Membership)
            .where(
                Membership.user_id == login_session.user_id,
                Membership.org_id == login_session.org_id,
            )
            .limit(1)
        )
        if not membership:
            login_session.revoked_at = now
            session.add(
                AuditEvent(
                    org_id=login_session.org_id,
                    actor=user.email,
                    action="SESSION_ACCESS_REVOKED",
                    entity_type="login_session",
                    entity_id=login_session.session_id,
                    metadata_json={"reason": "membership_removed"},
                )
            )
            session.commit()
            return None
        if login_session.role != membership.role:
            previous_role = login_session.role
            login_session.role = membership.role
            session.add(
                AuditEvent(
                    org_id=login_session.org_id,
                    actor=user.email,
                    action="SESSION_ROLE_REFRESHED",
                    entity_type="login_session",
                    entity_id=login_session.session_id,
                    before_json={"role": previous_role},
                    after_json={"role": membership.role},
                )
            )
            session.commit()
        if (now - login_session.last_seen_at).total_seconds() >= 300:
            login_session.last_seen_at = now
            session.commit()
        return Principal(
            user_id=user.user_id,
            email=user.email,
            name=user.name,
            org_id=org.org_id,
            org_name=org.name,
            role=membership.role,
            session_id=login_session.session_id,
            expires_at=login_session.expires_at,
        )


def request_path_requires_auth(path: str) -> bool:
    normalized = path[4:] if path.startswith("/api/") else path
    if not normalized.startswith("/v1/"):
        return False
    return normalized not in {"/v1/auth/login"}


def resolve_request_principal(request: Request) -> Principal | None:
    return _principal_for_cookie(request.cookies.get(COOKIE_NAME, ""), request)


def validate_production_auth_settings() -> None:
    settings = get_settings()
    if not settings.auth_required:
        return
    if len(settings.auth_secret) < 32:
        raise RuntimeError("BUILI_AUTH_SECRET must contain at least 32 characters")
    if settings.pilot_seed_enabled and not settings.pilot_password:
        raise RuntimeError("BUILI_PILOT_PASSWORD is required when pilot seeding is enabled")
    if settings.pilot_password and len(settings.pilot_password) < 12:
        raise RuntimeError("BUILI_PILOT_PASSWORD must contain at least 12 characters")


def _identity_payload(
    principal: Principal,
    session: Session,
) -> dict[str, Any]:
    projects = list(
        session.scalars(
            select(Project)
            .where(Project.org_id == principal.org_id)
            .order_by(Project.created_at.asc())
        ).all()
    )
    return {
        "user": {
            "user_id": principal.user_id,
            "email": principal.email,
            "name": principal.name,
            "role": principal.role,
            "organization": {
                "org_id": principal.org_id,
                "name": principal.org_name,
            },
        },
        "projects": [
            {
                "project_id": project.project_id,
                "org_id": project.org_id,
                "name": project.name,
                "address": project.address,
                "project_type": project.project_type,
                "status": project.status,
            }
            for project in projects
        ],
        "expires_at": principal.expires_at.isoformat() + "Z",
    }


def ensure_pilot_identity(session: Session, project: Project) -> User | None:
    settings = get_settings()
    if not settings.pilot_seed_enabled:
        return None
    email = settings.pilot_email.strip().lower()
    user = session.scalar(select(User).where(User.email == email))
    if not user:
        user = User(email=email, name=settings.pilot_name)
        session.add(user)
        session.flush()
    else:
        user.name = settings.pilot_name
    membership = session.scalar(
        select(Membership).where(
            Membership.user_id == user.user_id,
            Membership.org_id == project.org_id,
        )
    )
    if not membership:
        membership = Membership(
            user_id=user.user_id,
            org_id=project.org_id,
            role="project_manager",
        )
        session.add(membership)
    else:
        membership.role = "project_manager"
    credential = session.scalar(
        select(UserCredential).where(UserCredential.user_id == user.user_id)
    )
    password = settings.pilot_password or (
        "BuiliPilot!2026" if not settings.auth_required else ""
    )
    if not credential:
        if not password:
            raise RuntimeError("pilot password is required to create the pilot credential")
        credential = UserCredential(
            user_id=user.user_id,
            password_hash=hash_password(password),
        )
        session.add(credential)
    elif password and not verify_password(password, credential.password_hash):
        # The Render-managed pilot secret is also the rotation source.  Updating
        # it is safe and idempotent because only a one-way hash is persisted.
        credential.password_hash = hash_password(password)
        credential.password_changed_at = _utcnow()
        credential.failed_attempts = 0
        credential.locked_until = None
        now = _utcnow()
        active_sessions = list(
            session.scalars(
                select(LoginSession).where(
                    LoginSession.user_id == user.user_id,
                    LoginSession.revoked_at.is_(None),
                )
            ).all()
        )
        for active_session in active_sessions:
            active_session.revoked_at = now
        session.add(
            AuditEvent(
                org_id=project.org_id,
                actor="system",
                action="PILOT_PASSWORD_ROTATED",
                entity_type="user",
                entity_id=user.user_id,
                metadata_json={"sessions_revoked": len(active_sessions)},
            )
        )
    session.commit()
    return user


router = APIRouter(prefix="/v1/auth", tags=["authentication"])


@router.post("/login")
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    email = payload.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=401, detail="invalid email or password")
    now = _utcnow()
    user = session.scalar(select(User).where(User.email == email))
    credential = (
        session.scalar(select(UserCredential).where(UserCredential.user_id == user.user_id))
        if user
        else None
    )
    password_matches = verify_password(
        payload.password,
        credential.password_hash if credential else _DUMMY_HASH,
    )
    if (
        not user
        or not credential
        or credential.disabled_at is not None
        or (credential.locked_until is not None and credential.locked_until > now)
        or not password_matches
    ):
        if credential and credential.disabled_at is None:
            credential.failed_attempts += 1
            if credential.failed_attempts >= MAX_FAILED_ATTEMPTS:
                credential.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
                credential.failed_attempts = 0
            session.commit()
        raise HTTPException(status_code=401, detail="invalid email or password")

    membership = session.scalar(
        select(Membership)
        .where(Membership.user_id == user.user_id)
        .order_by(Membership.created_at.asc())
    )
    if not membership:
        raise HTTPException(status_code=403, detail="account has no active organization access")
    org = session.get(Organization, membership.org_id)
    if not org:
        raise HTTPException(status_code=403, detail="account has no active organization access")

    credential.failed_attempts = 0
    credential.locked_until = None
    session.execute(
        delete(LoginSession).where(
            LoginSession.user_id == user.user_id,
            LoginSession.expires_at <= now,
        )
    )
    duration = (
        timedelta(days=get_settings().remember_session_days)
        if payload.remember_me
        else timedelta(hours=get_settings().session_hours)
    )
    expires_at = now + duration
    session_id = new_id("ses")
    bearer = secrets.token_urlsafe(32)
    user_agent_hash, ip_hash = _client_fingerprint(request)
    record = LoginSession(
        session_id=session_id,
        user_id=user.user_id,
        org_id=org.org_id,
        role=membership.role,
        token_hash=_keyed_hash(bearer),
        expires_at=expires_at,
        last_seen_at=now,
        user_agent_hash=user_agent_hash,
        ip_hash=ip_hash,
    )
    session.add(record)
    session.add(
        AuditEvent(
            org_id=org.org_id,
            actor=user.email,
            action="USER_LOGGED_IN",
            entity_type="login_session",
            entity_id=session_id,
            metadata_json={"remember_me": payload.remember_me},
        )
    )
    session.commit()
    settings = get_settings()
    response.set_cookie(
        key=COOKIE_NAME,
        value=f"{session_id}.{bearer}",
        max_age=max(1, int(duration.total_seconds())),
        expires=datetime.now(UTC) + duration,
        path="/",
        secure=settings.secure_cookies,
        httponly=True,
        samesite="lax",
    )
    principal = Principal(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        org_id=org.org_id,
        org_name=org.name,
        role=membership.role,
        session_id=session_id,
        expires_at=expires_at,
    )
    token = bind_principal(principal)
    try:
        return _identity_payload(principal, session)
    finally:
        reset_principal(token)


@router.get("/me")
def me(request: Request, session: Session = Depends(get_session)) -> dict[str, Any]:
    principal: Principal | None = getattr(request.state, "principal", None)
    if not principal:
        raise HTTPException(status_code=401, detail="authentication required")
    return _identity_payload(principal, session)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
) -> None:
    parsed = _parse_cookie(request.cookies.get(COOKIE_NAME, ""))
    if parsed:
        record = session.get(LoginSession, parsed[0])
        if record and record.revoked_at is None:
            record.revoked_at = _utcnow()
            session.add(
                AuditEvent(
                    org_id=record.org_id,
                    actor=getattr(getattr(request.state, "principal", None), "email", "system"),
                    action="USER_LOGGED_OUT",
                    entity_type="login_session",
                    entity_id=record.session_id,
                )
            )
            session.commit()
    response.delete_cookie(
        COOKIE_NAME,
        path="/",
        secure=get_settings().secure_cookies,
        httponly=True,
        samesite="lax",
    )
