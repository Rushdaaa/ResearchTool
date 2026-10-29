import os
import sys
import json
import uuid
import queue
import threading
import csv
import shutil
from pathlib import Path
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

# ------------------------------
# Paths + imports
# ------------------------------
ROOT = Path(__file__).resolve().parent               # ResearchTool/
CODE_DIR = ROOT / "code"
DATA_DIR = ROOT / "data"
PROJECTS_DIR = DATA_DIR / "projects"

DATA_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

# Allow imports from ./code without making it a package
sys.path.insert(0, str(CODE_DIR))

# Import existing pipeline helpers
from main_stream import split_into_clauses, classify_text_clause
from config import TAU_HCR, TAU_TAG

# AssemblyAI streaming
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)

# ------------------------------
# Env
# ------------------------------
load_dotenv()
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not AAI_KEY:
    raise RuntimeError("Missing ASSEMBLYAI_API_KEY in .env")
aai.settings.api_key = AAI_KEY


# ------------------------------
# Storage helpers
# ------------------------------
def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _project_folder(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def _sessions_folder(project_id: str) -> Path:
    return _project_folder(project_id) / "sessions"


def _exports_folder(project_id: str) -> Path:
    out = _project_folder(project_id) / "exports"
    out.mkdir(exist_ok=True, parents=True)
    return out


def _final_reqs_path(project_id: str) -> Path:
    return _project_folder(project_id) / "final_requirements.json"


def create_project(project_name: str, domain: str) -> dict:
    name = (project_name or "").strip()
    if not name:
        raise ValueError("Project name is required.")

    project_id = str(uuid.uuid4())[:8]
    pf = _project_folder(project_id)
    sf = _sessions_folder(project_id)
    pf.mkdir(exist_ok=True)
    sf.mkdir(exist_ok=True)

    meta = {
        "project_id": project_id,
        "project_name": name,
        "domain": domain,
        "created_at": _now_iso(),
    }
    (pf / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

def list_projects() -> list[dict]:
    projects = []
    for p in PROJECTS_DIR.iterdir():
        if not p.is_dir():
            continue
        f = p / "project.json"
        if f.exists():
            try:
                projects.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    projects.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return projects

def delete_project(project_id: str) -> bool:
    pf = _project_folder(project_id)
    if not pf.exists() or not pf.is_dir():
        return False
    shutil.rmtree(pf)
    return True

def rename_project(project_id: str, new_name: str) -> bool:
    new_name = (new_name or "").strip()
    if not new_name:
        return False

    f = _project_folder(project_id) / "project.json"
    if not f.exists():
        return False

    try:
        meta = json.loads(f.read_text(encoding="utf-8"))
        meta["project_name"] = new_name
        meta["updated_at"] = _now_iso()
        f.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False

def list_sessions(project_id: str) -> list[dict]:
    sf = _sessions_folder(project_id)
    if not sf.exists():
        return []
    sessions = []
    for f in sf.glob("session_*.json"):
        try:
            sessions.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    sessions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return sessions

def save_session(project_id: str, session_payload: dict) -> Path:
    sf = _sessions_folder(project_id)
    sf.mkdir(exist_ok=True, parents=True)
    sid = session_payload["session_id"]
    out = sf / f"session_{sid}.json"
    out.write_text(json.dumps(session_payload, indent=2), encoding="utf-8")
    return out


def delete_session_file(project_id: str, session_id: str) -> bool:
    f = _sessions_folder(project_id) / f"session_{session_id}.json"
    if f.exists():
        f.unlink()
        return True
    return False

def rename_session(project_id: str, session_id: str, new_title: str) -> bool:
    new_title = (new_title or "").strip()
    if not new_title:
        return False

    f = _sessions_folder(project_id) / f"session_{session_id}.json"
    if not f.exists():
        return False

    try:
        payload = json.loads(f.read_text(encoding="utf-8"))
        payload["session_title"] = new_title
        payload["updated_at"] = _now_iso()
        f.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False

def load_final_requirements(project_id: str) -> list[dict]:
    p = _final_reqs_path(project_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("final_requirements", []) if isinstance(data, dict) else []
    except Exception:
        return []


def save_final_requirements(project_id: str, items: list[dict]) -> Path:
    payload = {
        "project_id": project_id,
        "updated_at": _now_iso(),
        "final_requirements": items,
    }
    p = _final_reqs_path(project_id)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


# ------------------------------
# Rendering (chips + cards)
# ------------------------------
def chips_html(items: list[str]) -> str:
    if not items:
        return "<span style='opacity:.6'>None</span>"
    return " ".join(
        f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
        f"border:1px solid #999;margin:2px;font-size:12px'>{x}</span>"
        for x in items
    )

def _pct(x: float, digits: int = 0) -> str:
    """Format a probability in [0,1] as a percentage string."""
    try:
        v = float(x)
    except Exception:
        v = 0.0
    v = max(0.0, min(1.0, v))
    return f"{v * 100:.{digits}f}%"

def requirement_cards_html(clauses: list[dict]) -> str:
    blocks = []

    for i, c in enumerate(clauses, 1):
        fr = c["frnfr"]
        h = c["hcr"]

        p_hcr = float(h.get("p_hcr", 0.0))
        tags = h.get("tags", [])

        # Decide how to display HCR tags based on threshold
        if p_hcr < TAU_HCR:
            tags_html = "<span style='opacity:.6'>— (below HCR threshold)</span>"
        else:
            tags_html = chips_html(tags)

        blocks.append(
            f"""
            <div style="border:1px solid #444;border-radius:12px;padding:10px;margin:10px 0;">
              <div style="font-weight:600;margin-bottom:6px;">Requirement {i}</div>

              <div style="margin-bottom:8px;">
                {c["text"]}
              </div>

              <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <div>
                  <b>FR/NFR:</b> {fr["label"]}
                  <span style="opacity:.7">({_pct(fr.get("confidence", 0.0), 0)})</span>
                </div>

                <div>
                  <b>HCR:</b>
                  <span style="opacity:.7">{_pct(p_hcr, 0)}</span>
                </div>
              </div>

              <div style="margin-top:8px;">
                <b>HCR tags:</b> {tags_html}
              </div>
            </div>
            """
        )

    return "\n".join(blocks) if blocks else "<span style='opacity:.6'>No requirements yet.</span>"

# ------------------------------
# Streaming worker (thread)
# ------------------------------
class StreamWorker(threading.Thread):
    """
    Runs AssemblyAI streaming in a background thread and pushes updates to a queue.
    """

    def __init__(self, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q = out_queue
        self.stop_event = stop_event
        self.client = None

    def run(self):
        try:
            self._run_stream()
        except Exception as e:
            self.q.put({"type": "error", "message": str(e)})

    def _run_stream(self):
        self.client = StreamingClient(
            StreamingClientOptions(api_key=AAI_KEY, api_host="streaming.assemblyai.com")
        )

        def on_begin(_self, event: BeginEvent):
            self.q.put({"type": "status", "message": "Listening…"})

        def on_turn(_self, event: TurnEvent):
            raw = (event.transcript or "").strip()
            if not raw:
                return

            # partial
            if not event.end_of_turn:
                self.q.put({"type": "partial", "text": raw})
                return

            # ensure formatted final
            if not event.turn_is_formatted:
                params = StreamingSessionParameters(format_turns=True)
                _self.set_params(params)
                return

            final_text = raw.strip()
            if not final_text:
                return

            clauses = split_into_clauses(final_text)
            clause_results = [classify_text_clause(c) for c in clauses]

            self.q.put(
                {
                    "type": "final",
                    "final_text": final_text,
                    "clauses": clause_results,
                }
            )

        def on_terminated(_self, event: TerminationEvent):
            self.q.put({"type": "status", "message": "Stopped."})

        def on_error(_self, error: StreamingError):
            self.q.put({"type": "error", "message": f"{error}"})

        self.client.on(StreamingEvents.Begin, on_begin)
        self.client.on(StreamingEvents.Turn, on_turn)
        self.client.on(StreamingEvents.Termination, on_terminated)
        self.client.on(StreamingEvents.Error, on_error)

        self.client.connect(StreamingParameters(sample_rate=16000, format_turns=True))

        try:
            mic = aai.extras.MicrophoneStream(sample_rate=16000)
            # Stream until stop requested
            while not self.stop_event.is_set():
                self.client.stream(mic)
                break
        finally:
            try:
                self.client.disconnect(terminate=True)
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()
        try:
            if self.client:
                self.client.disconnect(terminate=True)
        except Exception:
            pass


# ------------------------------
# UI helpers (dropdown parsing + refresh)
# ------------------------------
def ui_refresh_projects():
    projects = list_projects()

    # choices as (label, value) so IDs NEVER show in the UI
    choices = [
        (f'{p["project_name"]} ({p["domain"]})', p["project_id"])
        for p in projects
    ]

    first_value = choices[0][1] if choices else None
    return gr.Dropdown(choices=choices, value=first_value)


def parse_project_id(project_dropdown_value: str | None) -> str | None:
    # value is already the project_id now
    return project_dropdown_value


def ui_list_sessions(project_dropdown_value, keep_value: str | None = None):
    pid = parse_project_id(project_dropdown_value)
    if not pid:
        return gr.Dropdown(choices=[], value=None)

    sessions = list_sessions(pid)

    # Optional: name Session 1..N in chronological order (oldest -> newest)
    sessions_sorted = sorted(sessions, key=lambda s: s.get("started_at", ""))

    choices = []
    for idx, s in enumerate(sessions_sorted, 1):
        title = s.get("session_title") or f"Session {idx}"
        choices.append((title, s["session_id"]))

    # keep selection if requested AND still exists
    valid_values = {v for _, v in choices}
    value = keep_value if keep_value in valid_values else None

    return gr.Dropdown(choices=choices, value=value)

def parse_session_id(session_dropdown_value: str | None) -> str | None:
    # value is already the session_id now
    return session_dropdown_value

def ui_create_project(name, domain):
    name = (name or "").strip()
    if not name:
        # leave dropdown unchanged
        return gr.update(), "⚠️ Please enter a project name before creating."

    try:
        meta = create_project(name, domain)
    except Exception as e:
        return gr.update(), f"ERROR: {e}"

    dd = ui_refresh_projects()
    msg = f"Created project: {meta['project_name']} ({meta['domain']})"
    return dd, msg

def ui_delete_project(project_dropdown_value):
    pid = parse_project_id(project_dropdown_value)
    if not pid:
        return gr.update(), gr.update(choices=[], value=None), "⚠️ Select a project to delete.", "", "<span style='opacity:.6'>No requirements yet.</span>", {}, gr.update(visible=False)

    ok = delete_project(pid)

    # refresh projects + clear sessions + clear right panel
    projects_dd = ui_refresh_projects()
    sessions_dd = gr.Dropdown(choices=[], value=None)

    msg = "🗑️ Project deleted." if ok else "Project not found."
    return projects_dd, sessions_dd, msg, "", "<span style='opacity:.6'>No requirements yet.</span>", {}, gr.update(visible=False)

def ui_rename_project(project_dropdown_value, new_name):
    pid = parse_project_id(project_dropdown_value)
    if not pid:
        return gr.update(), "Select a project first.", gr.update(value="")

    ok = rename_project(pid, new_name)
    if not ok:
        return gr.update(), "Enter a valid project name.", gr.update(value="")

    # refresh dropdown, keep same project selected
    projects = list_projects()
    choices = [(f'{p["project_name"]} ({p["domain"]})', p["project_id"]) for p in projects]
    dd = gr.Dropdown(choices=choices, value=pid)

    return dd, "Project renamed.", gr.update(value="")

# ------------------------------
# Session load/delete
# ------------------------------
def ui_load_session(project_dropdown_value, session_dropdown_value):
    pid = parse_project_id(project_dropdown_value)
    sid = parse_session_id(session_dropdown_value)

    # If dropdown is cleared (None), DON'T overwrite the "Ready..." message from New Session
    if not pid or not sid:
        return gr.update(), gr.update(), gr.update(), {}

    sf = _sessions_folder(pid) / f"session_{sid}.json"
    if not sf.exists():
        return "", "", "<span style='opacity:.6'>Session file missing.</span>", {}

    payload = json.loads(sf.read_text(encoding="utf-8"))

    # Do NOT load transcript text
    cards = requirement_cards_html(payload.get("all_clauses", []))
    return "", "", cards, payload


def ui_render_cards_from_raw(raw_payload: dict):
    if not raw_payload:
        return "<span style='opacity:.6'>No requirements yet.</span>"
    return requirement_cards_html(raw_payload.get("all_clauses", []))


def ui_new_session(project_dropdown_value):
    pid = parse_project_id(project_dropdown_value)
    if not pid:
        return (
            {}, 
            "Select a project first.", 
            "", 
            "<span style='opacity:.6'>No requirements yet.</span>", 
            {}, 
            gr.update(value=None),          # clear selection (blank dropdown)
            gr.update(interactive=False),   # Start disabled
            gr.update(visible=False),       # transcript panel hidden
        )

    next_num = len(list_sessions(pid)) + 1
    session_title = f"Session {next_num}"

    session_state = {
        "session_id": str(uuid.uuid4())[:8],
        "session_title": session_title,
        "project_id": pid,
        "started_at": _now_iso(),
        "final_turns": [],
        "all_clauses": [],
        "_running": False,
        "_should_save": False,
    }

    return (
        session_state,
        f"Ready: **{session_title}** — click Start to record",
        "",
        "<span style='opacity:.6'>No requirements yet.</span>",
        {},
        gr.update(value=None),          
        gr.update(interactive=True),       
        gr.update(visible=True),           
    )


def ui_delete_session(project_dropdown_value, session_dropdown_value):
    pid = parse_project_id(project_dropdown_value)
    sid = parse_session_id(session_dropdown_value)
    if not pid or not sid:
        return gr.Dropdown(choices=[], value=None), "", "", "<span style='opacity:.6'>Select a session to delete.</span>", {}

    ok = delete_session_file(pid, sid)
    sessions_dd = ui_list_sessions(project_dropdown_value)

    msg = "Deleted session." if ok else "Session not found."
    return sessions_dd, msg, "", "<span style='opacity:.6'>No requirements yet.</span>", {}

def ui_rename_session_step1(project_dropdown_value, session_dropdown_value, new_title):
    pid = parse_project_id(project_dropdown_value)
    sid = parse_session_id(session_dropdown_value)

    if not pid or not sid:
        return gr.update(), "Select a session first.", gr.update(value=""), None

    ok = rename_session(pid, sid, new_title)
    if not ok:
        return gr.update(), "Enter a valid session name.", gr.update(value=""), None

    # Step 1: refresh choices but CLEAR selection to force UI repaint
    dd_clear = ui_list_sessions(project_dropdown_value, keep_value=None)
    return dd_clear, "Session renamed.", gr.update(value=""), sid


def ui_rename_session_step2(project_dropdown_value, sid_to_reselect):
    # Step 2: reselect the same sid (label updates in collapsed view)
    if not sid_to_reselect:
        return gr.update(), None
    dd = ui_list_sessions(project_dropdown_value, keep_value=sid_to_reselect)
    return dd, None

# ------------------------------
# Live session controls (Start / Pause / Resume / Stop+Save)
# ------------------------------
def _start_worker(session_state: dict) -> dict:
    q = queue.Queue()
    stop_event = threading.Event()
    worker = StreamWorker(q, stop_event)
    session_state["_queue"] = q
    session_state["_stop_event"] = stop_event
    session_state["_worker"] = worker
    session_state["_running"] = True
    worker.start()
    return session_state


def _stop_worker(session_state: dict):
    if session_state and session_state.get("_stop_event"):
        session_state["_stop_event"].set()
    if session_state and session_state.get("_worker"):
        try:
            session_state["_worker"].stop()
        except Exception:
            pass
    session_state["_running"] = False

def _serializable_session_copy(session_state: dict) -> dict:
    """Make a JSON-safe copy (no thread/queue/event objects)."""
    if not session_state:
        return {}
    out = dict(session_state)
    out.pop("_worker", None)
    out.pop("_stop_event", None)
    out.pop("_queue", None)
    out.pop("_running", None)
    out.pop("_should_save", None)
    return out

def ui_start_recording(project_dropdown_value, session_state):
    """
    Starts recording into the *existing* prepared session_state created by ui_new_session.
    Generator streams updates.
    """
    pid = parse_project_id(project_dropdown_value)

    if not pid:
        yield "Select a project first.", "", "<span style='opacity:.6'>No requirements yet.</span>", session_state, ui_list_sessions(project_dropdown_value)
        return

    if not session_state or session_state.get("project_id") != pid or not session_state.get("session_id"):
        yield "Click **New session** first, then Start.", "", "<span style='opacity:.6'>No requirements yet.</span>", session_state, ui_list_sessions(project_dropdown_value)
        return

    # Prevent double-start
    if session_state.get("_running"):
        yield "Already recording…", "\n".join(session_state.get("final_turns", [])), requirement_cards_html(session_state.get("all_clauses", [])), session_state, ui_list_sessions(project_dropdown_value)
        return

    # Start worker (recording)
    session_state["_should_save"] = False
    session_state = _start_worker(session_state)

    partial_md = "Listening…"
    transcript = "\n".join(session_state.get("final_turns", []))
    cards = requirement_cards_html(session_state.get("all_clauses", []))
    sessions_dd = ui_list_sessions(project_dropdown_value)

    yield partial_md, transcript, cards, session_state, sessions_dd

    while True:
        if session_state.get("_stop_event") and session_state["_stop_event"].is_set():
            break

        try:
            evt = session_state["_queue"].get(timeout=0.2)
        except queue.Empty:
            continue

        if evt["type"] == "partial":
            partial_md = f"{evt['text']}..."
        elif evt["type"] == "final":
            # After a final turn is committed, return to idle status
            partial_md = "Listening…"
            final_text = evt["final_text"]
            session_state["final_turns"].append(final_text)
            session_state["all_clauses"].extend(evt["clauses"])
            transcript = "\n".join(session_state["final_turns"])
            cards = requirement_cards_html(session_state["all_clauses"])
        elif evt["type"] == "status":
            partial_md = evt["message"]
        elif evt["type"] == "error":
            partial_md = f"ERROR: {evt['message']}"
            break

        yield partial_md, transcript, cards, session_state, sessions_dd

    # Stopped (pause or stop/save handles actual saving)
    _stop_worker(session_state)
    session_state["ended_at"] = _now_iso()

    # Clean worker objects (keep session content)
    session_state.pop("_worker", None)
    session_state.pop("_stop_event", None)
    session_state.pop("_queue", None)
    session_state.pop("_running", None)

    # If pause requested, show paused message instead of "Stopped."
    if session_state.get("_pause_requested"):
        session_state["_pause_requested"] = False
        yield "⏸ Paused (not saved).", "\n".join(session_state["final_turns"]), requirement_cards_html(session_state["all_clauses"]), session_state, sessions_dd
    else:
        yield "Stopped.", "\n".join(session_state["final_turns"]), requirement_cards_html(session_state["all_clauses"]), session_state, sessions_dd


def ui_pause(session_state):
    """
    Pause: stop streaming but do NOT save.
    """
    if not session_state:
        return session_state, "Nothing to pause."
    session_state["_should_save"] = False
    session_state["_pause_requested"] = True   # tell generator this is a pause
    _stop_worker(session_state)
    return session_state, "⏸ Pausing…"

def ui_resume(project_dropdown_value, session_state):
    """
    Resume: continue streaming into the SAME in-memory session state.
    Generator streams updates.
    """
    pid = parse_project_id(project_dropdown_value)
    if not pid or not session_state or session_state.get("project_id") != pid:
        yield "Select a project and start a session first.", "", "<span style='opacity:.6'>No requirements yet.</span>", session_state, ui_list_sessions(project_dropdown_value)
        return

    # Prevent double-start
    if session_state.get("_running"):
        yield "Already recording…", "\n".join(session_state.get("final_turns", [])), requirement_cards_html(session_state.get("all_clauses", [])), session_state, ui_list_sessions(project_dropdown_value)
        return

    # Reattach worker objects and stream again
    session_state["_should_save"] = False
    session_state = _start_worker(session_state)

    partial_md = "Listening…"
    transcript = "\n".join(session_state.get("final_turns", []))
    cards = requirement_cards_html(session_state.get("all_clauses", []))
    sessions_dd = ui_list_sessions(project_dropdown_value)

    yield partial_md, transcript, cards, session_state, sessions_dd

    while True:
        if session_state.get("_stop_event") and session_state["_stop_event"].is_set():
            break

        try:
            evt = session_state["_queue"].get(timeout=0.2)
        except queue.Empty:
            continue

        if evt["type"] == "partial":
            partial_md = f"{evt['text']}..."
        elif evt["type"] == "final":
            # After a final turn is committed, return to idle status
            partial_md = "Listening…"
            final_text = evt["final_text"]
            session_state["final_turns"].append(final_text)
            session_state["all_clauses"].extend(evt["clauses"])
            transcript = "\n".join(session_state["final_turns"])
            cards = requirement_cards_html(session_state["all_clauses"])
        elif evt["type"] == "status":
            partial_md = evt["message"]
        elif evt["type"] == "error":
            partial_md = f"ERROR: {evt['message']}"
            break

        yield partial_md, transcript, cards, session_state, sessions_dd

    _stop_worker(session_state)
    session_state["ended_at"] = _now_iso()

    # Clean non-serializables
    session_state.pop("_worker", None)
    session_state.pop("_stop_event", None)
    session_state.pop("_queue", None)
    session_state.pop("_running", None)

    # If pause requested, show paused message instead of "Stopped."
    if session_state.get("_pause_requested"):
        session_state["_pause_requested"] = False
        yield "⏸ Paused (not saved).", "\n".join(session_state["final_turns"]), requirement_cards_html(session_state["all_clauses"]), session_state, sessions_dd
    else:
        yield "Stopped.", "\n".join(session_state["final_turns"]), requirement_cards_html(session_state["all_clauses"]), session_state, sessions_dd


def ui_stop_and_save(project_dropdown_value, session_state):
    """
    Stop + Save: save immediately (snapshot), then stop streaming.
    After saving, refresh sessions dropdown AND auto-select the saved session.
    """
    pid = parse_project_id(project_dropdown_value)
    if not pid or not session_state:
        return session_state, "Nothing to stop/save.", ui_list_sessions(project_dropdown_value)

    session_state["_should_save"] = True
    session_state["ended_at"] = _now_iso()
    _stop_worker(session_state)

    payload = _serializable_session_copy(session_state)
    save_session(pid, payload)

    # Refresh dropdown and auto-select the saved session id
    sessions_dd = ui_list_sessions(project_dropdown_value, keep_value=payload.get("session_id"))

    msg = f"Saved {payload.get('session_title','Session')}"
    return session_state, msg, sessions_dd

# ------------------------------
# Project-wide final requirements (curation)
# ------------------------------
def aggregate_requirements_from_sessions(project_id: str) -> list[dict]:
    sessions = list_sessions(project_id)
    out = []
    for s in sessions:
        title = s.get("session_title") or s.get("session_id")
        for clause in s.get("all_clauses", []):
            fr = clause.get("frnfr", {})
            h = clause.get("hcr", {})
            out.append(
                {
                    "text": clause.get("text", ""),
                    "frnfr": fr.get("label", ""),
                    "fr_conf": float(fr.get("confidence", 0.0)),
                    "p_hcr": float(h.get("p_hcr", 0.0)),
                    "tags": ",".join(h.get("tags", [])),
                    "source_session": title,
                }
            )
    return out


def ui_load_all_requirements(project_dropdown_value):
    pid = parse_project_id(project_dropdown_value)
    if not pid:
        return [], "<span style='opacity:.6'>Select a project first.</span>"

    items = aggregate_requirements_from_sessions(pid)

    # Convert list[dict] -> list[list] in the exact header order
    rows = [
        [
            it.get("text", ""),
            it.get("frnfr", ""),
            float(it.get("fr_conf", 0.0) or 0.0),
            float(it.get("p_hcr", 0.0) or 0.0),
            it.get("tags", ""),
            it.get("source_session", ""),
        ]
        for it in (items or [])
    ]

    return rows, f"<span style='opacity:.7'>Loaded {len(rows)} requirements from sessions.</span>"


def ui_load_saved_final(project_dropdown_value):
    pid = parse_project_id(project_dropdown_value)
    if not pid:
        return [], "<span style='opacity:.6'>Select a project first.</span>"

    items = load_final_requirements(pid)

    rows = [
        [
            it.get("text", ""),
            it.get("frnfr", ""),
            float(it.get("fr_conf", 0.0) or 0.0),
            float(it.get("p_hcr", 0.0) or 0.0),
            it.get("tags", ""),
            it.get("source_session", ""),
        ]
        for it in (items or [])
    ]

    return rows, f"<span style='opacity:.7'>Loaded {len(rows)} saved final requirements.</span>"


def ui_save_final(project_dropdown_value, table_rows):
    pid = parse_project_id(project_dropdown_value)
    if not pid:
        return "<span style='opacity:.6'>Select a project first.</span>"

    cleaned = []

    # table_rows will be list[list] (array mode)
    for r in (table_rows or []):
        if not r or len(r) < 6:
            continue

        text = (r[0] or "").strip()
        if not text:
            continue

        frnfr = (r[1] or "").strip()
        try:
            fr_conf = float(r[2] or 0.0)
        except Exception:
            fr_conf = 0.0

        try:
            p_hcr = float(r[3] or 0.0)
        except Exception:
            p_hcr = 0.0

        tags = (r[4] or "").strip()
        source_session = (r[5] or "").strip()

        cleaned.append(
            {
                "text": text,
                "frnfr": frnfr,
                "fr_conf": fr_conf,
                "p_hcr": p_hcr,
                "tags": tags,
                "source_session": source_session,
            }
        )

    save_final_requirements(pid, cleaned)
    return f"<span style='opacity:.7'>Saved {len(cleaned)} final requirements.</span>"

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def ui_pick_row(table_rows, evt: gr.SelectData):
    """
    When user clicks a row, populate the editor fields.
    evt.index is typically (row, col).
    """
    if not table_rows or not hasattr(evt, "index") or evt.index is None:
        return "", "", 0.0, 0.0, "", -1

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    if row_idx < 0 or row_idx >= len(table_rows):
        return "", "", 0.0, 0.0, "", -1

    r = table_rows[row_idx]
    # r = [Requirement, FR/NFR, Confidence, Human-Centric %, Tags, Session]
    return (
        (r[0] or ""),
        (r[1] or ""),
        _safe_float(r[2], 0.0),
        _safe_float(r[3], 0.0),
        (r[4] or ""),
        row_idx,
    )


def ui_apply_edit(table_rows, row_idx, req_text, frnfr, fr_conf, p_hcr, tags):
    if table_rows is None:
        table_rows = []
    if row_idx is None or row_idx < 0 or row_idx >= len(table_rows):
        return table_rows, "<span style='opacity:.7'>Select a row first.</span>"

    # Keep Session column unchanged (index 5)
    session_val = table_rows[row_idx][5] if len(table_rows[row_idx]) >= 6 else ""

    table_rows[row_idx] = [
        (req_text or "").strip(),
        (frnfr or "").strip(),
        _safe_float(fr_conf, 0.0),
        _safe_float(p_hcr, 0.0),
        (tags or "").strip(),
        session_val,
    ]
    return table_rows, "<span style='opacity:.7'>Row updated (remember to Save final).</span>"


def ui_delete_row(table_rows, row_idx):
    if not table_rows:
        return [], "<span style='opacity:.7'>Nothing to delete.</span>", -1

    if row_idx is None or row_idx < 0 or row_idx >= len(table_rows):
        return table_rows, "<span style='opacity:.7'>Select a row first.</span>", -1

    table_rows.pop(row_idx)
    return table_rows, "<span style='opacity:.7'>🗑️ Row removed (remember to Save final).</span>", -1


def ui_add_row(table_rows):
    if table_rows is None:
        table_rows = []

    table_rows.append(["", "", 0.0, 0.0, "", "Manual"])
    return table_rows, "<span style='opacity:.7'>➕ Added a blank row (fill it in, then Save final).</span>"

def _sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    keep = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)
    return keep[:80] or "project"


def ui_export_saved_final_csv(project_dropdown_value):
    pid = parse_project_id(project_dropdown_value)
    if not pid:
        return None, "<span style='opacity:.6'>Select a project first.</span>"

    items = load_final_requirements(pid)
    if not items:
        return None, "<span style='opacity:.6'>No saved final requirements to export. Click Save final first.</span>"

    meta_path = _project_folder(pid) / "project.json"
    project_name = pid
    if meta_path.exists():
        try:
            project_name = json.loads(meta_path.read_text(encoding="utf-8")).get("project_name", pid)
        except Exception:
            pass

    exports_dir = _exports_folder(pid)
    filename = f"final_requirements_{_sanitize_filename(project_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = exports_dir / filename

    headers = ["Requirement", "FR/NFR", "Confidence", "Human-Centric %", "Human-Centric Tags", "Session"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for it in items:
            writer.writerow([
                it.get("text", ""),
                it.get("frnfr", ""),
                it.get("fr_conf", 0.0),
                it.get("p_hcr", 0.0),
                it.get("tags", ""),
                it.get("source_session", ""),
            ])

    return str(out_path), f"<span style='opacity:.7'>Exported CSV: {filename}</span>"


# ------------------------------
# Gradio UI
# ------------------------------

HCR_TAG_CHOICES = ["AC", "US", "FA", "EX", "TR", "WB", "IN"]

CSS = """
#btn_pause { background: #2563eb !important; color: white !important; }
#btn_pause:hover { filter: brightness(1.05); }

#btn_start_green { background: #16a34a !important; color: white !important; border: none !important; }
#btn_start_green:hover { filter: brightness(1.05); }

/* Make disabled Start look dim */
#btn_start_green[disabled] { opacity: 0.45 !important; cursor: not-allowed !important; }

/* Add spacing under Save session name button */
#btn_save_session_name { margin-top: 0px !important; }

/* --- Button color tweaks (All Requirements tab) --- */
#btn_apply_edit_blue { background: #2563eb !important; color: white !important; border: none !important; }
#btn_apply_edit_blue:hover { filter: brightness(1.05); }

#btn_add_req_green { background: #16a34a !important; color: white !important; border: none !important; }
#btn_add_req_green:hover { filter: brightness(1.05); }

#btn_load_sessions_blue { background: #2563eb !important; color: white !important; border: none !important; }
#btn_load_sessions_blue:hover { filter: brightness(1.05); }

#btn_load_saved_green { background: #16a34a !important; color: white !important; border: none !important; }
#btn_load_saved_green:hover { filter: brightness(1.05); }

/* --- Final requirements table readability --- */
#final_reqs_table .ag-theme-alpine-dark .ag-cell {
  font-size: 14px !important;
  line-height: 1.35 !important;
  padding-top: 10px !important;
  padding-bottom: 10px !important;
  white-space: normal !important;
}

#final_reqs_table .ag-theme-alpine-dark .ag-header-cell-text {
  font-size: 13px !important;
  font-weight: 700 !important;
  line-height: 1.1 !important;
}

#final_reqs_table .ag-theme-alpine-dark .ag-header-row {
  height: 46px !important;
}

#final_reqs_table .ag-header-cell[col-id="1"],
#final_reqs_table .ag-cell[col-id="1"] {
  min-width: 90px !important;
  width: 90px !important;
  max-width: 90px !important;
  flex: 0 0 90px !important;
  text-align: center !important;
}

#final_reqs_table .ag-header-cell[col-id="4"],
#final_reqs_table .ag-cell[col-id="4"] {
  min-width: 120px !important;
  width: 120px !important;
  max-width: 180px !important;
  white-space: normal !important;
}

.gr-text-input input,
.gr-text-input textarea {
  background: rgba(255,255,255,0.06) !important;
  color: #eee !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
.gr-text-input input:focus,
.gr-text-input textarea:focus {
  background: rgba(255,255,255,0.08) !important;
}
"""

with gr.Blocks(
    title="Voice-Driven Requirements Elicitation Tool",
    css=CSS,
) as demo:

    gr.Markdown("## Voice-Driven Requirements Elicitation Tool")

    session_state = gr.State({})
    sid_to_reselect = gr.State(None)

    with gr.Row():
        # ---------------- LEFT COLUMN ----------------
        with gr.Column(scale=1):
            gr.Markdown("### New Project")

            proj_name = gr.Textbox(
                label="Project name",
                placeholder="e.g., Autonomous Driving Elicitation",
                lines=1,
                max_lines=1,
            )

            proj_domain = gr.Dropdown(
                label="Domain",
                choices=["Healthcare", "Autonomous Systems", "Aviation", "General SCS", "Other"],
                value="Autonomous Systems",
                filterable=False,
            )

            with gr.Row():
                btn_create_project = gr.Button("Create Project", variant="primary")
                btn_delete_project = gr.Button("Delete Project", variant="stop")

            project_dropdown = gr.Dropdown(
                label="Open project",
                choices=[],
                value=None,
                filterable=False,
            )

            proj_rename = gr.Textbox(
                label="Rename project",
                placeholder="Type new project name…",
                lines=1,
                max_lines=1,
            )
            btn_rename_project = gr.Button("Save project name", variant="secondary")

            # ---------- SESSIONS ----------
            gr.Markdown("### Sessions")

            with gr.Group():
                session_dropdown = gr.Dropdown(
                    label="Open session",
                    choices=[],
                    value=None,
                    filterable=False,
                )

                session_rename = gr.Textbox(
                    label="Rename session",
                    placeholder="Type new session name…",
                    lines=1,
                    max_lines=1,
                )

            btn_rename_session = gr.Button(
                "Save session name",
                variant="secondary",
                elem_id="btn_save_session_name",
            )

            with gr.Row():
                btn_new_session = gr.Button("New Session", variant="primary")
                btn_delete_session = gr.Button("Delete Session", variant="stop")

            gr.Markdown("---")
            btn_reload = gr.Button("Refresh All", variant="secondary")

        # ---------------- RIGHT COLUMN ----------------
        with gr.Column(scale=2):
            with gr.Tabs():
                # ---------- LIVE SESSION TAB ----------
                with gr.TabItem("Live Session"):
                    gr.Markdown("### Session Controls")

                    with gr.Row():
                        btn_start = gr.Button(
                            "▶ Start",
                            variant="secondary",
                            elem_id="btn_start_green",
                            interactive=False,
                        )
                        btn_pause = gr.Button("⏸ Pause", variant="secondary", elem_id="btn_pause")
                        btn_resume = gr.Button("⏵ Resume", variant="secondary")
                        btn_stop_save = gr.Button("⏹ Stop + Save", variant="stop")

                    live_line = gr.Markdown("")

                    transcript_panel = gr.Column(visible=False)
                    with transcript_panel:
                        transcript = gr.Textbox(label="Session Transcript", lines=10, value="")

                    gr.Markdown("### Classified Requirements")
                    cards = gr.HTML("<span style='opacity:.6'>No requirements yet.</span>")

                    with gr.Accordion("Developer Details", open=False):
                        raw_json = gr.JSON({})

                # ---------- ALL REQUIREMENTS TAB ----------
                with gr.TabItem("All Requirements"):
                    gr.Markdown("### Project-Wide Final Requirements")
                    gr.Markdown("Curate a final set for the project. Session logs remain unchanged.")

                    with gr.Row():
                        btn_load_from_sessions = gr.Button(
                            "Load from sessions", variant="secondary", elem_id="btn_load_sessions_blue"
                        )
                        btn_load_saved_final = gr.Button(
                            "Load saved final", variant="secondary", elem_id="btn_load_saved_green"
                        )
                        btn_save_final = gr.Button("Save final", variant="primary")

                    all_req_status = gr.HTML("<span style='opacity:.6'>Select a project, then load.</span>")

                    all_requirements_table = gr.Dataframe(
                        headers=[
                            "Requirement",
                            "FR/NFR",
                            "Confidence",
                            "Human-Centric",
                            "Human-Centric Tags",
                            "Session",
                        ],
                        datatype=["str", "str", "number", "number", "str", "str"],
                        type="array",
                        value=[],
                        row_count=10,
                        col_count=(6, "fixed"),
                        wrap=True,
                        interactive=True,
                        label="Requirements Table",
                        elem_id="final_reqs_table",
                    )

                    # Export section: hidden until Load saved final
                    with gr.Row(visible=False) as export_row:
                        btn_export_csv = gr.Button("Export saved final (CSV)", variant="secondary")

                    export_file = gr.File(
                        label="Download export",
                        interactive=False,
                        visible=False,
                    )

                    gr.Markdown("### Edit selected requirement")
                    selected_row_idx = gr.State(-1)

                    with gr.Row():
                        edit_req_text = gr.Textbox(label="Requirement", lines=3)

                    with gr.Row():
                        edit_frnfr = gr.Dropdown(
                            label="FR/NFR", choices=["FR", "NFR"], value="FR", interactive=True
                        )
                        edit_fr_conf = gr.Number(label="Confidence", value=0.0)
                        edit_p_hcr = gr.Number(label="Human-Centric", value=0.0)

                    edit_tags = gr.Dropdown(
                        label="Human-Centric Tags",
                        choices=HCR_TAG_CHOICES,
                        multiselect=True,
                        value=[],
                        interactive=True,
                    )

                    with gr.Row():
                        btn_apply_edit = gr.Button(
                            "Apply edit", variant="secondary", elem_id="btn_apply_edit_blue"
                        )
                        btn_delete_row = gr.Button("Delete row", variant="stop")
                        btn_add_row = gr.Button(
                            "Add new requirement", variant="secondary", elem_id="btn_add_req_green"
                        )

    # ---------------- INITIAL LOAD / REFRESH ----------------
    demo.load(fn=ui_refresh_projects, outputs=project_dropdown)
    btn_reload.click(fn=ui_refresh_projects, outputs=project_dropdown)

    # When project changes: refresh sessions + clear live UI + disable Start
    project_dropdown.change(
        fn=lambda p: ui_list_sessions(p, keep_value=None),
        inputs=project_dropdown,
        outputs=session_dropdown,
    )
    project_dropdown.change(
        fn=lambda: (
            {},
            gr.update(interactive=False),
            "",
            "",
            "<span style='opacity:.6'>No requirements yet.</span>",
            {},
            gr.update(visible=False),
        ),
        inputs=None,
        outputs=[session_state, btn_start, live_line, transcript, cards, raw_json, transcript_panel],
    )

    # On project change: clear All Requirements + hide export UI + clear export file
    project_dropdown.change(
        fn=lambda: (
            [],
            "<span style='opacity:.6'>Select a project, then load.</span>",
            -1,
            "",
            "FR",
            0.0,
            0.0,
            [],
            gr.update(visible=False),  # export_row hidden
            None,                      # export_file cleared (File expects None/path, not gr.update(value=...))
        ),
        inputs=None,
        outputs=[
            all_requirements_table,
            all_req_status,
            selected_row_idx,
            edit_req_text,
            edit_frnfr,
            edit_fr_conf,
            edit_p_hcr,
            edit_tags,
            export_row,
            export_file,
        ],
    )


    # ---------------- PROJECT CREATION ----------------
    btn_create_project.click(
        fn=ui_create_project,
        inputs=[proj_name, proj_domain],
        outputs=[project_dropdown, live_line],
    )

    # ---------------- PROJECT DELETE ----------------
    btn_delete_project.click(
        fn=ui_delete_project,
        inputs=[project_dropdown],
        outputs=[project_dropdown, session_dropdown, live_line, transcript, cards, raw_json, transcript_panel],
    )
    btn_delete_project.click(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=btn_start,
    )

    # ---------------- PROJECT RENAME ----------------
    btn_rename_project.click(
        fn=ui_rename_project,
        inputs=[project_dropdown, proj_rename],
        outputs=[project_dropdown, live_line, proj_rename],
    )

    # ---------------- SESSION SELECTION ----------------
    session_dropdown.change(
        fn=ui_load_session,
        inputs=[project_dropdown, session_dropdown],
        outputs=[live_line, transcript, cards, raw_json],
    )

    def _toggle_transcript_panel(session_id):
        return gr.update(visible=False) if session_id else gr.update()

    session_dropdown.change(
        fn=_toggle_transcript_panel,
        inputs=[session_dropdown],
        outputs=transcript_panel,
    )

    session_dropdown.change(
        fn=lambda sid: (gr.update(interactive=False) if sid else gr.update()),
        inputs=[session_dropdown],
        outputs=btn_start,
    )

    # ---------------- NEW SESSION ----------------
    btn_new_session.click(
        fn=ui_new_session,
        inputs=[project_dropdown],
        outputs=[session_state, live_line, transcript, cards, raw_json, session_dropdown, btn_start, transcript_panel],
    )

    # ---------------- SESSION RENAME ----------------
    btn_rename_session.click(
        fn=ui_rename_session_step1,
        inputs=[project_dropdown, session_dropdown, session_rename],
        outputs=[session_dropdown, live_line, session_rename, sid_to_reselect],
    )
    btn_rename_session.click(
        fn=ui_rename_session_step2,
        inputs=[project_dropdown, sid_to_reselect],
        outputs=[session_dropdown, sid_to_reselect],
    )

    # ---------------- DELETE SESSION ----------------
    btn_delete_session.click(
        fn=ui_delete_session,
        inputs=[project_dropdown, session_dropdown],
        outputs=[session_dropdown, live_line, transcript, cards, raw_json],
    )
    btn_delete_session.click(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=btn_start,
    )
    btn_delete_session.click(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=transcript_panel,
    )

    # ---------------- LIVE SESSION CONTROLS ----------------
    btn_start.click(
        fn=ui_start_recording,
        inputs=[project_dropdown, session_state],
        outputs=[live_line, transcript, cards, session_state, session_dropdown],
    )
    btn_start.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=transcript_panel,
    )

    btn_pause.click(
        fn=ui_pause,
        inputs=[session_state],
        outputs=[session_state, live_line],
    )

    btn_resume.click(
        fn=ui_resume,
        inputs=[project_dropdown, session_state],
        outputs=[live_line, transcript, cards, session_state, session_dropdown],
    )
    btn_resume.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=transcript_panel,
    )

    btn_stop_save.click(
        fn=ui_stop_and_save,
        inputs=[project_dropdown, session_state],
        outputs=[session_state, live_line, session_dropdown],
    )
    btn_stop_save.click(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=btn_start,
    )
    btn_stop_save.click(
        fn=lambda: gr.update(visible=False),
        inputs=None,
        outputs=transcript_panel,
    )

    # ---------------- ALL REQUIREMENTS TAB ----------------
    # Hide export when loading from sessions
    btn_load_from_sessions.click(
        fn=ui_load_all_requirements,
        inputs=[project_dropdown],
        outputs=[all_requirements_table, all_req_status],
    )
    btn_load_from_sessions.click(
        fn=lambda: (gr.update(visible=False), gr.update(value=None, visible=False)),
        inputs=None,
        outputs=[export_row, export_file],
    )

    # Show export when loading saved final
    btn_load_saved_final.click(
        fn=ui_load_saved_final,
        inputs=[project_dropdown],
        outputs=[all_requirements_table, all_req_status],
    )
    btn_load_saved_final.click(
        fn=lambda: (gr.update(visible=True), gr.update(value=None, visible=True)),
        inputs=None,
        outputs=[export_row, export_file],
    )

    btn_save_final.click(
        fn=ui_save_final,
        inputs=[project_dropdown, all_requirements_table],
        outputs=[all_req_status],
    )

    # Export (saved final only)
    btn_export_csv.click(
        fn=ui_export_saved_final_csv,
        inputs=[project_dropdown],
        outputs=[export_file, all_req_status],
    )

    def _pick_row_with_multitag(table_rows, evt: gr.SelectData):
        req, frnfr, conf, p_hcr, tags_str, idx = ui_pick_row(table_rows, evt)
        tags_list = [t.strip() for t in (tags_str or "").split(",") if t.strip()]
        tags_list = [t for t in tags_list if t in HCR_TAG_CHOICES]
        return req, frnfr, conf, p_hcr, tags_list, idx

    all_requirements_table.select(
        fn=_pick_row_with_multitag,
        inputs=[all_requirements_table],
        outputs=[edit_req_text, edit_frnfr, edit_fr_conf, edit_p_hcr, edit_tags, selected_row_idx],
    )

    def _apply_edit_with_multitag(table_rows, row_idx, req_text, frnfr, fr_conf, p_hcr, tags_list):
        tags_str = ", ".join(tags_list or [])
        return ui_apply_edit(table_rows, row_idx, req_text, frnfr, fr_conf, p_hcr, tags_str)

    btn_apply_edit.click(
        fn=_apply_edit_with_multitag,
        inputs=[all_requirements_table, selected_row_idx, edit_req_text, edit_frnfr, edit_fr_conf, edit_p_hcr, edit_tags],
        outputs=[all_requirements_table, all_req_status],
    )

    btn_delete_row.click(
        fn=ui_delete_row,
        inputs=[all_requirements_table, selected_row_idx],
        outputs=[all_requirements_table, all_req_status, selected_row_idx],
    )

    btn_add_row.click(
        fn=ui_add_row,
        inputs=[all_requirements_table],
        outputs=[all_requirements_table, all_req_status],
    )

demo.launch()