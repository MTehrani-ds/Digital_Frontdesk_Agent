# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import uuid
import os
import re

app = FastAPI(title="Digital Frontdesk – Agent v1")

SESSIONS: Dict[str, Dict[str, Any]] = {}
TICKETS: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    channel: Optional[str] = "webchat"
    practice_name: Optional[str] = "Example Dental Clinic"
    prior_state: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    session_id: str
    reply_text: str
    state: Dict[str, Any]
    actions_executed: List[Dict[str, Any]] = []
    ticket_id: Optional[str] = None


DEFAULT_STATE = {
    "step": "TRIAGE",  # TRIAGE -> COLLECT_CONTACT -> READY_TO_HANDOFF -> LIMITED_RESPONSE
    "intent": None,    # PRICING, INSURANCE, SERVICES, EMERGENCY, MEDICAL_ADVICE, OPENING_HOURS, BOOK_APPOINTMENT
    "procedure": None,
    "topic": None,
    "details": [],
    "collected": {"name": None, "phone": None, "best_time": None},
    "created_at": None,
    "updated_at": None
}

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def get_or_init_state(session_id: str, prior_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if prior_state:
        state = prior_state
    elif session_id in SESSIONS:
        state = SESSIONS[session_id]
    else:
        state = dict(DEFAULT_STATE)
        state["created_at"] = now_iso()
        state["updated_at"] = now_iso()
    return state

def save_state(session_id: str, state: Dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    SESSIONS[session_id] = state


def classify_intent_and_procedure(text: str) -> Dict[str, Optional[str]]:
    t = (text or "").lower().strip()

    # Intent detection (order matters)
    if any(x in t for x in ["antibiotic", "antibiotics", "amoxicillin", "penicillin"]):
        intent = "MEDICAL_ADVICE"
    elif any(x in t for x in ["emergency", "severe pain", "unbearable pain", "bleeding", "swollen", "swelling", "urgent"]):
        intent = "EMERGENCY"
    elif any(x in t for x in ["book", "booking", "appointment", "make an appointment", "schedule", "set an appointment"]):
        intent = "BOOK_APPOINTMENT"
    elif any(x in t for x in [
        "opening hours", "open hours", "working hours", "business hours",
        "when are you open", "when do you open", "when do you close",
        "opening time", "closing time", "hours"
    ]):
        intent = "OPENING_HOURS"
    elif any(x in t for x in ["insurance", "public insurance", "private pay", "private insurance", "coverage", "insured"]):
        intent = "INSURANCE"
    elif any(x in t for x in ["price", "cost", "how much", "pricing", "fee", "rates", "quote"]):
        intent = "PRICING"
    elif any(x in t for x in ["do you offer", "services", "what do you do", "offer", "treatments", "procedures"]):
        intent = "SERVICES"
    else:
        intent = None

    # Procedure detection
    procedure = None
    if any(x in t for x in ["implant", "implants"]):
        procedure = "implant"
    elif "cleaning" in t or "scale" in t or "scaling" in t:
        procedure = "cleaning"
    elif any(x in t for x in ["filling", "fillings", "cavity"]):
        procedure = "filling"
    elif "root canal" in t or "endodont" in t:
        procedure = "root_canal"
    elif any(x in t for x in ["crown", "crowns", "bridge", "bridges"]):
        procedure = "crown_bridge"
    elif any(x in t for x in ["check-up", "checkup", "consult", "consultation", "examination"]):
        procedure = "consultation"
    elif any(x in t for x in ["kids", "child", "children", "pediatric"]):
        procedure = "kids_dentistry"
    elif any(x in t for x in ["toothache", "pain", "emergency"]):
        if procedure is None:
            procedure = "emergency_consult"

    return {"intent": intent, "procedure": procedure}


def _looks_like_name(s: str) -> bool:
    s = s.strip()
    if not (1 <= len(s) <= 60):
        return False
    # Allow letters, spaces, hyphen, apostrophe
    if not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ \-']*", s):
        return False
    # Avoid generic answers
    banned = {"yes", "no", "okay", "ok", "sure", "thanks", "thank you"}
    if s.lower() in banned:
        return False
    return True


def update_collected_from_text(state: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Deterministic extraction.
    IMPORTANT: supports "Alex" as a name when we are explicitly asking for a name (COLLECT_CONTACT).
    """
    t = (text or "").strip()
    low = t.lower()

    # --- If we're collecting contact info, treat short plain text as the requested field ---
    if state.get("step") == "COLLECT_CONTACT":
        c = state.get("collected", {})
        if c.get("name") is None:
            # If user just typed "Alex"
            if _looks_like_name(t):
                c["name"] = t
                state["collected"] = c
                return state

    # Name patterns
    if state["collected"]["name"] is None:
        m = re.search(r"\bmy name is\b\s+(.+)$", low)
        if m:
            candidate = t[m.start():]  # use original casing
            # Extract after "my name is" in original text
            candidate = re.split(r"\bmy name is\b", candidate, flags=re.IGNORECASE)[-1].strip()
            if _looks_like_name(candidate):
                state["collected"]["name"] = candidate

        if state["collected"]["name"] is None:
            m2 = re.search(r"\bi am\b\s+(.+)$", low)
            if m2:
                candidate = re.split(r"\bi am\b", t, flags=re.IGNORECASE)[-1].strip()
                if _looks_like_name(candidate):
                    state["collected"]["name"] = candidate

    # Phone: accept + and digits
    if state["collected"]["phone"] is None:
        digits = "".join(ch for ch in t if ch.isdigit() or ch == "+")
        if len("".join(ch for ch in digits if ch.isdigit())) >= 8:
            state["collected"]["phone"] = digits

    # Best time: basic keywords
    if state["collected"]["best_time"] is None:
        if any(x in low for x in [
            "morning", "afternoon", "evening", "tomorrow", "today",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
        ]):
            state["collected"]["best_time"] = t

    return state


def infer_topic(state: Dict[str, Any]) -> str:
    intent = state.get("intent")
    proc = state.get("procedure")

    if intent == "OPENING_HOURS":
        return "Opening hours"
    if intent == "BOOK_APPOINTMENT":
        return "Appointment booking"
    if intent == "PRICING" and proc:
        return f"{proc.title()} pricing"
    if intent == "INSURANCE" and proc:
        return f"Insurance coverage for {proc.title()}"
    if intent == "INSURANCE":
        return "Insurance question"
    if intent == "EMERGENCY":
        return "Emergency / urgent issue"
    if intent == "MEDICAL_ADVICE":
        return "Medical advice request (needs clinician)"
    if intent == "SERVICES":
        return "Services inquiry"
    return "General inquiry"


def build_summary(state: Dict[str, Any]) -> str:
    parts = []
    if state.get("intent"):
        parts.append(f"Intent: {state['intent']}")
    if state.get("procedure"):
        parts.append(f"Procedure: {state['procedure']}")
    if state.get("topic"):
        parts.append(f"Topic: {state['topic']}")
    c = state.get("collected", {})
    if c.get("name") or c.get("phone") or c.get("best_time"):
        parts.append(f"Contact: name={c.get('name')}, phone={c.get('phone')}, best_time={c.get('best_time')}")
    if state.get("details"):
        parts.append("User said: " + " | ".join(state["details"][-3:]))
    return "\n".join(parts).strip()


def upsert_ticket(session_id: str, state: Dict[str, Any]) -> str:
    existing = None
    for tid, t in TICKETS.items():
        if t.get("session_id") == session_id:
            existing = tid
            break

    ticket_id = existing or str(uuid.uuid4())[:8]
    topic = state.get("topic") or infer_topic(state)
    summary = build_summary(state)

    ticket = {
        "ticket_id": ticket_id,
        "session_id": session_id,
        "created_at": TICKETS.get(ticket_id, {}).get("created_at") or now_iso(),
        "updated_at": now_iso(),
        "intent": state.get("intent"),
        "procedure": state.get("procedure"),
        "topic": topic,
        "summary": summary,
        "contact": state.get("collected", {}),
        "conversation_facts": state.get("details", [])[-10:],
        "status": "open" if state.get("step") != "RESOLVED" else "closed",
    }
    TICKETS[ticket_id] = ticket
    return ticket_id


ActionType = Literal["upsert_ticket", "notify_staff", "handoff_if_needed"]

class ToolResult(BaseModel):
    ok: bool
    data: Dict[str, Any] = {}


class Tools:
    def upsert_ticket(self, session_id: str, state: Dict[str, Any], **kwargs) -> ToolResult:
        tid = upsert_ticket(session_id, state)
        return ToolResult(ok=True, data={"ticket_id": tid})

    def notify_staff(self, session_id: str, state: Dict[str, Any], ticket_id: Optional[str] = None, **kwargs) -> ToolResult:
        return ToolResult(ok=True, data={"notified": True, "ticket_id": ticket_id})

    def handoff_if_needed(self, session_id: str, state: Dict[str, Any], **kwargs) -> ToolResult:
        return ToolResult(ok=True, data={"handoff_checked": True})


class AgentRunner:
    def __init__(self):
        self.tools = Tools()

    def run_actions(self, session_id: str, state: Dict[str, Any], actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        executed = []
        for a in actions:
            typ: str = a.get("type")
            params: Dict[str, Any] = a.get("params", {})

            if typ == "upsert_ticket":
                res = self.tools.upsert_ticket(session_id=session_id, state=state, **params)
            elif typ == "notify_staff":
                res = self.tools.notify_staff(session_id=session_id, state=state, **params)
            elif typ == "handoff_if_needed":
                res = self.tools.handoff_if_needed(session_id=session_id, state=state, **params)
            else:
                res = ToolResult(ok=False, data={"error": f"Unknown action type: {typ}"})

            executed.append({"type": typ, "ok": res.ok, "data": res.data})
        return executed


agent = AgentRunner()


def plan_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = []
    intent = state.get("intent")
    proc = state.get("procedure")
    collected = state.get("collected", {})
    has_contact_signal = any(collected.get(k) for k in ["name", "phone", "best_time"])
    meaningful = bool(intent or proc or state.get("topic") or state.get("details"))

    # No tickets for opening hours
    if intent != "OPENING_HOURS" and (meaningful or has_contact_signal):
        actions.append({"type": "upsert_ticket", "params": {}})

    if intent in ["EMERGENCY", "MEDICAL_ADVICE"]:
        actions.append({"type": "notify_staff", "params": {}})
        actions.append({"type": "handoff_if_needed", "params": {}})

    if state.get("step") == "READY_TO_HANDOFF":
        actions.append({"type": "notify_staff", "params": {}})

    return actions


def next_reply(state: Dict[str, Any], user_text: str) -> str:
    intent = state.get("intent")
    proc = state.get("procedure")
    step = state.get("step")
    collected = state.get("collected", {})

    if intent == "OPENING_HOURS" and step == "TRIAGE":
        state["step"] = "TRIAGE"
        return (
            "Our typical opening hours are:\n"
            "Monday–Friday: 09:00–18:00\n"
            "Saturday: 09:00–13:00\n"
            "Sunday: Closed\n\n"
            "Would you like to book an appointment or request a callback?"
        )

    if intent == "BOOK_APPOINTMENT":
        state["step"] = "COLLECT_CONTACT"
        missing = [k for k in ["name", "phone", "best_time"] if not collected.get(k)]
        if missing:
            field = missing[0]
            if field == "name":
                return "Sure — I can help with booking an appointment.\n\nTo get started, may I have your name?"
            if field == "phone":
                return "Thanks, {name}. What’s the best phone number to reach you?".format(name=collected.get("name", ""))
            if field == "best_time":
                return "When is the best time to contact you to confirm the appointment?"
        state["step"] = "READY_TO_HANDOFF"
        return "Perfect — I’ve noted your details and the team will contact you shortly to confirm the appointment."

    if intent == "MEDICAL_ADVICE":
        state["step"] = "LIMITED_RESPONSE"
        return (
            "I can’t safely provide medical advice over chat. Antibiotics or other treatments "
            "can only be recommended after a clinician has assessed you.\n\n"
            "If you have fever, facial swelling, trouble swallowing or breathing, or severe pain, "
            "please seek urgent care.\n\n"
            "If you’d like, you can share your name and phone number and we can arrange a clinician callback."
        )

    if intent == "EMERGENCY":
        state["step"] = "READY_TO_HANDOFF"
        return (
            "If this is severe pain, swelling, uncontrolled bleeding, or fever, "
            "please treat it as urgent and contact the clinic immediately "
            "(or emergency services if needed).\n\n"
            "If you share your name and phone number, we can arrange an urgent callback."
        )

    if intent == "INSURANCE":
        state["step"] = "TRIAGE"
        if not proc:
            return (
                "Insurance coverage depends on the type of treatment and your plan.\n\n"
                "Which service are you asking about (for example: cleaning, filling, implant)?"
            )
        return (
            f"Coverage can vary by plan and procedure. For **{proc.replace('_', ' ')}**, "
            "we can confirm details after a brief assessment and checking your insurance.\n\n"
            "If you’d like, share your name and phone number and we can arrange a callback with more details."
        )

    if intent == "PRICING":
        if not proc:
            state["step"] = "TRIAGE"
            return "Sure — which treatment are you asking about (for example: cleaning, filling, implant)?"
        state["step"] = "COLLECT_CONTACT"
        return (
            f"Pricing for **{proc.replace('_', ' ')}** depends on clinical assessment and case complexity.\n\n"
            "If you share your name, phone number, and best time to reach you, "
            "we can arrange a callback with an estimated price range."
        )

    if intent == "SERVICES":
        state["step"] = "TRIAGE"
        return (
            "We offer the following services:\n"
            "- Check-ups & consultations\n"
            "- Professional cleaning\n"
            "- Fillings\n"
            "- Root canal treatment (by assessment)\n"
            "- Crowns & bridges\n"
            "- Implants (by assessment)\n"
            "- Kids dentistry\n"
            "- Emergency pain consultations\n\n"
            "What would you like help with?"
        )

    # Generic contact collection continuation (if intent got lost)
    if step in ["COLLECT_CONTACT", "READY_TO_HANDOFF"]:
        missing = [k for k in ["name", "phone", "best_time"] if not collected.get(k)]
        if missing:
            field = missing[0]
            if field == "name":
                return "To proceed, may I have your name?"
            if field == "phone":
                return "Thanks. What’s the best phone number to reach you?"
            if field == "best_time":
                return "When is the best time to call you?"
        state["step"] = "READY_TO_HANDOFF"
        return "Thanks — I’ll pass this on to the team and they’ll contact you shortly."

    state["step"] = "TRIAGE"
    return (
        "How can I help you today?\n"
        "You can ask about opening hours, pricing, insurance, or booking an appointment."
    )


@app.get("/", response_class=HTMLResponse)
def ui():
    path = os.path.join(os.getcwd(), "chat.html")
    if not os.path.exists(path):
        return HTMLResponse(
            "<h3>chat.html not found</h3><p>Put chat.html next to main.py (same folder) and refresh.</p>",
            status_code=200
        )
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read(), status_code=200)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = get_or_init_state(req.session_id, req.prior_state)

    msg = (req.user_message or "").strip()
    if msg:
        state["details"].append(msg)

    # NLU
    nlu = classify_intent_and_procedure(msg)

    # IMPORTANT: don't wipe intent during contact collection when nlu returns None
    if nlu.get("intent") is not None:
        state["intent"] = nlu["intent"]
    elif state.get("step") == "TRIAGE":
        state["intent"] = state.get("intent")  # keep as-is (safe default)

    if nlu.get("procedure"):
        state["procedure"] = nlu["procedure"]

    # Topic
    state["topic"] = infer_topic(state)

    # Extract contact (step-aware)
    state = update_collected_from_text(state, msg)

    # Reply
    reply_text = next_reply(state, msg)

    # Plan + execute actions
    planned_actions = plan_actions(state)
    executed = agent.run_actions(req.session_id, state, planned_actions)

    ticket_id = None
    for e in executed:
        if e["type"] == "upsert_ticket" and e["ok"]:
            ticket_id = e["data"].get("ticket_id")

    for e in executed:
        if e["type"] == "notify_staff" and e["ok"] and ticket_id and "ticket_id" not in e["data"]:
            e["data"]["ticket_id"] = ticket_id

    save_state(req.session_id, state)

    return ChatResponse(
        session_id=req.session_id,
        reply_text=reply_text,
        state=state,
        actions_executed=executed,
        ticket_id=ticket_id
    )


@app.get("/debug/tickets")
def debug_tickets():
    return {"tickets": list(TICKETS.values())}

@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    return {"session_id": session_id, "state": SESSIONS.get(session_id)}

@app.get("/health")
def health():
    return {"ok": True, "time": now_iso()}
