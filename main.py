from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import uuid
import os

app = FastAPI(title="Digital Frontdesk – Agent v1")

# -----------------------------
# In-memory stores (demo)
# Replace with DB later
# -----------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}
TICKETS: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# Request/Response models
# -----------------------------
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


# -----------------------------
# Agent State
# -----------------------------
DEFAULT_STATE = {
    "step": "TRIAGE",  # TRIAGE -> COLLECT_CONTACT -> READY_TO_HANDOFF -> LIMITED_RESPONSE
    "intent": None,    # PRICING, INSURANCE, SERVICES, EMERGENCY, MEDICAL_ADVICE, OPENING_HOURS
    "procedure": None, # implant, cleaning, filling...
    "topic": None,     # short phrase: "Implant pricing", "Insurance question" ...
    "details": [],     # list of user utterances (evidence trail)
    "collected": {
        "name": None,
        "phone": None,
        "best_time": None
    },
    "created_at": None,
    "updated_at": None
}


# -----------------------------
# Utilities
# -----------------------------
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


# -----------------------------
# VERY SIMPLE NLU (replace with your existing logic)
# -----------------------------
def classify_intent_and_procedure(text: str) -> Dict[str, Optional[str]]:
    t = (text or "").lower().strip()

    # -------- Intent detection (order matters) --------
    # 1) High-risk / safety intents first
    if any(x in t for x in ["antibiotic", "antibiotics", "amoxicillin", "penicillin"]):
        intent = "MEDICAL_ADVICE"

    elif any(x in t for x in ["emergency", "severe pain", "unbearable pain", "bleeding", "swollen", "swelling", "urgent"]):
        intent = "EMERGENCY"

    # 2) Booking / scheduling (must come BEFORE opening hours, because "open" can appear in booking messages)
    elif any(x in t for x in [
        "book", "booking", "appointment", "make an appointment", "schedule", "set an appointment"
    ]):
        intent = "BOOK_APPOINTMENT"

    # 3) Opening hours (read-only info)
    elif any(x in t for x in [
        "opening hours", "open hours", "working hours", "business hours",
        "when are you open", "when do you open", "when do you close",
        "opening time", "closing time", "hours"
    ]):
        intent = "OPENING_HOURS"

    # 4) Insurance
    elif any(x in t for x in [
        "insurance", "public insurance", "private pay", "private insurance", "coverage", "insured"
    ]):
        intent = "INSURANCE"

    # 5) Pricing
    elif any(x in t for x in [
        "price", "cost", "how much", "pricing", "fee", "rates", "quote"
    ]):
        intent = "PRICING"

    # 6) Services / offerings
    elif any(x in t for x in [
        "do you offer", "services", "what do you do", "offer", "treatments", "procedures"
    ]):
        intent = "SERVICES"

    else:
        intent = None

    # -------- Procedure detection --------
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
        # only set if nothing else already matched
        if procedure is None:
            procedure = "emergency_consult"

    return {"intent": intent, "procedure": procedure}



def update_collected_from_text(state: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Replace with your working extraction.
    Keep it deterministic: only fill fields when confidently detected.
    """
    t = text.strip()
    low = t.lower()

    # Name: "My name is Ali"
    if state["collected"]["name"] is None and ("my name is" in low or "i am " in low):
        if "my name is" in low:
            name = t.split("is", 1)[1].strip()
        else:
            # crude: "I am Ali"
            name = t.split("am", 1)[1].strip()
        if 1 <= len(name) <= 60 and all(ch.isalpha() or ch in " -'" for ch in name):
            state["collected"]["name"] = name

    # Phone: digits (demo)
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
            state["collected"]["best_time"] = t.strip()

    return state


# -----------------------------
# Ticketing (tool target)
# -----------------------------
def infer_topic(state: Dict[str, Any]) -> str:
    intent = state.get("intent")
    proc = state.get("procedure")

    if intent == "OPENING_HOURS":
        return "Opening hours"

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
    """
    Create or update one ticket per session for demo.
    Ticket includes WHAT the user asked + summary.
    """
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


# -----------------------------
# Tools + Agent Runner
# -----------------------------
ActionType = Literal["upsert_ticket", "notify_staff", "handoff_if_needed"]

class ToolResult(BaseModel):
    ok: bool
    data: Dict[str, Any] = {}


class Tools:
    def upsert_ticket(self, session_id: str, state: Dict[str, Any], **kwargs) -> ToolResult:
        tid = upsert_ticket(session_id, state)
        return ToolResult(ok=True, data={"ticket_id": tid})

    def notify_staff(self, session_id: str, state: Dict[str, Any], ticket_id: Optional[str] = None, **kwargs) -> ToolResult:
        # Demo: log-only. Replace with Slack/webhook/email later.
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


# -----------------------------
# Planner (agent brain)
# -----------------------------
def plan_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Agent v1 planning rules:
    - Create/update ticket for meaningful clinical/admin intents (NOT opening hours).
    - Notify staff only for READY_TO_HANDOFF or EMERGENCY / MEDICAL_ADVICE.
    """
    actions = []
    intent = state.get("intent")
    proc = state.get("procedure")
    collected = state.get("collected", {})
    has_contact_signal = any(collected.get(k) for k in ["name", "phone", "best_time"])
    meaningful = bool(intent or proc or state.get("topic") or state.get("details"))

    # Do NOT create tickets for simple informational intents like opening hours
    if intent != "OPENING_HOURS" and (meaningful or has_contact_signal):
        actions.append({"type": "upsert_ticket", "params": {}})

    if intent in ["EMERGENCY", "MEDICAL_ADVICE"]:
        actions.append({"type": "notify_staff", "params": {}})
        actions.append({"type": "handoff_if_needed", "params": {}})

    if state.get("step") == "READY_TO_HANDOFF":
        actions.append({"type": "notify_staff", "params": {}})

    return actions


# -----------------------------
# Reply policy (safe + relevant)
# -----------------------------
def next_reply(state: Dict[str, Any], user_text: str) -> str:
    intent = state.get("intent")
    proc = state.get("procedure")

    # Opening hours (read-only info)
    if intent == "OPENING_HOURS":
        state["step"] = "TRIAGE"
        return (
            "Our typical opening hours are:\n"
            "Monday–Friday: 09:00–18:00\n"
            "Saturday: 09:00–13:00\n"
            "Sunday: Closed\n\n"
            "Would you like to book an appointment or request a callback?"
        )

    # Safety: medical advice / antibiotics
    if intent == "MEDICAL_ADVICE":
        state["step"] = "LIMITED_RESPONSE"
        return (
            "I can’t safely advise on antibiotics over chat. Antibiotics are only appropriate after a clinician "
            "assesses your symptoms and history.\n\n"
            "If you have fever, facial swelling, trouble swallowing/breathing, or severe pain, please seek urgent care.\n\n"
            "If you share your name + phone number, we can arrange a clinician callback."
        )

    # Emergency
    if intent == "EMERGENCY":
        state["step"] = "READY_TO_HANDOFF"
        return (
            "If this is severe pain, swelling, uncontrolled bleeding, or fever, please treat it as urgent and call the clinic "
            "right away (or emergency services if needed).\n\n"
            "If you share your name + phone number, we can arrange an urgent callback."
        )

    # Insurance question
    if intent == "INSURANCE":
        if not proc:
            state["step"] = "TRIAGE"
            return (
                "Pricing depends on the service and your insurance coverage. We accept public insurance and private pay (demo).\n\n"
                "Which service are you asking about (e.g., cleaning, filling, implant)?"
            )
        state["step"] = "TRIAGE"
        return (
            f"Coverage can vary by procedure and plan. For **{proc.replace('_', ' ')}**, we can confirm after a quick assessment "
            "and checking your insurance.\n\n"
            "If you share your name + phone number and best time to reach you, we can arrange a callback with an estimated range."
        )

    # Pricing question
    if intent == "PRICING":
        if not proc:
            state["step"] = "TRIAGE"
            return "Sure — which treatment are you asking about (e.g., cleaning, filling, implant)?"
        state["step"] = "COLLECT_CONTACT"
        return (
            f"For **{proc.replace('_', ' ')}**, pricing depends on clinical assessment and case complexity.\n\n"
            "If you share your **name**, **phone number**, and **best time to call**, we can arrange a callback with an estimated range."
        )

    # Services
    if intent == "SERVICES":
        state["step"] = "TRIAGE"
        return (
            "We offer:\n"
            "- Check-ups & consultations\n"
            "- Professional cleaning\n"
            "- Fillings\n"
            "- Root canal treatment (by assessment)\n"
            "- Crowns/bridges\n"
            "- Implants (by assessment)\n"
            "- Kids dentistry\n"
            "- Emergency pain consultations\n\n"
            "What do you need help with?"
        )

    # If we are collecting contact: guide intelligently
    c = state.get("collected", {})
    if state.get("step") in ["COLLECT_CONTACT", "READY_TO_HANDOFF"]:
        missing = [k for k in ["name", "phone", "best_time"] if not c.get(k)]
        if missing:
            field = missing[0]
            if field == "name":
                return "To arrange a callback, what’s your name?"
            if field == "phone":
                return "Thanks — what’s the best phone number to reach you?"
            if field == "best_time":
                return "When is the best time to call you (e.g., today afternoon, tomorrow morning)?"
        else:
            state["step"] = "READY_TO_HANDOFF"
            return "Perfect — I’ll pass this to the team and they’ll call you at your requested time."

    # Default
    state["step"] = "TRIAGE"
    return "How can I help you today — is it about opening hours, pricing, insurance, booking, or an urgent issue?"


# -----------------------------
# UI route (serves chat.html)
# -----------------------------
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


# -----------------------------
# Main chat endpoint
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = get_or_init_state(req.session_id, req.prior_state)

    # Evidence trail (improves ticket quality)
    msg = (req.user_message or "").strip()
    if msg:
        state["details"].append(msg)

    # NLU update
    nlu = classify_intent_and_procedure(msg)
    if nlu.get("intent"):
        state["intent"] = nlu["intent"]
    if nlu.get("procedure"):
        state["procedure"] = nlu["procedure"]

    # Topic for tickets/debug
    state["topic"] = infer_topic(state)

    # Extract contact fields
    state = update_collected_from_text(state, msg)

    # Reply
    reply_text = next_reply(state, msg)

    # Plan actions
    planned_actions = plan_actions(state)

    # Execute actions
    executed = agent.run_actions(req.session_id, state, planned_actions)

    # Pull ticket_id if created
    ticket_id = None
    for e in executed:
        if e["type"] == "upsert_ticket" and e["ok"]:
            ticket_id = e["data"].get("ticket_id")

    # If notify_staff ran, attach ticket_id for traceability
    for e in executed:
        if e["type"] == "notify_staff" and e["ok"] and ticket_id and "ticket_id" not in e["data"]:
            e["data"]["ticket_id"] = ticket_id

    # Save
    save_state(req.session_id, state)

    return ChatResponse(
        session_id=req.session_id,
        reply_text=reply_text,
        state=state,
        actions_executed=executed,
        ticket_id=ticket_id
    )


# -----------------------------
# Debug endpoints (demo)
# -----------------------------
@app.get("/debug/tickets")
def debug_tickets():
    return {"tickets": list(TICKETS.values())}

@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    return {"session_id": session_id, "state": SESSIONS.get(session_id)}


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": now_iso()}
