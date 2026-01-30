from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import uuid

app = FastAPI(title="Dental Agent v1")

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
    "intent": None,    # e.g. PRICING, INSURANCE, SERVICES, EMERGENCY, MEDICAL_ADVICE
    "procedure": None, # e.g. implant, cleaning, filling
    "topic": None,     # short phrase: "Implant pricing", "Insurance coverage", "Emergency pain"
    "details": [],     # list of user facts / key utterances
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
    t = text.lower().strip()

    # Intent
    if any(x in t for x in ["antibiotic", "antibiotics", "amoxicillin", "penicillin"]):
        intent = "MEDICAL_ADVICE"
    elif any(x in t for x in ["emergency", "severe pain", "bleeding", "swollen", "swelling", "urgent"]):
        intent = "EMERGENCY"
    elif "insurance" in t or "public insurance" in t or "private" in t:
        intent = "INSURANCE"
    elif any(x in t for x in ["price", "cost", "how much", "pricing"]):
        intent = "PRICING"
    elif any(x in t for x in ["do you offer", "services", "what do you do", "offer"]):
        intent = "SERVICES"
    else:
        intent = None

    # Procedure
    procedure = None
    if "implant" in t or "implants" in t:
        procedure = "implant"
    elif "cleaning" in t:
        procedure = "cleaning"
    elif "filling" in t:
        procedure = "filling"
    elif "root canal" in t:
        procedure = "root_canal"
    elif "crown" in t or "bridge" in t:
        procedure = "crown_bridge"
    elif "check" in t or "consult" in t:
        procedure = "consultation"

    return {"intent": intent, "procedure": procedure}


def update_collected_from_text(state: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Replace this with your working extraction.
    Keep it deterministic: only fill fields when confidently detected.
    """
    t = text.strip()

    # SUPER naive examples:
    # Name: "My name is Ali"
    if state["collected"]["name"] is None and "my name is" in t.lower():
        name = t.split("is", 1)[1].strip()
        if 1 <= len(name) <= 60:
            state["collected"]["name"] = name

    # Phone: look for digits (demo)
    if state["collected"]["phone"] is None:
        digits = "".join(ch for ch in t if ch.isdigit() or ch in "+")
        if len("".join(ch for ch in digits if ch.isdigit())) >= 8:
            state["collected"]["phone"] = digits

    # Best time: keywords
    if state["collected"]["best_time"] is None:
        low = t.lower()
        if any(x in low for x in ["morning", "afternoon", "evening", "tomorrow", "today", "monday", "tuesday",
                                  "wednesday", "thursday", "friday", "saturday", "sunday"]):
            state["collected"]["best_time"] = t.strip()

    return state


# -----------------------------
# Ticketing (tool target)
# -----------------------------
def upsert_ticket(session_id: str, state: Dict[str, Any]) -> str:
    """
    Create or update one ticket per session for demo.
    The key improvement: ticket includes WHAT the user asked + summary.
    """
    # Find existing ticket
    existing = None
    for tid, t in TICKETS.items():
        if t.get("session_id") == session_id:
            existing = tid
            break

    ticket_id = existing or str(uuid.uuid4())[:8]

    # Build summary fields
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
        "conversation_facts": state.get("details", [])[-10:],  # last 10 key lines
        "status": "open" if state.get("step") != "RESOLVED" else "closed",
    }

    TICKETS[ticket_id] = ticket
    return ticket_id


def infer_topic(state: Dict[str, Any]) -> str:
    intent = state.get("intent")
    proc = state.get("procedure")
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
        # For demo: just log in result. Replace with Slack/webhook/email later.
        return ToolResult(ok=True, data={"notified": True, "ticket_id": ticket_id})

    def handoff_if_needed(self, session_id: str, state: Dict[str, Any], **kwargs) -> ToolResult:
        # For v1: this tool doesn't do much; the "handoff" is mostly handled by reply logic.
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
# Planner (this is the "agent brain")
# -----------------------------
def plan_actions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Agent v1 planning rules:
    - Always upsert ticket once we have any meaningful intent/procedure/topic or once contact info starts appearing.
    - Notify staff only when ready_to_handoff OR emergency/medical advice.
    """
    actions = []

    intent = state.get("intent")
    proc = state.get("procedure")
    collected = state.get("collected", {})
    has_contact_signal = any(collected.get(k) for k in ["name", "phone", "best_time"])

    meaningful = bool(intent or proc or state.get("topic") or state.get("details"))

    if meaningful or has_contact_signal:
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
    """
    Keep this deterministic and domain-safe.
    Key fix vs earlier behavior: reply MUST reflect user intent/procedure first,
    not immediately default to callback collection.
    """
    intent = state.get("intent")
    proc = state.get("procedure")

    # Safety: medical advice / antibiotics
    if intent == "MEDICAL_ADVICE":
        state["step"] = "LIMITED_RESPONSE"
        return (
            "I can’t safely advise on antibiotics over chat. "
            "Antibiotics are only appropriate after a clinician assesses your symptoms and history.\n\n"
            "If you’re having fever, facial swelling, trouble swallowing/breathing, or severe pain, "
            "please seek urgent care immediately.\n\n"
            "If you want, share your name + phone number and we can arrange a clinician callback."
        )

    # Emergency
    if intent == "EMERGENCY":
        state["step"] = "READY_TO_HANDOFF"
        return (
            "If this is severe pain, swelling, uncontrolled bleeding, or fever, "
            "please treat it as urgent and call the clinic right away (or emergency services if needed).\n\n"
            "If you share your name + phone number, we can arrange an urgent callback."
        )

    # Insurance question
    if intent == "INSURANCE":
        # Keep it relevant; ask for the procedure if missing
        if not proc:
            state["step"] = "TRIAGE"
            return (
                "Pricing depends on the service and your insurance coverage. "
                "We accept public insurance and private pay (demo).\n\n"
                "Which service are you asking about (e.g., cleaning, filling, implant)?"
            )
        else:
            state["step"] = "TRIAGE"
            return (
                f"Coverage can vary by procedure and plan. For **{proc.replace('_',' ')}**, "
                "we can confirm after a quick assessment and checking your insurance.\n\n"
                "If you share your name + phone number and best time to reach you, we can arrange a callback with an estimate range."
            )

    # Pricing question
    if intent == "PRICING":
        if not proc:
            state["step"] = "TRIAGE"
            return "Sure — which treatment are you asking about (e.g., cleaning, filling, implant)?"
        state["step"] = "COLLECT_CONTACT"
        return (
            f"For **{proc.replace('_',' ')}**, pricing depends on clinical assessment and case complexity.\n\n"
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
            # Ask for the next missing item only
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
    return "How can I help you today — is it about pricing, insurance, booking, or an urgent issue?"


# -----------------------------
# Main chat endpoint
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = get_or_init_state(req.session_id, req.prior_state)

    # Keep a compact evidence trail (this improves tickets a lot)
    state["details"].append(req.user_message.strip())

    # 1) NLU update
    nlu = classify_intent_and_procedure(req.user_message)
    if nlu.get("intent"):
        state["intent"] = nlu["intent"]
    if nlu.get("procedure"):
        state["procedure"] = nlu["procedure"]

    # Optional: set topic early to improve ticket quality
    state["topic"] = infer_topic(state)

    # 2) Extract contact fields (your existing working function should go here)
    state = update_collected_from_text(state, req.user_message)

    # 3) Generate reply
    reply_text = next_reply(state, req.user_message)

    # 4) Plan actions (agent planning)
    planned_actions = plan_actions(state)

    # 5) Execute actions (agent tool calls)
    executed = agent.run_actions(req.session_id, state, planned_actions)

    # Pull ticket_id if created
    ticket_id = None
    for e in executed:
        if e["type"] == "upsert_ticket" and e["ok"]:
            ticket_id = e["data"].get("ticket_id")

    # If notify_staff ran, attach ticket_id to its data for traceability
    for e in executed:
        if e["type"] == "notify_staff" and e["ok"] and ticket_id and "ticket_id" not in e["data"]:
            e["data"]["ticket_id"] = ticket_id

    # 6) Save session state
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
