import os
import json
import hashlib
from collections import Counter
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================
#  Setup
# ============================================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Set it in a .env file or your environment.")

client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="Ask-AI Support Coach Demo",
    layout="wide",
)

# ============================================================
#  Mock Zendesk payload (for explainer)
# ============================================================

MOCK_ZENDESK_TICKET = {
    "id": 98765,
    "subject": "Same billing issue again - why wasn't this fixed?",
    "comments": [
        {
            "author_role": "customer",
            "body": (
                "Hi, I'm having the same billing issue I reported about 6 months ago. "
                "Your system is charging me twice for the same subscription period."
            ),
        },
        {
            "author_role": "agent",
            "body": (
                "Hi, sorry to hear that. Can you please send a screenshot of the double charge?"
            ),
        },
        {
            "author_role": "customer",
            "body": (
                "I already sent screenshots the last time this happened. "
                "They said it was a known issue and that it was fixed. "
                "Can you check my previous ticket?"
            ),
        },
        {
            "author_role": "agent",
            "body": (
                "If it's happening again it might be something different. "
                "Please send the screenshots again and I'll take a look."
            ),
        },
    ],
}


def normalize_zendesk_ticket(zendesk_payload: dict) -> str:
    """
    Take a Zendesk-style ticket JSON and turn it into the plain-text
    transcript we feed into the QA evaluator.
    """
    subject = zendesk_payload.get("subject", "")
    comments = zendesk_payload.get("comments", [])

    parts = []
    if subject:
        parts.append(f"Subject: {subject}")

    for c in comments:
        author_role = c.get("author_role", "unknown").strip() or "unknown"
        body = c.get("body", "").strip()
        if not body:
            continue
        label = author_role.capitalize()
        parts.append(f"{label}: {body}")

    return "\n\n".join(parts)


# ============================================================
#  Core model prompts
# ============================================================

def build_qa_prompt(ticket_text: str) -> str:
    """
    Build the core instructions for the QA coaching agent.
    """
    return f"""
You are a senior support QA coach.

You will be given the full text of a customer support ticket, including:
- subject
- customer messages
- agent replies
- any internal notes if present

Your job is to evaluate the AGENT'S performance only.

Evaluate the agent on the following 5 criteria, each scored from 1–5:
1. technical_accuracy – Did the agent provide factually correct and relevant information?
2. clarity_and_tone – Was the response clear, well-structured, and appropriately empathetic/professional?
3. diagnostic_depth – Did the agent ask good questions, check assumptions, and narrow down the root cause?
4. ownership_and_follow_through – Did the agent take ownership, set expectations, and move the case towards resolution?
5. escalation_judgment – Did the agent handle escalation appropriately (or explain when/why escalation was not needed)?

Then, analyze the ROOT CAUSE of any problems in this interaction.

You MUST choose one of:
- "agent_performance"
- "content_gap"
- "mixed"

Use these decision rules:

1) "agent_performance"
   Choose this when:
   - Appropriate documentation, playbooks, or prior cases clearly COULD have helped,
   - But the agent failed to use them, ignored clear signals, or behaved poorly.

2) "content_gap"
   Choose this when:
   - The main issue is that there is NO good documentation or playbook available,
   - And the agent is clearly struggling because the organization has not documented the scenario well.
   Strong signals for content_gap:
   - The customer asks explicitly for documentation or limits (e.g. API limits, SLAs, size caps),
     and the agent says there is nothing in the help center or KB.
   - The agent has to improvise or guess because there is no documented guidance.

   Example pattern (should be classified as content_gap, not pure agent_performance):
   - Customer: "Do you have any documented limits for CSV size or processing time?"
   - Agent: "I don't see anything in our help center about that. You may just need to try smaller batches."

3) "mixed"
   Choose this when:
   - There is clearly a documentation or content gap AND
   - The agent also misses obvious steps, ignores history, or mishandles tone/ownership.

Return STRICTLY valid JSON in this format:

{{
  "criteria": {{
    "technical_accuracy": {{
      "score": <integer 1-5>,
      "justification": "<short explanation>"
    }},
    "clarity_and_tone": {{
      "score": <integer 1-5>,
      "justification": "<short explanation>"
    }},
    "diagnostic_depth": {{
      "score": <integer 1-5>,
      "justification": "<short explanation>"
    }},
    "ownership_and_follow_through": {{
      "score": <integer 1-5>,
      "justification": "<short explanation>"
    }},
    "escalation_judgment": {{
      "score": <integer 1-5>,
      "justification": "<short explanation>"
    }}
  }},
  "overall_rating": {{
    "score": <integer 1-5>,
    "justification": "<2–3 sentences summarizing overall performance>"
  }},
  "root_cause": {{
    "label": "agent_performance" | "content_gap" | "mixed",
    "explanation": "<1–3 sentences explaining why>",
    "kb_article_suggestion": "<if content_gap or mixed, suggest a KB article title and outline; otherwise use an empty string>"
  }},
  "coaching_summary": "<3–6 bullet-style coaching points, in plain text>"
}}

Ticket:
\"\"\"{ticket_text}\"\"\"
"""


def evaluate_ticket(ticket_text: str) -> Dict[str, Any]:
    """
    Call OpenAI to evaluate the ticket and return parsed JSON.
    """
    prompt = build_qa_prompt(ticket_text)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful, precise QA evaluator. "
                    "Always return strictly valid JSON as requested."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    raw_content = response.choices[0].message.content or ""

    # Try to parse JSON safely (strip ```json fences if present)
    try:
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
        return data
    except Exception as e:
        return {"error": f"Failed to parse model output as JSON: {e}", "raw_output": raw_content}


def generate_kb_article(ticket_text: str, kb_suggestion: str) -> str:
    """
    Given a ticket and a KB article suggestion, ask the model to write
    a full draft knowledge base article to close the content gap.
    """
    prompt = f"""
You are a senior technical writer for a B2B SaaS support organization.

You are given:
- A support ticket transcript (including customer and agent messages)
- A suggested knowledge base article idea that would help prevent similar issues in the future

Write a **clear, structured KB article** that a support agent or customer could use for self-serve resolution.

Requirements:
- Neutral, professional tone.
- Sections:
  - Overview
  - Symptoms
  - Root cause (as far as can be inferred)
  - Step-by-step resolution
  - Verification steps
  - When to escalate

Return only the article text, formatted in Markdown (no JSON).

Suggested KB article idea:
\"\"\"{kb_suggestion}\"\"\"

Ticket:
\"\"\"{ticket_text}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an excellent technical writer for support KB articles."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content or ""


def generate_team_insights(coaching_items: List[Dict[str, str]]) -> str:
    """
    Aggregate multiple coaching summaries into a team-wide 'Coaching Canon',
    with explicit focus on business / revenue impact.
    """
    lines = []
    for item in coaching_items:
        label = item.get("label", "Unnamed ticket")
        root_cause = item.get("root_cause", "unknown")
        overall = item.get("overall_score", "N/A")
        summary = item.get("coaching_summary", "").strip()
        lines.append(
            f"- Ticket: {label}\n"
            f"  - Root cause: {root_cause}\n"
            f"  - Overall score: {overall}\n"
            f"  - Coaching summary:\n    {summary}\n"
        )

    coaching_block = "\n\n".join(lines)

    prompt = f"""
You are a Director of Support Enablement presenting to a VP of Support and a CRO.

You will be given several ticket-level coaching summaries.

Your job is to synthesize them into a **single, team-wide coaching document**
that clearly ties coaching themes to business outcomes.

Focus on:
- Common agent weaknesses and anti-patterns
- Systematic behavior patterns across tickets
- Org-wide coaching themes
- Recommended best practices for all agents
- Training or playbook updates that would help
- Repeated signals of documentation or content gaps

For each theme, explicitly connect to metrics such as:
- case deflection / self-serve rate
- first-contact resolution (FCR)
- time to resolution
- escalations avoided (Tier-2 / Engineering)
- churn / renewal risk
- compliance / audit risk for regulated customers

Return a concise, actionable document in Markdown.

Coaching items:
{coaching_block}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a seasoned Director of Support Enablement."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content or ""


# ============================================================
#  Sample tickets
# ============================================================

DEFAULT_EXAMPLE = """\
Customer: Jason Miller
Role: IT Manager, Larkspur Biotech
Priority: High
Plan: Enterprise Plus
SLA: 4 hrs
Subject: Inconsistent Access to SecureVault — Okta Group Sync Partially Failing

Customer Message
Hi,
We have a critical access issue across our “DataOps” and “ComplianceAudit” Okta groups. Roughly half of the users in each group are getting a 403 “Access Denied” when trying to log into SecureVault. The rest can log in normally.

We checked Okta — group membership looks correct, and SCIM sync says “Success” from this morning. We didn’t change any roles or entitlements.

We’re mid-audit for SOC 2. I need this resolved fast — we cannot afford downtime with our internal audit team locked out.

Thanks,
Jason

Agent Response (Tier 1 Agent — Julia)
Hi Jason,

Thanks for reaching out, and I’m sorry to hear about the access issues. Just to confirm — the affected users are all within the same two Okta groups? Have you had your Okta admin revalidate group assignments and SCIM settings?

Sometimes stale metadata causes SCIM to “succeed” but silently skip changes. Try manually syncing the group again and clearing your app cache. Let us know if the issue persists.

Best,
Julia

Customer Response
We already tried re-syncing groups and cleared app metadata. Still the same issue. Also, the problem is not consistent — one user logged in fine this morning and was blocked 20 minutes later. Another couldn’t log in, then suddenly got access an hour later.

Our CISO is now involved and wants a timeline + explanation. Please escalate.

Agent Response (Tier 2 Engineer — Mike)
Hi Jason,

Thanks for the update and additional detail. That fluctuation does sound unusual. It’s possible that your SCIM token is partially expired — we’ve seen issues where older SecureVault orgs had tokens that were not auto-rotated.

Please go to:
Admin → Integrations → SCIM → Regenerate your token and update it in Okta. Once that’s done, run a full re-sync. This should realign all group memberships. Let me know how it goes.

Cheers,
Mike

Customer Response
Mike — I already regenerated the SCIM token last week, and we re-synced this morning. Issue still happening.

Please dig deeper. This looks like something is intermittently breaking on your side — not Okta’s. Users don’t randomly gain and lose access.

Also, this happened to us six months ago — and your team said it was a race condition in your role evaluation engine. Is that back?

Agent Response
Thanks for the follow-up. I wasn’t aware of the prior incident, but I’ll review our internal logs for any recent regressions related to the role engine.

In the meantime, can you provide user emails for 3 affected and 3 unaffected users? That will help isolate the issue. I’ll escalate to Engineering if I find anything in the logs.

Appreciate your patience.

Customer Final Message
Sent the user list. Please do not ask us to “retry sync” again unless you’ve confirmed a root cause. We’re under scrutiny and cannot explain uncertainty to our auditors.

Also — please log this as a Sev 1 and give me a ticket ID I can reference in tomorrow’s audit debrief.
"""

SAMPLE_AGENT_LAZY = """\
Subject: Same billing issue again - why wasn't this fixed?

Customer: Hi, I'm having the same billing issue I reported about 6 months ago. Your system is charging me twice for the same subscription period.

Agent: Hi, sorry to hear that. Can you please send a screenshot of the double charge?

Customer: I already sent screenshots the last time this happened. They said it was a known issue and that it was fixed. Can you check my previous ticket?

Agent: If it's happening again it might be something different. Please send the screenshots again and I'll take a look.

Customer: This is really frustrating. I feel like I'm explaining this from scratch every time.

Agent: Once I get the screenshots I'll see what I can do about a refund.
"""

SAMPLE_CONTENT_GAP = """\
Subject: API timeout when uploading large CSV

Customer: Hi, every time I upload a 200k-row CSV via the /bulk-import API, the request times out after about 30 seconds. Smaller files work fine.

Agent: Hi there, thanks for reaching out. Timeouts can happen for a lot of reasons. Can you try again in an incognito window?

Customer: This is happening from our backend server, not a browser. We're calling the API directly.

Agent: Okay, in that case can you try from a different browser or device to see if it still happens?

Customer: Again, this is a backend integration, there is no browser. Do you have any documented limits for CSV size or processing time?

Agent: I don't see anything in our help center about that. If it's timing out, you may just need to try smaller batches.

Customer: That's really not an acceptable answer. We need to know what the limits are.
"""

SAMPLE_STRONG_AGENT = """\
Subject: Account locked after suspicious login alert

Customer: I got an email saying there was a suspicious login to my account from a new device and now I'm locked out. I need access for a client meeting in an hour.

Agent: Hi, thanks for getting in touch and I'm sorry for the stress this is causing, especially with a client meeting coming up. I can help get you back in securely.
...
"""

SAMPLE_TICKETS = {
    "Assignment example – SecureVault / Okta / SOC2": DEFAULT_EXAMPLE,
    "Agent issue – Ignored prior history (billing)": SAMPLE_AGENT_LAZY,
    "Content gap – No docs on API limits": SAMPLE_CONTENT_GAP,
    "Strong agent – Security lock, good handling": SAMPLE_STRONG_AGENT,
}

# ============================================================
#  Session state
# ============================================================

if "ticket_text" not in st.session_state:
    st.session_state["ticket_text"] = DEFAULT_EXAMPLE

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "last_ticket_text" not in st.session_state:
    st.session_state["last_ticket_text"] = ""

if "kb_draft" not in st.session_state:
    st.session_state["kb_draft"] = ""

if "coaching_history" not in st.session_state:
    st.session_state["coaching_history"] = []

if "coaching_keys" not in st.session_state:
    st.session_state["coaching_keys"] = []

if "team_insights" not in st.session_state:
    st.session_state["team_insights"] = ""

if "current_ticket_label" not in st.session_state:
    st.session_state["current_ticket_label"] = "Assignment example – SecureVault / Okta / SOC2"

# ============================================================
#  Sidebar
# ============================================================

with st.sidebar:
    st.subheader("What this demo shows")
    st.write(
        "- **Step 1:** Coach a single ticket like a QA lead\n"
        "- **Step 2:** Automatically close content gaps with KB drafts\n"
        "- **Step 3:** Roll up coaching into org-wide themes\n"
        "- **Step 4:** Tie gains to hard dollars (deflection & churn)\n"
    )
    st.markdown("---")
    st.write("Models: `gpt-4.1-mini`")
    st.markdown("---")
    st.caption(
        "Dark theme is controlled via `.streamlit/config.toml` – "
        "set `base = \"dark\"` there for default dark mode."
    )

# ============================================================
#  Header
# ============================================================

st.title("Ask-AI Support Coach Demo")
st.caption(
    "From a single messy ticket → to coaching, content gaps, team-wide patterns and a back-of-the-envelope ROI."
)
st.markdown("---")

# ============================================================
#  STEP 1 — Coach a single ticket
# ============================================================

st.markdown("## 1️⃣ Coach a single ticket")

st.markdown(
    "Drop in a real ticket (or load a sample). The agent is scored on five dimensions, "
    "and the system explains **what went wrong and why**."
)

col_samples, _ = st.columns([2, 1])
with col_samples:
    sample_choice = st.selectbox(
        "Load a sample ticket (optional)",
        ["None"] + list(SAMPLE_TICKETS.keys()),
    )

if st.button("Load selected sample"):
    if sample_choice != "None":
        st.session_state["ticket_text"] = SAMPLE_TICKETS[sample_choice]
        st.session_state["last_result"] = None
        st.session_state["kb_draft"] = ""
        st.session_state["current_ticket_label"] = sample_choice
        st.session_state["last_ticket_text"] = SAMPLE_TICKETS[sample_choice]

ticket_text = st.text_area(
    "Ticket transcript",
    key="ticket_text",
    height=260,
    help="Include subject + customer + agent messages. The more context, the better.",
)

col_run, _ = st.columns([1, 3])
with col_run:
    run_eval = st.button("Evaluate agent performance", type="primary")

if run_eval:
    if not ticket_text.strip():
        st.warning("Please paste a ticket before evaluating.")
    else:
        with st.spinner("Evaluating ticket like a QA lead..."):
            result = evaluate_ticket(ticket_text)

        st.session_state["last_result"] = result
        st.session_state["last_ticket_text"] = ticket_text
        st.session_state["kb_draft"] = ""
        st.session_state["team_insights"] = ""

        if "error" not in result:
            coaching = result.get("coaching_summary", "")
            root = result.get("root_cause", {}) or {}
            overall = result.get("overall_rating", {}) or {}
            root_label = root.get("label", "unknown")
            overall_score = overall.get("score", "N/A")

            if coaching:
                ticket_key = hashlib.sha256(ticket_text.encode("utf-8")).hexdigest()
                if ticket_key not in st.session_state["coaching_keys"]:
                    st.session_state["coaching_keys"].append(ticket_key)
                    st.session_state["coaching_history"].append(
                        {
                            "label": st.session_state.get("current_ticket_label", "Ad-hoc ticket"),
                            "overall_score": overall_score,
                            "root_cause": root_label,
                            "coaching_summary": coaching,
                        }
                    )

result = st.session_state.get("last_result")

if result is None:
    st.info("Run an evaluation to see scores, root cause, and coaching.")
else:
    if "error" in result:
        st.error(result["error"])
    else:
        criteria = result.get("criteria", {})
        overall = result.get("overall_rating", {})
        root = result.get("root_cause", {})
        coaching = result.get("coaching_summary", "")

        # Overall
        st.markdown("### Overall rating")
        overall_score = overall.get("score", "N/A")
        overall_just = overall.get("justification", "")
        st.markdown(f"**Score:** {overall_score} / 5")
        if overall_just:
            st.write(overall_just)

        # Criteria
        st.markdown("### Criteria breakdown")
        if criteria:
            bullets = []
            for key, value in criteria.items():
                score = value.get("score", "N/A")
                just = value.get("justification", "")
                label = key.replace("_", " ").title()
                bullets.append(f"- **{label}** — {score}/5\n  {just}")
            st.markdown("\n".join(bullets))
        else:
            st.write("No criteria were returned by the model.")

        # Root cause
        st.markdown("### Root cause analysis")

        root_label = root.get("label", "N/A")
        explanation = root.get("explanation", "")
        kb_suggestion = root.get("kb_article_suggestion", "")

        st.markdown(f"**Primary label:** `{root_label}`")
        if explanation:
            st.write(explanation)
        if kb_suggestion:
            st.markdown("**KB opportunity (short idea):**")
            st.write(kb_suggestion)

        # Coaching summary
        st.markdown("### Coaching summary (for team lead)")
        st.write(coaching or "No coaching summary returned.")

# ============================================================
#  STEP 2 — Close the loop on content gaps (KB drafting)
# ============================================================

st.markdown("---")
st.markdown("## 2️⃣ Close the loop on content gaps")

st.markdown(
    "When the system detects a **content gap** (or mixed case), it can draft a KB article "
    "so that the same question becomes self-serve instead of a ticket."
)

if result and "error" not in result:
    root = result.get("root_cause", {}) or {}
    root_label = root.get("label", "")
    kb_suggestion = root.get("kb_article_suggestion", "")

    if kb_suggestion and root_label in ("content_gap", "mixed"):
        st.success(
            f"This ticket surfaced a **{root_label}** – there’s documentation to be written here."
        )

        if st.button("Generate KB article draft", key="btn_generate_kb"):
            with st.spinner("Generating KB article draft..."):
                kb_text = generate_kb_article(
                    st.session_state.get("last_ticket_text", st.session_state["ticket_text"]),
                    kb_suggestion,
                )
                st.session_state["kb_draft"] = kb_text

        if st.session_state.get("kb_draft"):
            with st.expander("Proposed KB article draft", expanded=True):
                kb_text = st.text_area(
                    "KB article draft (editable)",
                    value=st.session_state["kb_draft"],
                    key="kb_editor",
                    height=280,
                )
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Approve"):
                        st.success(
                            "Approved. In production this would be pushed into your KB / help center."
                        )
                with col_b:
                    if st.button("Mark for revision"):
                        st.info("Marked for revision. A docs owner could refine this before publishing.")
                with col_c:
                    if st.button("Reject"):
                        st.warning(
                            "Rejected. This gap might need a different format (runbook, product change, etc.)."
                        )
    else:
        st.caption(
            "The last evaluated ticket was primarily an **agent-performance** issue. "
            "No KB draft is suggested for this one."
        )
else:
    st.caption("Run an evaluation first to see if any content gaps appear.")

# ============================================================
#  STEP 3 — Team-wide Coaching Canon
# ============================================================

st.markdown("---")
st.markdown("## 3️⃣ Team-wide Coaching Canon")

history = st.session_state.get("coaching_history", [])
if not history:
    st.write(
        "As you run this on real tickets, each evaluation is logged here. "
        "You can then roll them up into org-wide themes."
    )
else:
    st.write(
        f"{len(history)} unique ticket-level coaching summaries captured in this session."
    )

    root_counts = Counter(item.get("root_cause", "unknown") for item in history)
    st.markdown("**Pattern snapshot**")
    bullets = []
    if root_counts.get("agent_performance"):
        bullets.append(
            f"- {root_counts['agent_performance']} cases are primarily **agent-performance** "
            "(coaching and process issues)."
        )
    if root_counts.get("content_gap"):
        bullets.append(
            f"- {root_counts['content_gap']} cases are primarily **content-gaps** "
            "(deflectable via better docs)."
        )
    if root_counts.get("mixed"):
        bullets.append(
            f"- {root_counts['mixed']} cases are **mixed** (both coaching and docs)."
        )
    if bullets:
        st.markdown("\n".join(bullets))
        st.caption(
            "Content-gaps and mixed cases are where you unlock self-serve and deflection. "
            "Agent-performance cases point to enablement and QA."
        )

    if st.button("Generate team-wide coaching document", key="btn_team_insights"):
        with st.spinner("Aggregating into a Coaching Canon..."):
            insights = generate_team_insights(history)
            st.session_state["team_insights"] = insights

    if st.session_state.get("team_insights"):
        with st.expander("Team-wide Coaching Canon", expanded=True):
            st.markdown(st.session_state["team_insights"])

# ============================================================
#  STEP 4 — ROI / Business impact sandbox
# ============================================================

st.markdown("---")
st.markdown("## 4️⃣ ROI / business impact sandbox")

st.markdown(
    "This is a simple, transparent sandbox to connect **better coaching + better docs** "
    "to **support cost and churn**. Plug in your own numbers."
)

col_left, col_right = st.columns(2)

with col_left:
    monthly_tickets = st.number_input(
        "Monthly ticket volume",
        min_value=0,
        value=800,
        step=50,
    )
    avg_cost_per_case = st.number_input(
        "Average fully-loaded cost per handled case ($)",
        min_value=0.0,
        value=35.0,
        step=1.0,
    )
    current_deflection = st.slider(
        "Current self-serve / deflection rate (%)",
        min_value=0,
        max_value=80,
        value=20,
        step=5,
    )
    expected_deflection_uplift = st.slider(
        "Expected uplift in deflection with Ask-AI (%)",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
    )

with col_right:
    high_risk_tickets_per_month = st.number_input(
        "High-risk / strategic tickets per month",
        min_value=0,
        value=20,
        step=5,
        help="Think SOC2 audits, CISO-level issues, large logos."
    )
    revenue_per_churned_account = st.number_input(
        "Average annual revenue per strategic account ($)",
        min_value=0.0,
        value=50000.0,
        step=5000.0,
    )
    churn_prob_without = st.slider(
        "Churn probability on those tickets today (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
    )
    churn_prob_with = st.slider(
        "Churn probability after better QA / docs (%)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
    )

# Calculations
annual_tickets = monthly_tickets * 12
current_handled_by_agents = annual_tickets * (1 - current_deflection / 100.0)
new_deflection = min(current_deflection + expected_deflection_uplift, 100)
future_handled_by_agents = annual_tickets * (1 - new_deflection / 100.0)
delta_cases = max(current_handled_by_agents - future_handled_by_agents, 0)
annual_support_savings = delta_cases * avg_cost_per_case

annual_high_risk_tickets = high_risk_tickets_per_month * 12
expected_churn_without = annual_high_risk_tickets * (churn_prob_without / 100.0)
expected_churn_with = annual_high_risk_tickets * (churn_prob_with / 100.0)
avoided_churn_accounts = max(expected_churn_without - expected_churn_with, 0)
annual_revenue_preserved = avoided_churn_accounts * revenue_per_churned_account

total_annual_impact = annual_support_savings + annual_revenue_preserved

st.markdown("### Headline impact (annualized)")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric(
        "Support cost savings (est.)",
        f"${annual_support_savings:,.0f}",
        help="Fewer tickets hitting humans thanks to higher deflection / better first-contact resolution.",
    )
with col_b:
    st.metric(
        "Revenue preserved from lower churn (est.)",
        f"${annual_revenue_preserved:,.0f}",
    )
with col_c:
    st.metric(
        "Total annual impact (est.)",
        f"${total_annual_impact:,.0f}",
    )

# tie back to real evaluated tickets
history = st.session_state.get("coaching_history", [])
if history:
    root_counts_all = Counter(item.get("root_cause", "unknown") for item in history)
    total_hist = sum(root_counts_all.values())
    if total_hist > 0:
        content_share = (
            root_counts_all.get("content_gap", 0) + root_counts_all.get("mixed", 0)
        ) / total_hist
        st.caption(
            f"In this session, about **{content_share:.0%}** of evaluated tickets surfaced "
            "documentation or mixed content gaps – exactly where deflection and churn improvements come from."
        )

st.caption(
    "_In production, Ask-AI would plug in real ticket data and historical deflection / NRR numbers. "
    "Here it's intentionally transparent and tweakable for an interview demo._"
)

# ============================================================
#  Appendix: raw JSON + Zendesk explainer
# ============================================================

st.markdown("---")
st.markdown("## 📎 Appendix")

with st.expander("Raw JSON from the evaluator (for architects / SEs)"):
    result_for_raw = st.session_state.get("last_result")
    if result_for_raw and "error" not in (result_for_raw or {}):
        st.code(json.dumps(result_for_raw, indent=2), language="json")
    elif result_for_raw and "error" in result_for_raw:
        st.write("Last call returned an error:")
        st.code(result_for_raw.get("raw_output", ""), language="json")
    else:
        st.caption("Run a ticket evaluation first.")

with st.expander("How this plugs into Zendesk in production"):
    st.markdown(
        "In this demo, tickets are pasted as plain text. In production, tickets would be "
        "ingested directly from Zendesk via webhooks or exports, then normalized."
    )
    st.markdown("**Example Zendesk-style payload**")
    st.code(json.dumps(MOCK_ZENDESK_TICKET, indent=2), language="json")

    st.markdown("**Normalized transcript fed into the QA engine**")
    normalized_example = normalize_zendesk_ticket(MOCK_ZENDESK_TICKET)
    st.code(normalized_example, language="text")

    st.markdown(
        "The QA logic stays the same – we just map structured JSON into the same transcript format."
    )
