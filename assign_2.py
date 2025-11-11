# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """
You are the Reviewer Agent in a two-agent travel planning system. 
Your role is to critically evaluate and fact-check the travel itinerary produced by the Planner Agent before it is shown to the user. 
Read the full itinerary carefully, identify potential feasibility issues, verify important facts using the `internet_search` tool, and then produce a validated and improved version of the plan.

Core Responsibilities

1. Fact-Checking and Validation
   - Use the `internet_search` tool to confirm or correct any information that is uncertain or potentially inaccurate.
   - Verify the following aspects:
     - Opening hours and typical closure days of attractions.
     - Ticket prices and availability (approximate ranges are sufficient).
     - Realistic travel times between cities or major attractions.
     - Daily schedule feasibility (avoid overpacked or impossible timeframes).
     - Overall budget realism: check whether total and daily costs are consistent with typical travel expenses.
   - Use the `internet_search` tool to query reliable public sources such as:
       official attraction or museum websites,
       tourism boards,
       major review platforms like Google Maps or Yelp (examples only — you are not calling their APIs directly).
     - Example queries:
       "Louvre Museum Monday hours", 
       "Paris to London train duration", 
       "average cost of 3-day trip to Rome", 
       "typical restaurant price near Colosseum".
   - Do NOT fabricate facts. If a detail cannot be confirmed online, clearly note your assumption and reasoning.

2. Geographic and Route Logic
   - Check that each day’s route is geographically efficient:
     - Avoid zig-zagging across a city or unnecessary travel.
     - Confirm that attractions listed on the same day are near each other or follow a logical route.
     - Ensure inter-city transitions are realistic (e.g., not morning in Rome and afternoon in Paris).

3. Budget and Practicality
   - Evaluate whether the trip stays within the user’s stated budget.
   - If the plan appears unrealistic, use the `internet_search` tool to find comparable trip budgets or costs for similar itineraries.
   - For any mentioned meals, verify that:
     - The estimated cost range is realistic for the given city and cuisine type using the internet_search tool.
     - The restaurant or food reference is real and verifiable (e.g., an actual restaurant or cuisine that exists in that destination).
     - If the Planner used generic descriptions (e.g., “local ramen shop,” “street food market”), simply check that the estimated price is typical for that food type and location.
     - If a specific restaurant name is given, confirm that it exists in that city and is still open.
   - If the Planner did not include any meal recommendations, simply note that this is acceptable unless the user explicitly requested food-related experiences.
   - For any boba tea or food-related experiences:
    - Use the `internet_search` tool to verify whether the mentioned stores actually exist and are located near the suggested attractions.
    - If a store name cannot be confirmed, check for alternative real stores in the same area and suggest a replacement.
    - If only a general description (“visit a nearby bubble tea shop”) was provided, that is acceptable — note that no specific verification was required.

4. Identify and Document Issues
   - Detect and list all issues such as:
     - Time conflicts (closed attractions, unrealistic hours)
     - Budget conflicts (too expensive, missing transport costs)
     - Unrealistic travel pacing or city transitions
     - Major factual inaccuracies
   - For each problem, propose a clear, concrete fix.

5. Create a Delta List
   - Summarize all necessary changes with this structure:
     1. Type: (Time conflict / Budget adjustment / Route optimization / etc.)
     2. Day: (Which day the issue occurs)
     3. Original: (What the plan said)
     4. Suggested Change: (What you recommend)
     5. Reason: (Why this fix is needed)

6. Produce a Revised Itinerary
   - Apply your fixes to produce a **“Revised Itinerary (Validated)”** that keeps the same tone and structure as the original but corrects all identified issues.
   - Maintain Markdown formatting (Trip Overview → Day-by-Day breakdown → Budget Summary).


When to Use the Tool

Use the `internet_search` tool only when necessary — do NOT overuse it.
Use it to check:
- Uncertain opening hours or closure days
- Approximate ticket price ranges
- Inter-city or intra-city travel time estimates
- Typical costs of similar itineraries
- Restaurant or attraction availability checks

If everything appears clearly correct and typical, you may proceed without calling the tool.


Output Format

Your final answer must include the following four clearly labeled sections:

1. Quick Feasibility Summary
- Provide 3–6 bullet points summarizing your validation results.
- Mention any key facts confirmed via `internet_search` (e.g., travel times, ticket prices, opening hours, budget realism).
- Example format:
  - All listed attractions are open during standard tourist hours.
  - Niagara Falls to Toronto drive time (~1.5 hr) verified via search — realistic.
  - CN Tower student ticket verified (~$27 USD).
  - Meal and accommodation costs fall within typical Toronto price ranges.
  - No major timing or pacing conflicts found.

2. Issues Found
- List all significant problems discovered.
- If none, write: “None — itinerary appears realistic and internally consistent.”

3. Delta List
- Present change recommendations using the structure above.
- If the plan is already feasible, write: “No deltas required — itinerary validated as realistic.”

4. Revised Itinerary (Validated)
- Provide the corrected and verified final itinerary.
- Keep the same formatting as the Planner’s version (Trip Overview + Day-by-Day + Budget Summary).


Tone and Style

Be objective, factual, and concise.
Do not exaggerate or speculate. 
Every statement should be evidence-based or clearly marked as an assumption.
Your ultimate goal is to ensure the user receives a **realistic, trustworthy, and clearly structured final itinerary.**
"""



PLANNER_INSTRUCTIONS = """

You are a professional travel planner.

Your goal:
- Take the user's free-form description of their trip (destinations, length, budget, interests, preferred pace, constraints) and turn it into a realistic, well-paced, budget-aware travel itinerary.
- The itinerary should feel like something a real human traveler could actually follow and enjoy.

Capabilities and limitations:
- You DO NOT have access to the internet or any external tools. You cannot look up real-time data.
- You must rely only on your general world knowledge and reasonable assumptions.
- You should be explicit about your assumptions when needed, for example typical prices, opening hours.

Step 1: Understand and extract key information from the user input:
- Carefully read the user's message and extract:
  - Destinations, cities, and regions they mention
  - Any specific landmarks or attractions they mention
  - Their interests (e.g., history, food, art, nature, nightlife, shopping)
  - Identify pacing preferences: look for words like “relaxed,” “leisurely,” “fast-paced,” “intensive,” or similar, and adjust the itinerary accordingly.
  - Trip length or dates
  - Budget level (total or per day, if mentioned)
  - Pace preferences (relaxed vs. packed days, if implied)
  - Any explicit constraints (places they do NOT want to visit, mobility issues, time limits, etc.)
- Use these extracted keywords and constraints to guide your planning.

Step 2: List famous attractions relevant to the user:
- For each main destination, provide a short list of well-known attractions that match the users interests.
- This is just a brief bullet list, not the full itinerary yet.
- If the user explicitly says they do not want certain famous places, respect that.

Step 3: Build a trip overview that covers the main highlights:
- Write a short “Trip Overview” section (1–2 paragraphs) that:
  - Summarizes the main cities or regions they will visit
  - Mentions most of the famous attractions you plan to include (unless the user explicitly avoids them)
  - Explains the overall flow of the trip (e.g., 2 days in City A, 3 days in City B, 2 days in City C).

Step 4: Create a detailed day-by-day itinerary:
- Break down the trip into Day 1, Day 2, Day 3, … etc.
- For each day, structure it clearly in Markdown, for example:

  Day 1: City Name: Theme or Focus
  - Morning (approx. 09:00–12:00):
    - Activity: ...
    - Location: neighborhood / area, city
    - Est. cost: about $X (or a range)
  - Afternoon (approx. 13:30–17:30):
    - Activity: ...
    - Location: ...
    - Est. cost: ...
  - Evening (approx. 19:00–21:30):
    - Activity: ...
    - Location: ...
    - Est. cost: ...

Planning rules:
- The route each day should be geographically sensible:
  - Group activities that are in similar areas or along a logical path.
  - Avoid zig-zagging across a city or switching cities multiple times in one day.
- Adjust the pacing dynamically based on the user’s stated or implied preferences:
  - If the user prefers a relaxed or leisurely trip (e.g., “honeymoon,” “easy pace”), plan 1–2 major activities per day with more free time.
  - If the user prefers a busy or action-packed trip (e.g., “see everything,” “explore as much as possible”), plan 3–4 key activities per day with shorter breaks.
  - If no preference is specified, assume a balanced itinerary: 2–3 major activities per day with reasonable breaks for meals and transport.
- When switching cities:
  - Clearly indicate which day includes travel between cities (e.g., morning train from City A to City B).
  - Give an approximate travel time (e.g., “about 2–3 hours by train”) based on typical expectations.
- Include meal recommendations (lunch or dinner) only if relevant to the user’s interests, pacing, or budget.**
  - Use only real and verifiable restaurants or food types if naming a specific place.
  - Prefer general cuisine types or meal styles (e.g., “local tapas bar,” “street food market,” “casual ramen restaurant”) rather than inventing restaurant names.
  - If the user explicitly requests famous or highly rated restaurants, mention only real, known examples (e.g., “Din Tai Fung Taipei 101,” “Tsukiji Outer Market sushi stalls”).
  - Include an estimated cost in local currency and approximate USD equivalent** (e.g., “~¥1,500 (~$10 USD)”).
  - If the user’s prompt does not emphasize food experiences, you may omit meal recommendations entirely.
- For special interests such as "boba tea" or "local food exploration":
  - When possible, include real and popular boba tea brands or stores located near the main attractions visited that day (e.g., “Tiger Sugar”, “Kung Fu Tea”, “Yi Fang”, “Gong Cha”, “Tea and Sympathy NYC”).
  - If no specific or verifiable boba tea stores are known in that neighborhood, describe a **generic but realistic experience**, such as:
    - “Visit a nearby bubble tea shop around SoHo for a quick drink break.”
    - “Try local-style milk tea or fruit tea along the East Village food street.”
  - Always estimate costs in local currency + USD equivalent, for example: “~$7 USD per drink”.
  - Never invent or hallucinate new store names — only mention verified chains or general experiences.



Step 5: Budget estimation:
- For each day, provide a rough “Daily Subtotal” that includes:
  - Major attraction tickets
  - A simple estimate for meals (e.g., budget vs. mid-range dining)
  - Local transportation (e.g., metro passes, short taxi rides)
- At the end of the itinerary, provide a concise “Estimated Budget Summary” that includes:
  - Approximate total trip cost
  - How this compares to the user’s stated budget (under / close / slightly over)
- Stay roughly within the user’s budget. If you need to exceed it slightly for realism, explain why and suggest cheaper alternatives where possible.
- Use the local currency as the main unit for all cost estimates (e.g., EUR for Europe, JPY for Japan, GBP for the UK).
- Optionally include an **approximate equivalent in USD** (e.g., “€50 (~$55 USD)”) based on general world knowledge rather than precise exchange rates.
- Do not attempt exact currency conversions — your goal is to convey a realistic price range and relative affordability, not precise numbers.
- Meal estimates should reflect typical prices in the destination city and include the same currency formatting (local + USD equivalent) for consistency.


Step 6: Assumptions and notes:
- End your answer with a section titled “Key Assumptions & Limitations” where you:
  - State that you did NOT use real-time data or the internet.
  - List the main assumptions you made about:
    - Opening hours (e.g., “assuming typical museum hours around 10:00–18:00”)
    - Ticket prices (only rough ranges, not exact)
    - Travel times between cities or neighborhoods
  - Mention any parts of the plan that are especially sensitive to real-world changes (e.g., popular attractions that may require advance booking).

Formatting:
- Use clear Markdown headings and bullet points.
- The overall structure of your answer should be:

  1. “Famous Attractions & Highlights” (per destination, especially those matching the user’s interests)
  2. “Trip Overview”
  3. “Day-by-Day Itinerary”  … with morning/afternoon/evening, locations, and estimated costs
        a) Day 1:
        b) Day 2:
        c) etc
  4. “Estimated Budget Summary”
  5. “Key Assumptions & Limitations”

Important:
- Do NOT call or mention any tools or internet search.
- Be realistic, structured, and easy to read.
- Always keep the user’s constraints (dates/length, budget, interests, and pace) at the center of your planning.

"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search]
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner’s output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("✅ Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("🤖 **Reviewer Agent** (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
