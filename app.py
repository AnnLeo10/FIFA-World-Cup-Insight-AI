
import streamlit as st
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

DATA_URL = (
    "https://huggingface.co/datasets/Mominullptr/"
    "fifa-world-cup-2026-dataset/resolve/main/matches_detailed.csv"
)

st.set_page_config(
    page_title="FIFA World CupInsight AI",
    page_icon="⚽",
    layout="centered",
)


@st.cache_data
def load_matches():
    # Load historical World Cup matches (1930–2022)
    historical = pd.read_csv("data/raw/matches_1930_2022.csv")
    historical.columns = historical.columns.str.strip()

    historical["Year"] = pd.to_numeric(
        historical["Year"], errors="coerce"
    )

    # Standardize historical column names
    historical = historical.rename(columns={
        "home_team": "home_team_name",
        "away_team": "away_team_name",
        "Year": "year",
        "Date": "date",
        "Round": "stage_name",
        "Venue": "stadium_name",
        "Host": "host",
    })

    # Load the 2026 dataset from Hugging Face
    DATA_URL = (
        "https://huggingface.co/datasets/Mominullptr/"
        "fifa-world-cup-2026-dataset/resolve/main/matches_detailed.csv"
    )

    current = pd.read_csv(DATA_URL)
    current.columns = current.columns.str.strip()

    # Standardize the year column
    current["year"] = 2026

    # Combine both datasets
    combined = pd.concat(
        [historical, current],
        ignore_index=True,
        sort=False,
    )

    return combined

@st.cache_resource
def get_client():
    workspace = WorkspaceClient(profile="football-agent")
    return DatabricksOpenAI(workspace_client=workspace)

matches = load_matches()


def calculate_goal_summary(data):
    """Calculate goals from available, recorded match scores."""
    import pandas as pd

    home_col = (
        "home_team_name" if "home_team_name" in data.columns
        else "home_team"
    )
    away_col = (
        "away_team_name" if "away_team_name" in data.columns
        else "away_team"
    )

    if "year" in data.columns:
        year_col = "year"
    elif "Year" in data.columns:
        year_col = "Year"
    else:
        return "The dataset has no year column."

    required = [home_col, away_col, "home_score", "away_score", year_col]
    missing = [col for col in required if col not in data.columns]

    if missing:
        return f"Cannot calculate totals; missing columns: {missing}"

    # Count only matches with both scores recorded.
    scores = data.copy()
    scores["home_score"] = pd.to_numeric(
        scores["home_score"], errors="coerce"
    )
    scores["away_score"] = pd.to_numeric(
        scores["away_score"], errors="coerce"
    )
    scores = scores.dropna(
        subset=[home_col, away_col, "home_score", "away_score", year_col]
    )

    if scores.empty:
        return "No completed matches with recorded scores were found."

    scores["total_goals"] = scores["home_score"] + scores["away_score"]

    team_goals = pd.concat([
        scores[[home_col, "home_score"]].rename(
            columns={home_col: "team", "home_score": "goals"}
        ),
        scores[[away_col, "away_score"]].rename(
            columns={away_col: "team", "away_score": "goals"}
        ),
    ])

    team_totals = (
        team_goals.groupby("team", as_index=False)["goals"]
        .sum()
        .sort_values("goals", ascending=False)
    )

    total_goals = int(scores["total_goals"].sum())

    return (
        f"Matches with recorded scores: {len(scores)}\n"
        f"Total goals: {total_goals}\n"
        f"Goals by team:\n{team_totals.to_string(index=False)}"
    )


def get_relevant_data(question):
    import re

    q = question.lower()
    data = matches.copy()

    # Support both original and standardized column names.
    home_col = (
        "home_team_name" if "home_team_name" in data.columns
        else "home_team"
    )
    away_col = (
        "away_team_name" if "away_team_name" in data.columns
        else "away_team"
    )
    year_col = "year" if "year" in data.columns else "Year"
    round_col = (
        "stage_name" if "stage_name" in data.columns
        else "Round"
    )

    # Identify teams mentioned in the question.
    teams = set(
        data[home_col].dropna().astype(str).tolist()
        + data[away_col].dropna().astype(str).tolist()
    )
    mentioned = [team for team in teams if team.lower() in q]

    if mentioned:
        data = data[
            data[home_col].isin(mentioned)
            | data[away_col].isin(mentioned)
        ]

    # Filter by year when specified.
    years = [
        int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", q)
    ]

    if years:
        data = data[
            pd.to_numeric(data[year_col], errors="coerce").isin(years)
        ]

    # Filter by tournament round when requested.
    round_names = data[round_col].fillna("").astype(str)

    if any(term in q for term in ["semi-final", "semi final", "semifinal"]):
        data = data[round_names.str.contains("semi", case=False)]
    elif "final" in q:
        data = data[round_names.str.fullmatch("Final", case=False)]

    # Select whichever equivalent columns exist in the combined data.
    possible_columns = [
        year_col, "date", "Date", round_col,
        home_col, away_col,
        "home_score", "away_score", "home_xg", "away_xg",
        "Score", "host", "Host", "stadium_name", "Venue",
        "city", "Attendance", "status",
    ]

    columns = list(dict.fromkeys(
        col for col in possible_columns if col in data.columns
    ))

    # Return useful context even when no matching rows exist.
    if data.empty:
        return "No matching records found in the available datasets.", mentioned

    return data[columns].head(50).to_string(index=False), mentioned




def answer_question(question):
    import re

    # Retrieve relevant match records.
    data_context, mentioned_teams = get_relevant_data(question)

    # Build an accurate inventory from the complete loaded dataset.
    year_col = "year" if "year" in matches.columns else "Year"

    available_years = sorted(
        pd.to_numeric(matches[year_col], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    home_col = (
        "home_team_name" if "home_team_name" in matches.columns
        else "home_team"
    )
    away_col = (
        "away_team_name" if "away_team_name" in matches.columns
        else "away_team"
    )

    # Determine which year the user is asking about.
    requested_years = [
        int(year)
        for year in re.findall(r"\b(?:19|20)\d{2}\b", question)
    ]

    # Provide reliable totals for questions asking about goals.
    goal_summary = ""
    q = question.lower()

    if "goal" in q and requested_years:
        selected = matches[
            pd.to_numeric(matches[year_col], errors="coerce").isin(
                requested_years
            )
        ].copy()

        if {"home_score", "away_score"}.issubset(selected.columns):
            selected["home_score"] = pd.to_numeric(
                selected["home_score"], errors="coerce"
            )
            selected["away_score"] = pd.to_numeric(
                selected["away_score"], errors="coerce"
            )

            completed = selected.dropna(
                subset=["home_score", "away_score"]
            )

            total = int(
                completed["home_score"].sum()
                + completed["away_score"].sum()
            )

            goal_summary = (
                f"\nVerified score calculation for {requested_years}: "
                f"{total} goals across {len(completed)} matches "
                f"with both scores recorded."
            )

    system_prompt = """
You are FIFA World Cup Insight AI, a careful football data analyst.

The application provides historical World Cup match data and a
separate 2026 dataset. Answer using the supplied records.

Rules:
- The complete dataset inventory tells you which years are present.
- The match context contains records selected for the user's question.
- Never claim the complete dataset contains only the years shown in
  the selected match context.
- Never invent match results, scores, statistics, or venues.
- Use supplied verified calculations for numerical answers.
- If some matches have missing scores, explain that totals cover only
  matches with both scores recorded.
- A known tournament winner is not proof that every match record
  from that tournament is available.
- If the supplied information cannot establish an answer, say so.
- Distinguish verified dataset facts from general football knowledge.
- Keep answers clear and concise.
"""

    user_prompt = f"""
User question:
{question}

Complete dataset inventory:
Available years: {available_years}

Teams identified in the question:
{mentioned_teams}

Retrieved match records:
{data_context}

{goal_summary}

Answer the question using the records and calculations above.
If the user asks about a year, do not confuse the selected records
with the full dataset.
"""

    client = get_client()

    response = client.chat.completions.create(
        model="databricks-gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return (
        response.choices[0].message.content
        or "I couldn't generate a response."
    )


st.title("⚽ FIFA World Cup Insight AI")


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("＋ New chat"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about a match or team...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing match data..."):
            try:
                answer = answer_question(question)
                st.markdown(answer)
            except Exception as exc:
                answer = (
                    "I couldn't generate a response. "
                    f"Please check the Databricks connection. Details: {exc}"
                )
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
