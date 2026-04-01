"""
Human-readable labels and help text for loan-form UI.

Internal / pipeline column names stay unchanged; only UI copy uses these strings.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

# Form section keys (order = order on screen)
SECTION_ORDER: tuple[str, ...] = (
    "household",
    "assets",
    "loan",
    "money",
    "work",
    "area",
    "credit",
    "contact",
    "misc",
)

SECTION_TITLES: dict[str, str] = {
    "household": "You & your household",
    "assets": "Vehicles & home",
    "loan": "This loan",
    "money": "Income & amounts",
    "work": "Work & employer",
    "area": "Area & living situation",
    "credit": "Credit history & scores",
    "contact": "Phone & how we reach you",
    "misc": "Other details",
}

SECTION_CAPTIONS: dict[str, str] = {
    "household": "Basic facts about the applicant and family.",
    "assets": "What the applicant owns — used as background for the estimate.",
    "loan": "Size and repayment for the loan you’re assessing.",
    "money": "Figures in the same units your team normally uses (e.g. monthly).",
    "work": "Job and employer — pick the closest match from the lists.",
    "area": "Region and housing — helps the model match similar cases.",
    "credit": "Scores and checks from credit bureaus, if you have them.",
    "contact": "Whether we have reliable phone numbers on file.",
    "misc": "Anything else the model expects.",
}

# Map each form column to a section (pipeline column name → section key)
COLUMN_SECTION: dict[str, str] = {
    "Age_Days": "household",
    "Child_Count": "household",
    "Client_Family_Members": "household",
    "Accompany_Client": "household",
    "Client_Marital_Status": "household",
    "Client_Gender": "household",
    "Car_Owned": "assets",
    "Bike_Owned": "assets",
    "House_Own": "assets",
    "Own_House_Age": "assets",
    "Credit_Amount": "loan",
    "Loan_Annuity": "loan",
    "Loan_Contract_Type": "loan",
    "Client_Income": "money",
    "Client_Income_Type": "money",
    "Population_Region_Relative": "area",
    "Client_Housing_Type": "area",
    "Cleint_City_Rating": "area",
    "Client_Education": "work",
    "Client_Occupation": "work",
    "Employed_Days": "work",
    "Type_Organization": "work",
    "Score_Source_1": "credit",
    "Score_Source_2": "credit",
    "Score_Source_3": "credit",
    "Social_Circle_Default": "credit",
    "Phone_Change": "credit",
    "Credit_Bureau": "credit",
    "Mobile_Tag": "contact",
    "Homephone_Tag": "contact",
    "Workphone_Working": "contact",
    "Client_Permanent_Match_Tag": "contact",
    "Client_Contact_Work_Tag": "contact",
    "Registration_Days": "area",
    "ID_Days": "household",
    "Active_Loan": "loan",
}

# Binary 0/1 fields — show Yes/No instead of a numeric stepper.
YES_NO_NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        "Car_Owned",
        "Bike_Owned",
        "Active_Loan",
        "Mobile_Tag",
        "Homephone_Tag",
        "Workphone_Working",
    }
)

# Sensible bounds & steps for number inputs (dataset-specific; adjust if your data differs).
NUMERIC_INPUT_BOUNDS: dict[str, dict[str, Any]] = {
    "Score_Source_1": {"min_value": 0.0, "max_value": 1.0, "step": 0.0001},
    "Score_Source_3": {"min_value": 0.0, "max_value": 1.0, "step": 0.0001},
    "Score_Source_2": {"min_value": 0.0, "max_value": 100.0, "step": 0.01},
    "Social_Circle_Default": {"min_value": 0.0, "max_value": 1.0, "step": 0.0001},
    "Phone_Change": {"min_value": 0.0, "step": 1.0},
    "Credit_Bureau": {"min_value": 0.0, "step": 1.0},
    "Child_Count": {"min_value": 0.0, "step": 1.0},
    "Client_Family_Members": {"min_value": 0.0, "step": 1.0},
    "Cleint_City_Rating": {"min_value": 1.0, "max_value": 3.0, "step": 1.0},
}

# Technical column name -> short label shown in the UI (plain language)
FEATURE_LABELS: dict[str, str] = {
    "ID": "Application reference",
    "Client_Income": "Income",
    "Car_Owned": "Has a car",
    "Bike_Owned": "Has a motorbike",
    "Active_Loan": "Already repaying another loan",
    "House_Own": "Owns their home",
    "Child_Count": "Children",
    "Credit_Amount": "Total loan amount",
    "Loan_Annuity": "Installment payment (per period)",
    "Accompany_Client": "Who applied with them",
    "Client_Income_Type": "How they earn income",
    "Client_Education": "Education",
    "Client_Marital_Status": "Marital status",
    "Client_Gender": "Gender",
    "Loan_Contract_Type": "Type of loan",
    "Client_Housing_Type": "Living situation",
    "Population_Region_Relative": "Regional population index",
    "Age_Days": "Age (in days)",
    "Employed_Days": "Time in current work (days)",
    "Registration_Days": "How long since registration (days)",
    "ID_Days": "How long since ID document (days)",
    "Own_House_Age": "Age of the home (years)",
    "Mobile_Tag": "Mobile number provided",
    "Homephone_Tag": "Home landline on file",
    "Workphone_Working": "Work number works",
    "Client_Occupation": "Job type",
    "Client_Family_Members": "People in the household",
    "Cleint_City_Rating": "City tier (1 = best)",
    "Application_Process_Day": "Application weekday",
    "Application_Process_Hour": "Application time (hour)",
    "Client_Permanent_Match_Tag": "Address matches records",
    "Client_Contact_Work_Tag": "Reachable at work",
    "Type_Organization": "Employer sector",
    "Score_Source_1": "External credit score 1",
    "Score_Source_2": "External credit score 2",
    "Score_Source_3": "External credit score 3",
    "Social_Circle_Default": "Contacts who defaulted (share)",
    "Phone_Change": "Days since phone number changed",
    "Credit_Bureau": "Recent credit checks",
}

# Tooltip text — short, plain language
FIELD_HELP: dict[str, str] = {
    "ID": "Reference number for this application.",
    "Client_Income": "Use the same units your team always uses (e.g. monthly). Any sensible positive number.",
    "Car_Owned": "Do they own a car?",
    "Bike_Owned": "Do they own a motorbike or scooter?",
    "Active_Loan": "Are they already paying back another loan?",
    "House_Own": "Do they own the home they live in? Enter the value your process uses, or leave the default if unsure.",
    "Child_Count": "How many dependent children (0, 1, 2, …).",
    "Credit_Amount": "Full amount of this loan, in your usual currency and units.",
    "Loan_Annuity": "Each scheduled payment (installment), same units as the loan amount.",
    "Accompany_Client": "Who came to the branch or applied with them — choose the closest option.",
    "Client_Income_Type": "Main source of income (employed, pension, business, etc.).",
    "Client_Education": "Highest completed level — pick from the list.",
    "Client_Marital_Status": "As stated on the application.",
    "Client_Gender": "As on the application or ID.",
    "Loan_Contract_Type": "Product type (e.g. cash loan vs revolving) — match your product list.",
    "Client_Housing_Type": "Renting, own home, with family, etc.",
    "Population_Region_Relative": "A number describing how populated their area is — use the value from your system if you have it.",
    "Age_Days": "Age expressed in days (multiply years × 365 if you only have years).",
    "Employed_Days": "Days in current job. Unusual negative values may mean “not employed” or special codes in your data.",
    "Registration_Days": "Days since an address or client registration — as recorded in your files.",
    "ID_Days": "Days since their ID was issued, if you track it.",
    "Own_House_Age": "How old the property is in years.",
    "Mobile_Tag": "Did they give a mobile number?",
    "Homephone_Tag": "Is there a home landline on file?",
    "Workphone_Working": "Can you reach them on a work number?",
    "Client_Occupation": "Job category — choose the closest match.",
    "Client_Family_Members": "Total people in the household (whole number).",
    "Cleint_City_Rating": "Your internal city grade: 1 = strongest, 3 = weakest (as in your data).",
    "Application_Process_Day": "Filled in automatically from typical timing if you don’t supply it.",
    "Application_Process_Hour": "Filled in automatically if omitted.",
    "Client_Permanent_Match_Tag": "Does their address match official records?",
    "Client_Contact_Work_Tag": "Can they be contacted at work?",
    "Type_Organization": "What kind of company employs them (government, private, self-employed, etc.).",
    "Score_Source_1": "First bureau score — often a decimal between 0 and 1; higher usually means safer credit.",
    "Score_Source_2": "Second bureau score — can be on a larger scale (e.g. up to 100). Enter what appears on the report.",
    "Score_Source_3": "Third bureau score — often between 0 and 1 like score 1.",
    "Social_Circle_Default": "Rough share of their contacts who have defaulted: 0 = none, 1 = all (if your data provides this).",
    "Phone_Change": "Whole days since they last changed their phone number.",
    "Credit_Bureau": "How many times lenders checked their credit recently (0, 1, 2, …).",
}


def display_label(column: str) -> str:
    """Return a short label for UI widgets; fall back to a readable title."""
    if column in FEATURE_LABELS:
        return FEATURE_LABELS[column]
    return column.replace("_", " ").strip().title()


def field_help(column: str) -> str:
    """Tooltip text: what to enter, units, and 0/1 meaning where relevant."""
    base = FIELD_HELP.get(column)
    if base:
        return base
    return "Enter a value in the same format and units as your loan dataset."


def field_help_cat_numeric() -> str:
    """Shared help for continuous fields encoded as categorical in the model."""
    return "Use the same units as the rest of your forms. Odd values may be treated as unknown."


def _section_priority(section_key: str) -> int:
    try:
        return SECTION_ORDER.index(section_key)
    except ValueError:
        return len(SECTION_ORDER)


def ordered_form_fields(
    vis_num: list[str],
    vis_free_cat: list[str],
    vis_discrete: list[str],
) -> list[tuple[str, str, str]]:
    """(section_key, field_kind, column) in friendly section order. field_kind: num | free_cat | discrete."""
    rows: list[tuple[int, int, str, str, str]] = []
    n = 0
    for c in vis_num:
        sk = COLUMN_SECTION.get(c, "misc")
        rows.append((_section_priority(sk), n, sk, "num", c))
        n += 1
    for c in vis_free_cat:
        sk = COLUMN_SECTION.get(c, "misc")
        rows.append((_section_priority(sk), n, sk, "free_cat", c))
        n += 1
    for c in vis_discrete:
        sk = COLUMN_SECTION.get(c, "misc")
        rows.append((_section_priority(sk), n, sk, "discrete", c))
        n += 1
    rows.sort(key=lambda x: (x[0], x[1]))
    return [(r[2], r[3], r[4]) for r in rows]


def group_by_section(
    ordered: list[tuple[str, str, str]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """(section_key, [(kind, col), ...]) preserving order within each section."""
    if not ordered:
        return []
    out: list[tuple[str, list[tuple[str, str]]]] = []
    cur = ordered[0][0]
    chunk: list[tuple[str, str]] = []
    for sec, kind, col in ordered:
        if sec != cur:
            out.append((cur, chunk))
            chunk = []
            cur = sec
        chunk.append((kind, col))
    out.append((cur, chunk))
    return out


def yes_no_radio_index(default: Any) -> int:
    """Default index for Yes/No radio: 0 = No, 1 = Yes."""
    try:
        v = float(default)
        if math.isnan(v):
            return 0
        return 1 if v >= 0.5 else 0
    except (TypeError, ValueError):
        return 0


def bounded_number_input_kwargs(col: str, default: Any) -> dict[str, Any]:
    """Kwargs for ``st.number_input`` with optional min/max/step from ``NUMERIC_INPUT_BOUNDS``."""
    try:
        dv = float(default)
        if math.isnan(dv):
            dv = 0.0
    except (TypeError, ValueError):
        dv = 0.0

    if col not in NUMERIC_INPUT_BOUNDS:
        return {"value": dv, "format": "%.6g"}

    b = dict(NUMERIC_INPUT_BOUNDS[col])
    mn = b.get("min_value")
    mx = b.get("max_value")
    if mn is not None:
        dv = max(float(mn), dv)
    if mx is not None:
        dv = min(float(mx), dv)

    fmt = "%.6g"
    if b.get("step") == 1.0 and col in ("Phone_Change", "Credit_Bureau", "Child_Count", "Client_Family_Members"):
        fmt = "%.0f"

    out = {"value": dv, "format": fmt, **b}
    return out
