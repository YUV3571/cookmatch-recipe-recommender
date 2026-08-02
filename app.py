"""CookMatch Streamlit Demo Application.

Interactive web UI for safety-constrained hybrid recipe recommendation engine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from config.intents import MEAL_INTENT_TAGS
from config.query_modes import QUERY_MODES
from src.data.loader import (
    clean_interactions,
    clean_recipes,
    load_interaction_split,
    load_recipes,
)
from src.models.session_context import SessionContext
from src.models.user_profile import UserProfile
from src.recommend.stage3 import Stage3Recommender

# Page Configuration
st.set_page_config(
    page_title="CookMatch — Recipe Recommender",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look
st.markdown(
    """
    <style>
    /* Prevent Streamlit from fading/dimming/freezing the screen during loading */
    div[data-test-script-state="running"],
    .stApp[data-test-script-state="running"],
    [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .recipe-card {
        background-color: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #313244;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-time { background-color: #313244; color: #cdd6f4; }
    .badge-pantry { background-color: #181825; color: #a6e3a1; border: 1px solid #a6e3a1; }
    .badge-intent { background-color: #181825; color: #f9e2af; border: 1px solid #f9e2af; }
    .badge-score { background-color: #89b4fa; color: #11111b; }
    .why-box {
        background-color: #181825;
        border-left: 4px solid #89b4fa;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 0.95rem;
        color: #cdd6f4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading dataset and training CookMatch recommendation model...")
def get_trained_recommender(recipe_limit: int = 5000) -> tuple[Stage3Recommender, pd.DataFrame, list[str]]:
    """Load recipes, interactions, and fit Stage3Recommender (cached)."""
    recipes_raw = load_recipes(
        nrows=recipe_limit,
        columns=["id", "name", "ingredients", "minutes", "tags"],
    )
    recipes = clean_recipes(recipes_raw)
    train_raw = load_interaction_split("train")
    train = clean_interactions(train_raw)

    recommender = Stage3Recommender().fit(recipes, train)

    # Dynamically extract top 200 ingredients from recipe dataset
    from collections import Counter
    import ast

    all_ingr = []
    for ingr_val in recipes["ingredients"]:
        if isinstance(ingr_val, list):
            all_ingr.extend([str(i).strip().lower() for i in ingr_val])
        elif isinstance(ingr_val, str):
            try:
                parsed = ast.literal_eval(ingr_val)
                if isinstance(parsed, list):
                    all_ingr.extend([str(i).strip().lower() for i in parsed])
            except Exception:
                all_ingr.extend([i.strip().lower() for i in ingr_val.split(",")])

    common_pantry = sorted([item for item, _ in Counter(all_ingr).most_common(200) if len(item) > 1])

    return recommender, train, common_pantry


def main():
    st.markdown('<div class="main-header">🍳 CookMatch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Safety-Constrained Hybrid Recipe Recommender</div>',
        unsafe_allow_html=True,
    )

    # Load data & engine
    with st.spinner("Initializing CookMatch engine..."):
        recommender, train, common_pantry = get_trained_recommender(recipe_limit=3000)

    # ==================== SIDEBAR: USER PROFILE ====================
    st.sidebar.header("🛡️ User Safety & Profile")
    st.sidebar.markdown("Stage 1 hard-constraint filters guarantee recipes are 100% safe.")

    user_diet = st.sidebar.selectbox(
        "Dietary Preference",
        options=[None, "vegetarian", "vegan"],
        format_func=lambda x: "No restrictions" if x is None else x.capitalize(),
    )

    user_allergens = st.sidebar.multiselect(
        "Allergens / Intolerances to Block",
        options=["nuts", "dairy", "gluten"],
        default=[],
        help="Select any ingredients to strictly exclude from recommendations.",
    )

    profile = UserProfile(diet=user_diet, allergens=user_allergens)

    st.sidebar.divider()
    available_users = train["user_id"].unique()[:20]
    user_options = [None] + [int(u) for u in available_users]
    known_user = st.sidebar.selectbox(
        "Select User Profile ID",
        options=user_options,
        index=0,
        format_func=lambda x: "🆕 New / Guest User (Cold-Start)" if x is None else f"👤 User #{x}",
        help="Simulates a new unauthenticated guest user vs a logged-in user with rating history.",
    )

    # Display selected user's past rating history
    if known_user is not None:
        user_history = train[train["user_id"] == known_user].head(5)
        with st.sidebar.expander(f"📜 User #{known_user}'s Past Ratings", expanded=True):
            for _, row in user_history.iterrows():
                rid = int(row["recipe_id"])
                rating = float(row["rating"])
                rname = recommender.stage2.recipe_names_.get(rid, f"Recipe #{rid}")
                st.markdown(f"⭐ **{rating:.0f}/5** — *{rname}*")
    else:
        with st.sidebar.expander("📜 User's Past Ratings", expanded=False):
            st.info("🆕 Guest User — No prior rating history in database.")

    # Funnel Stats calculation with live buffering message
    with st.spinner("⏳ Stage 1 Safety Filter: Applying dietary rules & scanning catalog for allergens..."):
        funnel = recommender.summarize_stage1_funnel(profile)

    with st.sidebar.expander("📊 Stage 1 Funnel Stats", expanded=False):
        st.write(f"**Total Catalog:** {funnel['total_recipes']} recipes")
        st.write(f"**After Allergen Filter:** {funnel['after_allergen_filter']}")
        st.write(f"**Final Safe Pool:** {funnel['safe_recipes']} ({funnel['safe_pct']}%)")
        st.write(f"**Blocked by Safety:** {funnel['blocked_recipes']}")

    # ==================== MAIN AREA: ACTIVE STATUS & QUERY MODES ====================
    # Detect Profile & User Changes to display explicit notification message
    if "last_diet" not in st.session_state:
        st.session_state["last_diet"] = user_diet
    if "last_allergens" not in st.session_state:
        st.session_state["last_allergens"] = user_allergens
    if "last_user" not in st.session_state:
        st.session_state["last_user"] = known_user

    change_logs = []
    if st.session_state["last_diet"] != user_diet:
        if user_diet:
            change_logs.append(
                f"🌱 **Dietary Preference Changed to '{user_diet.capitalize()}'** — Stage 1 safety filter updated! "
                f"**{funnel['blocked_recipes']} recipes blocked** by safety. "
                f"**{funnel['safe_recipes']} safe recipes remaining** in pool."
            )
        else:
            change_logs.append("🌱 **Dietary Preference Cleared** — All diet restrictions removed.")
        st.session_state["last_diet"] = user_diet

    if st.session_state["last_allergens"] != user_allergens:
        if user_allergens:
            change_logs.append(
                f"🚫 **Allergens Updated to: {', '.join([a.capitalize() for a in user_allergens])}** — "
                f"Strictly excluding any recipes containing these ingredients."
            )
        else:
            change_logs.append("🚫 **Allergen Exclusions Cleared** — No allergen restrictions active.")
        st.session_state["last_allergens"] = user_allergens

    if st.session_state["last_user"] != known_user:
        if known_user is not None:
            change_logs.append(
                f"👤 **User Switched to User #{known_user}** — "
                f"Collaborative Filtering (Stage 2 SVD) will now personalize recommendations based on User #{known_user}'s rating history."
            )
        else:
            change_logs.append(
                "🆕 **User Switched to Guest (Cold-Start)** — "
                "No rating history available. Engine will use Popularity Fallback for unauthenticated guest."
            )
        st.session_state["last_user"] = known_user

    # Render explicit change notifications if profile was changed
    if change_logs:
        for log in change_logs:
            st.warning(log)
        st.caption("💡 *Click any 'Find Recipes' button below to refresh recommendations for your updated profile.*")

    # Active Safety Constraints Banner
    status_parts = []
    if user_diet:
        status_parts.append(f"🌱 **Diet:** {user_diet.capitalize()}")
    else:
        status_parts.append("🌱 **Diet:** No restrictions")

    if user_allergens:
        status_parts.append(f"🚫 **Blocked Allergens:** {', '.join([a.capitalize() for a in user_allergens])}")

    user_label = f"👤 **User:** #{known_user}" if known_user is not None else "🆕 **User:** Guest (Cold-Start)"
    status_parts.append(user_label)
    status_parts.append(f"🛡️ **Safe Pool:** {funnel['safe_recipes']}/{funnel['total_recipes']} recipes ({funnel['safe_pct']}%)")

    st.info(" | ".join(status_parts))

    st.markdown("### 🔍 Select Query Mode")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🛒 Pantry Mode",
            "⏱️ Time Budget Mode",
            "🍽️ Meal Intent Mode",
            "🔀 Combined Mode",
        ]
    )

    recommendations = []
    selected_mode = "combined"
    context = SessionContext()

    # --- TAB 1: PANTRY MODE ---
    with tab1:
        st.markdown(
            "#### What can I cook with what I have?\n"
            "Ranks recipes based on ingredient overlap with your available pantry items."
        )
        default_items = [i for i in ["tomatoes", "tomato", "garlic", "pasta", "olive oil", "onion"] if i in common_pantry]
        if not default_items and common_pantry:
            default_items = common_pantry[:3]

        pantry_items = st.multiselect(
            "Select or search ingredients in your pantry:",
            options=common_pantry,
            default=default_items,
            key="pantry_tab_items",
            help="Type any ingredient name to search across 200+ dataset ingredients.",
        )

        if st.button("Find Recipes by Pantry", type="primary", key="btn_pantry"):
            with st.spinner("🍳 Filtering safe recipes & matching pantry ingredients..."):
                context = SessionContext(pantry=pantry_items)
                mode_cfg = QUERY_MODES["pantry"]
                recommendations = recommender.recommend(
                    profile,
                    context,
                    user_id=known_user,
                    top_n=10,
                    weights_override=mode_cfg["weights"],
                    mode="pantry",
                )
                selected_mode = "pantry"
            st.toast("✅ Pantry recommendations generated!", icon="🛒")

    # --- TAB 2: TIME MODE ---
    with tab2:
        st.markdown(
            "#### Recipes within my time budget\n"
            "Applies a hard cooking time cap and scores recipes fitting your available window."
        )
        max_mins = st.slider(
            "Maximum prep & cook time (minutes):",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            key="time_tab_slider",
        )

        if st.button("Find Recipes by Time", type="primary", key="btn_time"):
            with st.spinner(f"⏱️ Applying {max_mins}-minute hard time cap & scoring..."):
                context = SessionContext(max_minutes=max_mins)
                mode_cfg = QUERY_MODES["time"]
                recommendations = recommender.recommend(
                    profile,
                    context,
                    user_id=known_user,
                    top_n=10,
                    weights_override=mode_cfg["weights"],
                    mode="time",
                )
                selected_mode = "time"
            st.toast("✅ Time-budget recommendations generated!", icon="⏱️")

    # --- TAB 3: INTENT MODE ---
    with tab3:
        st.markdown(
            "#### Recipes matching my meal goal\n"
            "Filters and prioritizes recipes matching specific meal categories."
        )
        valid_intents = list(MEAL_INTENT_TAGS.keys())
        intent_choice = st.selectbox(
            "Select meal category:",
            options=valid_intents,
            index=valid_intents.index("main") if "main" in valid_intents else 0,
            key="intent_tab_choice",
        )

        if st.button("Find Recipes by Intent", type="primary", key="btn_intent"):
            with st.spinner(f"🍽️ Searching for '{intent_choice}' meal category..."):
                context = SessionContext(meal_intent=intent_choice)
                mode_cfg = QUERY_MODES["intent"]
                recommendations = recommender.recommend(
                    profile,
                    context,
                    user_id=known_user,
                    top_n=10,
                    weights_override=mode_cfg["weights"],
                    mode="intent",
                )
                selected_mode = "intent"
            st.toast("✅ Meal intent recommendations generated!", icon="🍽️")

    # --- TAB 4: COMBINED MODE ---
    with tab4:
        st.markdown(
            "#### Personalized blend of all signals\n"
            "Combines Collaborative Filtering (45%) + Pantry Match (30%) + Time (15%) + Intent (10%)."
        )
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            c_pantry = st.multiselect(
                "Pantry Items:",
                options=common_pantry,
                default=default_items,
                key="combined_pantry",
            )
        with col_c2:
            c_time = st.slider("Max Minutes:", 5, 120, 45, key="combined_time")
        with col_c3:
            intent_options = [None] + valid_intents
            c_intent = st.selectbox(
                "Meal Category (Optional):",
                options=intent_options,
                index=0,
                format_func=lambda x: "Any Category (No preference)" if x is None else x.capitalize(),
                key="combined_intent",
            )

        if st.button("Generate Combined Recommendations", type="primary", key="btn_combined"):
            with st.spinner("🔀 Running 3-Stage Cascade & blending all signals..."):
                context = SessionContext(pantry=c_pantry, max_minutes=c_time, meal_intent=c_intent)
                recommendations = recommender.recommend(
                    profile,
                    context,
                    user_id=known_user,
                    top_n=10,
                    mode="combined",
                )
                selected_mode = "combined"
            st.toast("✅ Combined recommendations generated!", icon="✨")

    # ==================== RESULTS DISPLAY ====================
    if recommendations:
        st.divider()
        st.subheader(f"✨ Top Recommendations ({selected_mode.capitalize()} Mode)")

        for i, rec in enumerate(recommendations, 1):
            with st.container():
                col_title, col_score = st.columns([4, 1])

                with col_title:
                    st.markdown(f"#### {i}. {rec.name}")

                with col_score:
                    st.markdown(
                        f"**Score:** `{rec.final_score:.2f}`",
                        help="Weighted blend score (0.0 to 1.0)",
                    )

                # Badges row
                badges_html = []
                if rec.minutes:
                    badges_html.append(f'<span class="badge badge-time">⏱️ {rec.minutes} mins</span>')
                if rec.pantry_score > 0:
                    badges_html.append(
                        f'<span class="badge badge-pantry">🛒 {rec.pantry_score:.0%} pantry match</span>'
                    )
                if rec.intent_score > 0:
                    badges_html.append('<span class="badge badge-intent">🍽️ Category Match</span>')

                if badges_html:
                    st.markdown("".join(badges_html), unsafe_allow_html=True)

                # Explanation Box
                st.markdown(
                    f'<div class="why-box">💡 <b>Why:</b> {rec.explanation}</div>',
                    unsafe_allow_html=True,
                )

                # Ingredients Expander
                if hasattr(rec, "ingredients") and rec.ingredients:
                    formatted_ingr = [i.title() for i in rec.ingredients]
                    with st.expander(f"📝 Ingredients ({len(formatted_ingr)})", expanded=False):
                        st.write(", ".join(formatted_ingr))

                st.write("")
    else:
        st.info("👆 Click any 'Find Recipes' button above to generate recommendations!")


if __name__ == "__main__":
    main()
# Cache refresh trigger

