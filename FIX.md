# CookMatch — Bug Fixes & Enhancements Log

This document records the issues identified, root-cause analyses, and solutions implemented in the **CookMatch** repository.

---

## 📌 1. Test Import Error (`ModuleNotFoundError: No module named 'src'`)

* **Issue:** Running `pytest tests/` failed with `ModuleNotFoundError: No module named 'src'`.
* **Root Cause:** Python did not automatically append the repository root directory to `sys.path` during test execution.
* **Fix:**
  - Added [`conftest.py`](conftest.py) to dynamically insert `PROJECT_ROOT` into `sys.path`.
  - Added [`pytest.ini`](pytest.ini) configuring `pythonpath = .` and `testpaths = tests`.
* **Verification:** All **69 unit tests** pass cleanly in 0.77s.

---

## 📌 2. Pantry Match Quality & Seasoning Salt Ranking Bug

* **Issue:** Selecting `["tomatoes", "garlic", "onion"]` ranked *"All Purpose Seasoning Salt"* (50% match) above real meal dishes like *"10 Minute Marinara Sauce"*.
* **Root Cause:**
  1. **Singular/Plural Mismatch:** User input `"tomatoes"` failed substring matching against dataset string `"tomato"`.
  2. **Ratio Bias:** Old formula `(matched / len(recipe))` gave 2-ingredient spice rubs high ratios (1/2 = 50%), while penalizing real dishes with 8+ ingredients.
  3. **Ignoring User Pantry Utilization:** Did not account for how many of the user's pantry items were actually used.
* **Fix:** Updated [`src/recommend/content_signals.py`](src/recommend/content_signals.py):
  - Implemented `_stem_text()` to normalize plurals (`"tomatoes"` $\rightarrow$ `"tomato"`, `"potatoes"` $\rightarrow$ `"potato"`, `"onions"` $\rightarrow$ `"onion"`).
  - Replaced ratio formula with **Hybrid Pantry Scoring**:
    $$
    \text{Pantry Score} = 0.70 \times \left(\frac{\text{Pantry Items Used}}{\text{Total User Pantry}}\right) + 0.30 \times \left(\frac{\text{Matched Recipe Ingredients}}{\text{Total Recipe Ingredients}}\right)
    $$
* **Verification:** Real meals (e.g. *Marinara Sauce*) matching all 3 pantry items jump to **80%+ match**, demoting spice rubs to the bottom.

---

## 📌 3. Interactive Web Demo UI ([`app.py`](app.py))

* **Enhancement:** Created a full Streamlit web application (`app.py`) for live demonstration:
  - **Stage 1 Safety Sidebar:** Controls for Dietary Preferences (`Vegetarian`, `Vegan`), Allergen Exclusions (`Nuts`, `Dairy`, `Gluten`), and Stage 1 Funnel Statistics.
  - **User Profile ID Selector:** Select logged-in user profiles to test Stage 2 SVD Matrix Factorization vs. `🆕 New / Guest User (Cold-Start)` for Popularity Fallback.
  - **User's Past Ratings Expander:** Displays the historical star ratings of the selected user.
  - **4 Query Mode Tabs:** Pantry Mode, Time Budget Mode, Meal Intent Mode, and Combined Mode.
  - **Recipe Recommendation Cards:** Displays prep time badge, score, match tag, and natural language explanation boxes.

---

## 📌 4. UI Fading/Dimming & Live Progress Indicators

* **Issue:** Streamlit's default behavior dimmed/faded the screen (`opacity: 0.4`) during script reruns, making the UI feel frozen.
* **Fix:**
  - Added custom CSS override rules in `app.py` (`[data-test-script-state="running"]`) to keep the web page crisp and 100% visible during loading.
  - Added explicit live buffering status spinners (`st.spinner()`) during Stage 1 funnel calculation and Stage 2/3 re-ranking.
  - Added state-tracking warnings (`st.session_state`) informing the user whenever Diet, Allergens, or User ID changes.

---

## 📌 5. Multiselect Default Value Exception (`StreamlitAPIException`)

* **Issue:** `st.multiselect` crashed with `StreamlitAPIException` if hardcoded default ingredients (e.g. `"tomato"`) were missing from the top 200 extracted dataset vocabulary.
* **Fix:** Dynamically filtered all default multiselect values using `default_items = [i for i in [...] if i in common_pantry]`.

---

## 📌 6. Combined Mode Category Fallback, Hard Time Cap & Weight Rebalancing

* **Issue:**
  1. In Tab 4 (Combined Mode), the `Meal Category` selectbox defaulted to `"dessert"` (the 1st category item), forcing non-dessert queries to include `"matches dessert"` and surfacing desserts like *"almost apple pie"*.
  2. Setting `Max Minutes: 45` in Combined Mode allowed recipes over 45 minutes (e.g. 50-minute pie) because `"hard_time_cap"` was set to `False`.
  3. Popularity weight (45%) in Combined Mode drowned out high pantry matches (30%), causing a 25% pantry match recipe to rank above an 87% pantry match dish (*10 Minute Marinara Sauce*).
* **Fix:**
  - Updated [`app.py`](app.py) Tab 4 to add `None` (`"Any Category (No preference)"`) as default option for `Meal Category`.
  - Updated [`config/query_modes.py`](config/query_modes.py) to enable `"hard_time_cap": True` for Combined Mode and rebalanced weights:
    $$
    \text{Combined Score} = 0.45 \times \text{Pantry} + 0.25 \times \text{CF/Popularity} + 0.20 \times \text{Time} + 0.10 \times \text{Intent}
    $$
  - Updated [`src/recommend/stage3.py`](src/recommend/stage3.py) to enforce hard time cap filtering when `hard_time_cap` config is enabled or mode is `combined`.
* **Verification:**
  - 50-minute recipes are strictly excluded when `Max Minutes: 45` is set.
  - *10 Minute Marinara Sauce* (87% pantry match) takes #1 spot over generic popular recipes.

---

## 📌 7. Recipe Ingredients Display in Recommendation Cards

* **Enhancement:** Users requested seeing the full ingredient list for each recommended recipe card in the Web UI.
* **Implementation:**
  - Added `ingredients: list[str]` field to `Stage3Recommendation` dataclass in [`src/recommend/stage3.py`](src/recommend/stage3.py).
  - Attached raw/normalized ingredient lists during Stage 3 candidate re-ranking.
  - Added an interactive **`📝 Ingredients (N)`** expander box under each recipe card in [`app.py`](app.py).

---

## 📌 8. Updated Dependencies ([`requirements.txt`](requirements.txt))

* **Update:** Added `streamlit>=1.30.0` to `requirements.txt`.
