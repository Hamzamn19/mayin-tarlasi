# Academic Machine Learning Diagnostic Dashboard

## Core Intent
Refactor the existing "Mine Explainer" dashboard from a sci-fi HUD aesthetic into a clean, authoritative, and highly readable academic diagnostic tool. The primary objective is to teach machine learning concepts to an academic committee through visual transparency and structural clarity.

## Visual Aesthetic & Priority
*   **Vibe:** Analytical, sophisticated, and "Medical-Grade". Inspired by professional diagnostic equipment and scientific publishing (LaTeX style).
*   **Color Palette:** Clean white backgrounds (#FFFFFF), light gray surfaces (#F8FAFC), with high-contrast slate text (#1E293B). Accent colors (Royal Blue, Crimson Red, Forest Green) are used only for functional data markers.
*   **Typography:** Professional sans-serif for numerical data and interface labels. Refined serif fonts for section headings to provide an authoritative tone. Minimum font size 14px for body text.
*   **Spacing:** Generous white space between all elements. No decorative borders, scan lines, or glitch effects.

## Functional Components

### 1. Stage 6: Logistic Regression (The Linear Scale)
*   **Step 1: Weighted Sum Calculation:** Present a clean table showing input feature values multiplied by their respective model weights.
*   **Step 2: Sigmoid Mapping:** Display a large, clear Sigmoid curve graph.
*   **Interactivity:** Plot a distinct dot on the curve representing the current calculated score to show how it maps to a probability (0.0 to 1.0).
*   **Tone:** Use plain academic language (e.g., "Step 1: Weighted Sum Calculation" instead of "LOG-ODDS ARCHITECTURE").

### 2. Stage 7: Random Forest (Consensus Matrix)
*   **Visualization:** Replace the abstract grid with 20 small "Tree Cards". Each card is colored either Red (Landmine) or Green (Background) based on its specific vote.
*   **Aggregation:** Feature a prominent vote counter: "74 / 100 Trees Voted: LANDMINE".
*   **Detail View:** Show ONE expanded decision tree logic path with 3 clear levels (IF -> THEN) highlighted, demonstrating the internal logic of an individual estimator.

### 3. Stage 8: Integrated Verdict
*   **Consensus View:** Show two probability bars side-by-side for comparison: "Logistic Regression Output" vs. "Random Forest Output".
*   **Mathematical Transparency:** Explicitly display the ensemble formula: `(LR_Prob + RF_Prob) / 2 = Final_Score`.
*   **Final Outcome:** Represent the decision as a simple, high-contrast badge: `[ LANDMINE ]` or `[ SAFE ]`. Remove all narrative "Drama" or military protocols.

## Technical Constraints
*   Ensure full responsiveness for desktop/tablet views.
*   Utilize standard scientific visualization styles for all graphs and charts.
*   Maintain the hybrid pipeline logic (Stages 1-8) while stripping all non-academic visual noise.
