# ref-ball.site — Implementation Plan

> Public-facing site for the ref-ball umbrella: the per-official dataset hub, the "Attention Load" (cranky-scott-foster) findings, and the "Does Harden Choke?" frozen archive. Target outcome: a visitor lands, picks any named official, and sees their suppressor/amplifier profile in under 30 seconds. Traction-first, not Sloan-first — everything ships open, fast.

---

## 0. Project posture (read this first — it constrains every decision below)

- **Goal:** Visibility and inbound links in the basketball-analytics community, **not** winning SSAC. Open-data, sharp writing, a lookup page that distributed itself.
- **Where the site lives:** `ref-ball/site/` is a new sub-directory of the existing ref-ball repo. ref-ball is already the surviving container (DHC's README says development moved to ref-ball). Don't make a 4th repo.
- **Lowercase the codename publicly.** "cranky-scott-foster" is a directory name only. Public-facing title is **"The Attention Load."** "does-harden-choke" stays as-is — it's a question, not an accusation.
- **The dataset is the moat.** Every structural choice optimizes for one page: the lookup page where the per-official attribution becomes visible. Prose-and-figure pages are the long tail.

---

## 1. Tech stack (fixed, with rationale)

| Component | Choice | Why |
|---|---|---|
| Site framework | **Astro** (static output) | First-load performance (critical for a data-lookup page that must feel instant), zero-JS by default, islands only where needed for the lookup, native Markdown for the DHC/Attention-Load prose pages, easy Vercel deploy. |
| Lookup interactivity | **Pre-generated static JSON + vanilla-JS filter/sort** | 88 officials × 40 players × small per-row payload = a few hundred KB of JSON. No backend. Indexes in Google. The "search an official" pattern is just client-side filter over a JSON file. |
| Charts | **Observable Plot** (lightweight, static-output) for bar/scatter; built-in D3 under the hood, no chart library runtime | Per-official suppressor bars, the crew-prediction scatter (r=0.406), MRT error-rate bars — all simple enough that Plot's declarative API covers them. Avoids pulling in Chart.js/Echarts. |
| Styling | **Plain CSS + one reset;** no Tailwind (overkill for a research site, adds build complexity) | Site is ~5 pages. Hand-written CSS keeps it readable by future maintainers. |
| Deploy | **Vercel** on Harris-owned custom domain (TBD) | Astro + Vercel is a known-good zero-config combo. Domain resolved later by Harris. |
| Build pipeline | **Python pre-build script** runs in ref-ball root, reads parquets, emits `site/src/data/*.json` and per-official/per-player detail JSON | The site never imports parquet at build time. The site only sees JSON. Decouples pandas/pyarrow from the frontend build and keeps the data transform logic next to ref-ball's source of truth. |

### Alternatives considered and rejected
- **Hugo** — easier prose pages but awkward for the data-lookup interactivity; Astro's island architecture is the reason to use it.
- **Next.js** — overkill; we don't need SSR, a server, or React's bundle for a research hub.
- **Live DB (SQLite DuckDB WASM)** — slicker but heavier; the static-data approach is faster, more cacheable, and enough at this dataset size.
- **Tailwind** — wrong tool for a 5-page site that future graders/LLMs will edit.

---

## 1b. Visual design spec

**Design brief:** Basketball-Reference's data density and drill-down UX, rendered through Tavus's modern-retro visual language, in 76ers team colors. The site should feel like a designed research artifact — not a raw stats dump, not a SaaS landing page. Think: a broadsheet sports-analytics publication that happens to have interactive tables.

### Reference sites and what we pull from each

| Reference | What we take | What we don't take |
|---|---|---|
| **Basketball-Reference** | Dense sortable tables, comprehensive drill-down index pages, "click any entity to see its page" navigation model, utilitarian data-first hierarchy | Visual design (too raw/crowded), ad-heavy layout, lack of whitespace |
| **Dunks & Threes** | Player-card format with stat tiles + percentile bars, prediction tables with inline visuals, polished modern SPA feel | Subscription gate, EPM-specific widgets, team logos as primary nav |
| **Cleaning the Glass** | Percentile-based stat interpretation ("is this number good?"), front-office authority tone, clean editorial quality | WordPress feel, paywall-first architecture |
| **Tavus** | Bold display typography, generous whitespace, card-based layouts with subtle shadows, dramatic hero with oversized type, modern-retro aesthetic, section-to-section visual rhythm, "designed artifact" quality | AI/video product framing, dark-mode-first, illustration-heavy sections |

### Color palette — 76ers

| Token | Hex | Usage |
|---|---|---|
| `--color-blue` | `#003DA5` | Primary brand — nav bar, headings, link color, active states, "Suppressor" pill background |
| `--color-red` | `#ED174C` | Accent — "Amplifier" pill background, negative deltas, alert states, hover accents |
| `--color-white` | `#FFFFFF` | Page background, card backgrounds, table cell backgrounds |
| `--color-off-white` | `#F5F5F0` | Section alternating backgrounds (every-other-section shading, à la Tavus), table header row |
| `--color-silver` | `#8E9196` | Secondary text (metadata, footnotes, "n=15 games" context), table borders |
| `--color-navy-dark` | `#00214D` | Footer background, hover states on nav, deep accents |
| `--color-text` | `#1A1A1A` | Body text (near-black, not pure black — Tavus uses this for readability) |

**Rule:** Red is accent-only, never dominant. Blue carries the brand. White dominates the canvas. The palette should read "authoritative sports publication," not "team merch store."

### Typography

| Role | Font | Fallback | Weight | Notes |
|---|---|---|---|---|
| Display / H1 | **Instrument Serif** | Georgia, serif | 400 | Tavus-style modern-retro serif for hero headlines and page titles. Large, dramatic. Free on Google Fonts. |
| H2–H4 | **Inter** | system-ui, sans-serif | 600–700 | Clean, legible, widely available. Section headers, table group headers. |
| Body | **Inter** | system-ui, sans-serif | 400 | Paragraphs, table cells, methodology prose. |
| Mono / data | **JetBrains Mono** | monospace | 400 | Stat values in tables (FTA/36, p-values, deltas). Monospace aligns columns visually. Optional — only use if column alignment matters. |
| Nav + UI | **Inter** | system-ui, sans-serif | 500 | Buttons, nav links, pills, breadcrumbs. |

**Type scale:** Use a modular scale (1.25 ratio). Hero H1 at ~48–56px. H2 at ~32px. Body at 16–18px. Table cells at 14–15px. Generous line-height (1.5 body, 1.2 headings).

### Layout principles (what makes it feel like Tavus, not BBRef)

1. **Generous whitespace.** Tavus uses ~80–120px vertical section padding. Replicate this between major sections (hero → three-movement panel → search). Tables themselves are dense (BBRef-style), but the space *around* them breathes.
2. **Card-based data containers.** Each stat tile, each official profile summary, each project card = a white card on the off-white background with a subtle `box-shadow: 0 1px 3px rgba(0,0,0,0.08)` and `border-radius: 8px`. This is the Tavus move — elevating content into discrete visual objects.
3. **Max content width: 1120px.** Centered. Tables can stretch wider with horizontal scroll on mobile, but the prose and card grid live inside 1120px. This prevents the BBRef "wall of data" feeling.
4. **Alternating section backgrounds.** Hero (white) → three-movement panel (off-white) → search (white) → data section (off-white). Tavus does this to create visual rhythm without borders.
5. **Bold hero, dense interior.** The homepage hero is Tavus-scale (big serif headline, short subtext, one CTA). Once you click into `/data/officials/[id]`, the density shifts to BBRef/D&T — stat tiles at top, sortable table below. The transition from "editorial entrance" to "data workspace" is intentional.
6. **Suppressor/Amplifier visual language.** The core visual motif: blue pill for suppressors, red pill for amplifiers. This color-coding propagates everywhere — table cell backgrounds tinted lightly, bar chart fills, profile headers. It's the site's signature move and should be immediately recognizable from any screenshot.
7. **Responsive.** Mobile-first for prose pages; tables degrade to horizontal-scroll card views on small screens. The search box must work on mobile (it's the primary entry point).

### Component visual spec (key components only)

**SuppressorPill** — the most important component:
- Blue (`#003DA5`) background + white text for suppressors (mean_adj_fta36_delta < −0.5)
- Red (`#ED174C`) background + white text for amplifiers (> +0.5)
- Silver (`#8E9196`) background + white text for neutral (−0.5 to +0.5)
- Rounded corners (`border-radius: 999px`), horizontal padding, bold weight
- Shows the label ("Suppressor" / "Amplifier" / "Neutral") + the numeric score

**StatTile** — the D&T-inspired stat block:
- White card, subtle shadow
- Large number (28px, Inter 700) in blue or red depending on valence
- Small label below (12px, Inter 400, silver)
- Used in a 4-across row on official profile pages

**Data tables:**
- Sticky header row (off-white background, Inter 600, 13px uppercase tracking)
- Alternating row shading (white / very light off-white, barely perceptible)
- Sortable column headers with a subtle chevron icon
- Delta columns: positive values in red text, negative in blue text (matches pill language — blue = suppression = fewer FTAs, red = amplification)
- Right-aligned numeric columns, left-aligned name columns
- Row hover: light blue tint (`rgba(0, 61, 165, 0.04)`)

**SearchInput:**
- Full-width within its container, large (48px height), 16px text
- Blue bottom border on focus (2px solid `#003DA5`)
- Autocomplete dropdown: white card, shadow, each result row shows official name + suppressor pill inline
- Placeholder: "Search an official — e.g., 'Foster', 'Tiven', 'Brothers'"

**Navigation bar:**
- Sticky top, navy-dark (`#00214D`) background, white text
- Left: "ref-ball" in Instrument Serif (the only serif in the nav — signals the brand)
- Right: Data / The Attention Load / Does Harden Choke? / Papers / About
- Current page: white underline accent
- Mobile: hamburger → slide-out

### What "modern retro" means concretely (so the implementer doesn't guess)

The Tavus aesthetic is "modern retro" because it combines:
- **Serif display type** (retro — newspapers, broadsheets, editorial magazines)
- **Sans-serif body and UI** (modern — clean, digital-native)
- **Dramatic scale contrast** between hero headlines and body text (retro — editorial layout)
- **Generous whitespace and card elevation** (modern — Material/Apple design language)
- **Muted, intentional color** rather than gradient-heavy or illustration-heavy (retro restraint)

For ref-ball, the retro layer is: Instrument Serif headlines, 76ers colorway (classic Americana sports palette), data presented with authority (like a broadsheet box score). The modern layer is: cards, shadows, smooth interactions, responsive layout, the search-as-primary-entry UX pattern. **The combination should feel like "what if Basketball-Reference were redesigned by a type-obsessed editorial designer in 2026."**

---

## 2. Site map

```
/                                 umbrella thesis + entry panel
/the-attention-load               Attention Load (CSF) findings + model
/does-harden-choke                DHC long-form article (read mode, frozen archive)
/data                             dataset landing — what's published, methodology, license
/data/officials                   index of all 88 named officials (search + sort)
/data/officials/[official_id]     per-official profile page (THE centerpiece)
/data/players                     index of all 40 players (search + sort)
/data/players/[player_slug]       per-player page — their deltas across all officials
/data/download                    raw parquet exports + schema + license
/thesis (or /methodology)         attribution derivation, foul-type specificity, ANOVA
/papers/ssac27                    Paper 1 submission landing, abstract, figures
/about                           authorship posture, open-data commitment, reproducibility
/404
```

### What is intentionally NOT on the site
- **No Layer 2 (contact-type classification) as browseable data.** It's a live research frontier with a gate not cleared. Putting unstable numbers on a public dataset page confuses the published layer (Layer 1 + profiles) — which is the point of the dataset page — with the experimental layer. Layer 2 appears in `the-attention-load` and `papers/ssac27` as "what's next," not as data.
- **No live query box over arbitrary joins.** The lookup is one official at a time or one player at a time. A query language suggests flexibility the published data doesn't have.

---

## 3. Build pipeline — source parquets → site JSON

This is the most important section. Everything else is presentation.

### 3.1 Script: `site/scripts/build_site_data.py`

Runs from the ref-ball repo root. Reads parquets, writes JSON into `site/src/data/`.

**Output file inventory (all paths relative to `site/src/data/`):**

| File | Source parquet | Purpose |
|---|---|---|
| `officials/index.json` | `official_calling_profiles.parquet` (88 rows) | List page + search; one row per official, top-line fields |
| `officials/[id].json` | join of `official_calling_profiles` + `defensive_adjusted_interactions` filtered to that official + `ref_profiles` for SF rates | Full per-official profile page |
| `players/index.json` | aggregated from `player_official_interactions.parquet` groupby player_name | List page, 40 rows |
| `players/[slug].json` | `player_official_interactions` + `defensive_adjusted_interactions` for that player | Per-player page — their deltas across all officials, sortable |
| `crew.json` | `crew_assignments.parquet` (40,804 rows) | Used by /data/officials/[id] to show "games officiated by" sample, not for live lookup |
| `meta.json` | hard-coded constants | glossary, ANOVA p-values, headlines (p=0.000003, r=−0.528, 3.5x lift), git SHA, build date |

### 3.2 Per-official profile JSON schema (`officials/[id].json`)

```json
{
  "official_id": "1627534",
  "pbp_name": "A.Nagy",
  "official_name": "Andy Nagy",
  "headline": {
    "suppressor_score": 0.733,
    "mean_adj_fta36_delta": -0.72,
    "n_players": 15,
    "n_pairs": 15,
    "total_games": 267,
    "sf_per_game": 7.40,
    "sf_per_game_RS": 7.40,
    "sf_per_game_PO": null,
    "sf_pct_of_fouls": 0.508
  },
  "player_deltas": [
    {
      "player_name": "James Harden",
      "n_games_with": 9,
      "fta36_with": 7.22,
      "fta36_without": 8.94,
      "defense_adjusted_fta36_delta": -1.72,
      "defrtg_with": 112.2,
      "defrtg_delta": 2.30,
      "adjustment_magnitude": 0.42
    }
  ],
  "foul_type_breakdown": {
    "sf_per_game_RS": 7.40,
    "sf_per_game_PO": null,
    "sf_per_game_delta": 0.0,
    "n_shooting_fouls_RS": 1976,
    "n_shooting_fouls_PO": 0,
    "note": "Foul-type-specific rates from ref_profiles.parquet; full foul-type classification (Layer 2) pending — see /papers."
  },
  "context": {
    "rs_po_delta": null,
    "n_players_rs": 15,
    "n_players_po": 0,
    "crew_roles": "official_3 primary (sample from crew_assignments)"
  }
}
```

**Build script invariant:** Each per-official JSON must contain every field even when null (PO data is sparse). The frontend never has to apologize for missing fields — it renders "—" instead. This is the single most important rule for the frontend.

### 3.3 Per-player profile JSON schema (`players/[slug].json`)

```json
{
  "player_name": "James Harden",
  "slug": "james_harden",
  "baseline": {
    "baseline_fta_per_game": 8.92,
    "baseline_sf_per_game": 0.580,
    "n_games_without_official": 896
  },
  "official_deltas": [
    {
      "official_id": "1628953",
      "pbp_name": "A.Moyer-Gleich",
      "official_name": "Ashley Moyer-Gleich",
      "n_games_with": 13,
      "fta_with": 78, "fta_without": 7933,
      "fta_per_game_with": 6.00, "fta_per_game_without": 8.96,
      "fta_delta": -2.96,
      "sf_with": 6, "sf_without": 509,
      "sf_per_game_with": 0.46, "sf_per_game_without": 0.575,
      "sf_delta": -0.114
    }
  ],
  "defense_adjusted": [
    {
      "official_id": "1627534",
      "official_name": "Andy Nagy",
      "n_games_with": 17,
      "fta36_with": 6.10, "fta36_without": 8.94,
      "raw_fta36_delta": -2.84,
      "defense_adjusted_fta36_delta": -3.21,
      "defrtg_with": 113.4, "defrtg_delta": 3.5
    }
  ]
}
```

### 3.4 Build pipeline rules
1. The build script is idempotent — deletes `site/src/data/**` and rebuilds from parquet. No hand-edited JSON.
2. **Don't ship the parquets via the site build.** Parquets are downloaded from `/data/download` via links to GitHub release artifacts (per §6 decision).
3. The build writes a `meta.json` with `git_sha`, `built_at`, and the headline constants. Frontend renders the SHA in the page footer ("Dataset built from ref-ball@<sha>").
4. **No player games logs (`player_games/*.parquet`) ship as JSON yet — they're per-game, too large at 40 players × ~1k games. Occasion to revisit if we add a player-game explorer.**

---

## 4. Page-by-page spec

### `/` — Home / umbrella

**Goal:** visitor understands in 30 seconds that this is one thesis in three movements, and that there is a lookup to play with.

Layout (top to bottom):
1. **Hero:** one-line umbrella thesis + sub-line crediting the dataset. Title candidate: "Three studies in how the box score lies." Author tag below.
2. **Three-movement panel:** three side-by-side cards, each:
   - Codename (small, gray) + public name (headline)
   - One-sentence claim ("Refs call different games" / "Officiating errors are structurally predictable, not random" / "Harden doesn't choke — two-mode architecture does")
   - One headline number: p=0.000003 / 3.5x lift / r=−0.528
   - State pill: Live / Live / Frozen archive
   - Link to its page
3. **Pull-quote box:** "Official name parsed from an unstructured PBP `description` field. Nobody had used it for hypothesis-driven research." → links to `/data`.
4. **Try the data panel:** single input "Search an official (e.g., 'Nagy', 'Tiven')" → autocompletes from `officials/index.json` → goes to `/data/officials/[id]`.
5. **SSAC27 abstract link** if abstract is live.
6. **Footer:** dataset SHA, build time, license, author.

### `/data/officials/[official_id]` — Centerpiece

This page does the entire site's work. **If only one page is shipped, this is it.**

Sections:
1. **Header:** `official_name` (e.g., "Andy Nagy"), role (derived from crew majority), `n_games`, `total_sf`.
2. **Suppressor / Amplifier pill:** big colored pill — Suppresses (≤−0.5 mean adj delta) / Neutral (−0.5..+0.5) / Amplifies (≥+0.5). With the score and N_players.
3. **Stat row (4 tiles):**
   - `mean_adj_fta36_delta` (defense-adjusted) — the headline
   - `sf_per_game_RS` vs `sf_per_game_PO` (delta highlighted if both present)
   - `sf_pct_of_fouls` (share of all fouls that were shooting fouls)
   - `rs_po_delta` (only populated where both RS and PO exist; show "—" otherwise — don't apologize)
4. **Per-player deltas table:** the `player_deltas` array, sortable columns (n_games_with, fta36_with, defense_adjusted_fta36_delta, defrtg_delta). Default sort: most negative adj delta first (so the visitor sees "who does this ref suppress hardest" immediately). Client-side sort, no API.
5. **Back to officials index** + prev/next official by suppressor_score (encourages browsing — this is how people "lose 20 minutes" on the site, which is what we want for the moat).
6. **Methodology footnote:** short paragraph — adj delta = defense-adjusted FTA/36 delta, suppressor_score = share of player-pairs where this official gives fewer FTAs than baseline. Link to `/thesis`.

### `/data/officials` — Index

- Search box (client-side filter over 88 rows).
- Sortable table: name, suppressor_score, mean_adj_fta36_delta, n_pairs, total_games, n_games_RS/PO.
- Inline bar (CSS-only) visualizing suppressor_score (0..1) → immediately legible which officials suppress and which amplify.
- Default sort: most suppressing first. The story is the distribution, so the visitor lands seeing both ends of the spectrum.

### `/data/players/[slug]` — Per-player

- Header: `player_name`, baseline_fta_per_game, baseline_sf_per_game, total games.
- Two sorted views (toggle): "Raw FTA deltas" and "Defense-adjusted FTA/36 deltas."
- Sorted by delta. Lists every official that player shared games with, with `n_games_with` as the trust knob — show a min-n filter slider.
- Link back to any official's page → closes the browse loop.

### `/data/download`

- Table of downloadable artifacts:
  - `ref_profiles.parquet` — 101 officials, SF rates
  - `official_calling_profiles.parquet` — 88 officials, suppressor scores
  - `player_official_interactions.parquet` — 3,846 player×official raw FTA deltas
  - `defensive_adjusted_interactions.parquet` — 1,678 defense-adjusted deltas
  - `crew_assignments.parquet` — 40,804 crew rows
  - `data/processed/games/*.parquet` — 13,278 ingested games (large; zip if served)
- Each row: file, size, row count, schema (columns listed), one-sentence description, link to git for source code that produced it (`src/ref_profiles.py`, etc.).
- License block (CC-BY 4.0 for data, MIT for code — decide in §6).
- Citation block (BibTeX with current git SHA).

### `/the-attention-load` — Attention Load (CSF)

Not a dataset page — a *finding* page. Structure mirrors the CSF README:
1. Question: are late-game ref errors random or structurally predictable?
2. Method: 51,130 L2M events, MRT-grounded taxonomy.
3. Findings: error-rate bar by category (Ordinary 4% / Off-ball 10% / Traveling & OOB 16% / Defensive 3s & shot clock 20%). Observable Plot bar chart, static-rendered.
4. The model: 3.5x lift on top-10% risk, table per season from CSF README.
5. What this means: framework for Structural Officiating Risk → replay-assist / automation targets.
6. **Lifecycle note (must exist on the page):** "This is an ongoing research project, not a finished paper." Don't let visitors confuse it with DHC.

### `/does-harden-choke` — DHC long-form

- Mirror the README's article structure as long-form Markdown in Astro. **Copy the README to `site/src/content/dhc.md` at build time**, hand-tuned headings. This keeps the page self-updating if DHC's README ever changes (it won't — it's frozen — but the discipline is valuable).
- Render the tables from the README (floor rates, opponent gradients, cohort roster).
- Add a TL;DR callout at top matching the README's.
- **State pill:** "Frozen research archive. Active development moved to ref-ball."

### `/papers/ssac27`

- Abstract draft (linked to `documents/ssac27-abstract-draft.md`).
- The two figures: `figure_b_crew_prediction_scatter.png`, `table_a_suppressor_amplifier.png`.
- Submission timeline sidebar (Abstract Oct 1, 2026 / Full paper Dec 4, 2026).
- Open-data statement, naming-named-officials stance.

### `/about`

- One paragraph on authorship.
- Reproducibility: the Makefile targets are published.
- Open-data commitment.
- Counter-position to typical sports-analytics opacity (Owen Phillips found the field, this uses it for hypothesis testing).

---

## 5. Component / data-flow spec for the lookup

The single piece of interactivity worth getting right.

### Implementation

```text
[officials/index.json]  (88 rows, ~25 KB)
       |
       v
<SearchInput> (vanilla JS, debounce 60ms, substring match on pos_name | full_name)
       |
       v
<ResultList> (renders matched rows, click → window.location = /data/officials/${official_id})
       |
       v
[data/officials/[id].json] static-rendered at build time, hydrates the detail page
```

**Astro usage:** each `/data/officials/[id].html` and `/data/players/[slug].html` is generated at build time from `getStaticPaths()` iterating the JSON directory. No SSR. The detail pages themselves use a tiny `<TableSorter client:load>` island for the per-row sort — that's the only JS on the entire site besides the search input.

### Why pre-generated per-page JSON instead of one big file
- Each detail page only loads its own ~5 KB of data — instant.
- Google indexes per-official pages individually (each is a unique URL with a unique title: "Andy Nagy — suppressor profile · ref-ball").
- The "prev/next by suppressor_score" navigation works trivially because the build script can pre-compute neighbors.

---

## 6. Open decisions the implementer MUST resolve before coding

These are decisions, not tasks. A developer can't infer them; Harris has to pick.

1. **Domain.** `ref-ball.com` / `refball.dev` / subdomain of a personal site? Pick before deploy.
2. **Authorship posture.** Named (Harris Gordon, with professional context of the kind in `analysis/context.md`) or pseudonymous (handle)? Traction goal says named — but it commits Harris publicly. Decide.
3. **Data license.** CC-BY 4.0 (attribution required, allows commercial) is the conventional choice for max traction; CC0 relinquishes attribution (less leverage when someone builds on it). Lean CC-BY 4.0.
4. **Code license.** MIT (permissive, standard) vs Apache-2.0 (explicit patent grant, slightly heavier). Either is fine; pick once.
5. **Parquet hosting.** **DECIDED:** GitHub release artifacts via `gh release upload`. Clean, large files OK, no size caps, no infra ops. The `/data/download` page links to the release-asset URLs.
6. **Named-officials stance.** **DECIDED:** No anonymization. README already commits to this. Site restates it explicitly on `/data` as a feature of the dataset, not something to hide.
7. **SSAC framing on the site.** **DECIDED:** SSAC is a downstream page (`/papers/ssac27`). The dataset hub is the homepage. Don't let the conference deadline wag the site's primary asset.

Also resolved:
- **Domain:** Harris acquires own domain later. Site must be domain-agnostic until then (use Vercel preview URL during dev).
- **Deploy:** **DECIDED:** Vercel (Harris-owned custom domain, TBD later).
- **Authorship posture:** **DECIDED:** Named — Harris Gordon. Stated on `/about` and in citation blocks. Plays to the traction goal.
- **Data license:** **DECIDED:** CC-BY 4.0 (attribution required, permits commercial use).
- **Code license:** **DECIDED:** MIT.

---

## 7. Repo layout for `ref-ball/site/`

```
ref-ball/
└── site/
    ├── scripts/
    │   └── build_site_data.py         # reads ../data/processed/*.parquet, writes src/data/**/*.json
    ├── src/
    │   ├── data/                      # GENERATED — gitignored internally but built before deploy
    │   │   ├── meta.json
    │   │   ├── officials/
    │   │   │   ├── index.json
    │   │   │   └── [id].json
    │   │   ├── players/
    │   │   │   ├── index.json
    │   │   │   └── [slug].json
    │   │   └── crew.json
    │   ├── content/
    │   │   ├── dhc.md                 # built-from-readme at build time (see §4 /does-harden-choke)
    │   │   └── attention-load.md      # authored, derived from CSF README + model_responses.md
    │   ├── components/
    │   │   ├── SearchInput.astro
    │   │   ├── TableSorter.astro       # the one client:load island
    │   │   ├── SuppressorPill.astro
    │   │   ├── StatTile.astro
    │   │   └── LifecyclePill.astro
    │   ├── layouts/
    │   │   └── Base.astro
    │   ├── pages/
    │   │   ├── index.astro
    │   │   ├── the-attention-load.astro
    │   │   ├── does-harden-choke.astro
    │   │   ├── about.astro
    │   │   ├── data/
    │   │   │   ├── index.astro
    │   │   │   ├── download.astro
    │   │   │   ├── officials/
    │   │   │   │   ├── index.astro
    │   │   │   │   └── [id].astro     # getStaticPaths from src/data/officials/
    │   │   │   └── players/
    │   │   │       ├── index.astro
    │   │   │       └── [slug].astro
    │   │   ├── papers/
    │   │   │   └── ssac27.astro
    │   │   └── thesis.astro
    │   └── styles/
    │       └── base.css
    ├── astro.config.mjs
    ├── package.json
    ├── .gitignore                    # includes src/data/
    └── README.md                      # how to develop and deploy
```

`.gitignore` entries to add at ref-ball root:
```
site/node_modules/
site/dist/
site/src/data/          # built artifacts; regenerable from parquet
```

---

## 8. Build & deploy flow

```bash
# Local dev
cd site && npm install
python scripts/build_site_data.py       # populates src/data/
npm run dev                              # Astro dev server

# Deploy (on push to main)
# Vercel build command (Harris's own domain, TBD):
cd site && npm install && python ../scripts/build_site_data.py && npm run build
# Output directory: site/dist
```

The `python` invocation needs `pandas` + `pyarrow` from `requirements.txt` (already in ref-ball's venv). Vercel build environment supports custom install commands including Python; if that's painful, **alternative:** run the build script locally, commit the generated JSON, and have Vercel only run `npm run build`. That breaks the regenerability invariant slightly but eliminates the Python-on-Vercel headache. Pick at deploy time.

---

## 9. Build-order checklist (suggested)

Phase 1 — ship the moat (minimum to be worth visiting):
- [ ] Resolve §6 decisions 1–6 (domain, license, hosting, authorship).
- [ ] `scripts/build_site_data.py` — emits officials/index.json + officials/[id].json + meta.json (only).
- [ ] `/`, `/data`, `/data/officials/index`, `/data/officials/[id]`.
- [ ] Deploy to Vercel (Harris's own domain, TBD).
- [ ] Smoke-test the search on a real deployed URL.

Phase 2 — close the browse loop:
- [ ] Build players/* JSON; add `/data/players/index` + `/data/players/[slug]`.
- [ ] Prev/next-by-suppressor navigation on official pages.

Phase 3 — load the long tail:
- [ ] `/does-harden-choke`, `/the-attention-load`, `/papers/ssac27`, `/about`.
- [ ] Charts: Observable Plot for the three headline visuals.
- [ ] Per-official foul-type breakdown section (RS vs PO deltas; link to `/thesis` for Layer 2 frontier).

Phase 4 — download & citation:
- [ ] `/data/download` with the parquet inventory.
- [ ] License + citation blocks.
- [ ] Cross-link to GitHub source code that produces each artifact.

Phase 5 — narrative polish:
- [ ] Author-tuned thesis copy on home and /thesis.
- [ ] Open Graph tags + per-page meta descriptions (so messages-board links embed nicely — important for traction goal).
- [ ] Add plausible to track which officials get searched most (informs which profiles to feature).

---

## 10. Invariants the implementer must not violate

1. **No parquet imports in the frontend build.** Python → JSON is a one-way door.
2. **No missing-field apology text.** Render "—" for nulls. (Frontend decides; build script emits null for missing.)
3. **No anonymized officials.** The dataset's value is the named attribution. Don't hide it behind IDs in the UI.
4. **No "coming soon" placeholders for Layer 2 data.** Layer 2 appears in prose, not in a browseable table that implies published numbers.
5. **Build script is the only writer to `site/src/data/`.** Hand-edited JSON is forbidden — it desyncs the site from the parquet.
6. **SSAC deadline does not determine homepage priority.** The dataset hub is the homepage; SSAC is a downstream page.
7. **Per-official detail pages must load in <100 ms on a cold cache.** Static, small JSON, minimal JS. If this breaks, the lookup premise breaks.

---

## 11. What "done" looks like (Phase 1 acceptance)

A user types `ref-ball.com`, sees the umbrella thesis, types "Tiven" into a search box, lands on Josh Tiven's profile, sees his suppressor score and top-5 most-suppressed players, clicks one, lands on James Harden's per-player page, sees his deltas across officials, and leaves having learned something the box score can't tell them. That loop shipping is "done." Everything else is enrichment.