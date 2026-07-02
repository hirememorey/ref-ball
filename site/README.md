# ref-ball.site

Static Astro site for the ref-ball research hub. See `documents/development/SITE-PLAN.md` for the full spec.

## Local development

```bash
cd site
npm install
../.venv/bin/python scripts/build_site_data.py
npm run dev -- --host    # Network URL for mobile preview
```

## Production build

```bash
../.venv/bin/python scripts/build_site_data.py
npm run build            # → dist/
npm run preview -- --host
```

## Deploy (Vercel)

- Root directory: `site`
- Build: `python scripts/build_site_data.py && npm run build`
- Output: `dist`

If Python isn't available on Vercel, run the data script locally and commit `src/data/` (gitignored by default).
