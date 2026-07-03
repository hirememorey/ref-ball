import fs from 'node:fs';
import path from 'node:path';

const DATA_DIR = path.join(process.cwd(), 'src', 'data');

export type OfficialIndexRow = {
  official_id: string;
  pbp_name: string;
  official_name: string;
  suppressor_score: number | null;
  mean_adj_fta36_delta: number | null;
  mean_raw_fta36_delta: number | null;
  n_players: number | null;
  n_pairs: number | null;
  total_games: number | null;
  n_games: number | null;
  sf_per_game: number | null;
  sf_per_game_RS: number | null;
  sf_per_game_PO: number | null;
  sf_per_game_delta: number | null;
  sf_pct_of_fouls: number | null;
  rs_po_delta: number | null;
  n_players_rs: number | null;
  n_players_po: number | null;
  crew_role: string | null;
};

export type OfficialDetail = {
  official_id: string;
  pbp_name: string;
  official_name: string;
  headline: Record<string, number | null>;
  player_deltas: Array<Record<string, string | number | null>>;
  foul_type_breakdown: Record<string, string | number | null>;
  context: Record<string, string | number | null>;
  navigation: { prev_by_suppressor: string | null; next_by_suppressor: string | null };
};

export type Meta = {
  built_at: string;
  git_sha: string;
  github_url: string;
  headlines: Record<string, number>;
  counts: Record<string, number>;
  license: { data: string; code: string };
  author: string;
};

function readJson<T>(p: string): T {
  return JSON.parse(fs.readFileSync(p, 'utf-8')) as T;
}

export function getMeta(): Meta {
  return readJson(path.join(DATA_DIR, 'meta.json'));
}

export function getOfficialsIndex(): OfficialIndexRow[] {
  return readJson(path.join(DATA_DIR, 'officials', 'index.json'));
}

export function getOfficialIds(): string[] {
  return fs
    .readdirSync(path.join(DATA_DIR, 'officials'))
    .filter((f) => f.endsWith('.json') && f !== 'index.json')
    .map((f) => f.replace('.json', ''));
}

export function getOfficial(id: string): OfficialDetail {
  return readJson(path.join(DATA_DIR, 'officials', `${id}.json`));
}

export function getPlayersIndex() {
  return readJson<Array<Record<string, string | number | null>>>(path.join(DATA_DIR, 'players', 'index.json'));
}

export function getPlayerSlugs(): string[] {
  return fs
    .readdirSync(path.join(DATA_DIR, 'players'))
    .filter((f) => f.endsWith('.json') && f !== 'index.json')
    .map((f) => f.replace('.json', ''));
}

export function getPlayer(slug: string) {
  return readJson(path.join(DATA_DIR, 'players', `${slug}.json`));
}

export function formatNum(v: number | null | undefined, d = 2): string {
  if (v === null || v === undefined) return '—';
  return v.toFixed(d);
}

export function formatSigned(v: number | null | undefined, d = 2): string {
  if (v === null || v === undefined) return '—';
  return `${v > 0 ? '+' : ''}${v.toFixed(d)}`;
}

export type Valence = 'suppressor' | 'amplifier' | 'neutral';

export function valence(meanAdj: number | null): Valence {
  if (meanAdj === null) return 'neutral';
  if (meanAdj <= -0.5) return 'suppressor';
  if (meanAdj >= 0.5) return 'amplifier';
  return 'neutral';
}

export function valenceLabel(v: Valence): string {
  return v === 'suppressor' ? 'Suppressor' : v === 'amplifier' ? 'Amplifier' : 'Neutral';
}

export function deltaClass(v: number | null): string {
  if (v === null) return '';
  if (v < 0) return 'delta-negative';
  if (v > 0) return 'delta-positive';
  return '';
}

export function roleLabel(role: string | number | null): string {
  if (!role) return '—';
  return String(role).replace(/_/g, ' ');
}

export function barFillClass(meanAdj: number | null): string {
  return valence(meanAdj) === 'amplifier' ? 'bar-fill--red' : 'bar-fill--gold';
}
