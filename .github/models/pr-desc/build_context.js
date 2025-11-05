#!/usr/bin/env node
// minimal context builder: no external deps
const { execSync } = require('child_process');
const { readFileSync, readdirSync, existsSync } = require('fs');
const { join } = require('path');

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i > -1 && process.argv[i + 1] ? process.argv[i + 1] : def;
}
const base = arg('--base', 'origin/main');
const head = arg('--head', 'HEAD');

function sh(cmd) {
  try { return execSync(cmd, { encoding: 'utf8' }).trim(); }
  catch (e) { return ''; }
}

function clip(s, max) {
  if (!s) return '';
  if (s.length <= max) return s;
  return s.slice(0, max) + `\n\n[truncated ${s.length - max} chars]`;
}

// 1) overview
const ctxRoot = '.github/models/pr-desc/context';
let overview = '';
const overviewPath = join(ctxRoot, 'overview.md');
if (existsSync(overviewPath)) {
  overview = readFileSync(overviewPath, 'utf8');
}
overview = clip(overview, 8000);

// 2) modules (pick a few md files deterministically)
let modules = '';
const modulesDir = join(ctxRoot, 'modules');
if (existsSync(modulesDir)) {
  const files = readdirSync(modulesDir).filter(f => f.endsWith('.md')).sort();
  const picked = files.slice(0, 5);
  for (const f of picked) {
    const body = readFileSync(join(modulesDir, f), 'utf8');
    modules += `\n\n## ${f}\n` + clip(body, 4000);
  }
}
modules = clip(modules, 12000);

// 3) diff summary (name-status + stat; include rename/copy detection)
const diffNameStatus = sh(`git diff --name-status -M -C ${base}...${head}`);
const diffStat       = sh(`git diff --stat ${base}...${head}`);
const diff = clip(
  `### name-status\n${diffNameStatus}\n\n### stat\n${diffStat}`,
  8000
);

// 4) commit subjects summary (naive CC buckets)
const subjects = sh(`git log --pretty=%s ${base}..${head}`);
const lines = subjects.split('\n').filter(Boolean);
const buckets = { feat:0, fix:0, refactor:0, test:0, docs:0, chore:0, other:0 };
for (const s of lines) {
  const t = s.toLowerCase();
  if      (t.startsWith('feat')) buckets.feat++;
  else if (t.startsWith('fix')) buckets.fix++;
  else if (t.startsWith('refactor')) buckets.refactor++;
  else if (t.startsWith('test')) buckets.test++;
  else if (t.startsWith('docs')) buckets.docs++;
  else if (t.startsWith('chore')) buckets.chore++;
  else buckets.other++;
}
const commits =
  `총 커밋: ${lines.length}\n` +
  Object.entries(buckets).map(([k,v]) => `- ${k}: ${v}`).join('\n') +
  (lines.length ? `\n\n샘플:\n- ${lines.slice(0,5).join('\n- ')}` : '');

const out = {
  overview,
  modules,
  diff,
  commits
};
process.stdout.write(JSON.stringify(out, null, 2));
