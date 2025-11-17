// .github/models/pr-desc/build_context.js
const { execSync } = require('child_process');
const { readFileSync, readdirSync, existsSync } = require('fs');
const fs = require('fs');
const { join, resolve, relative } = require('path');

const workspaceRoot = process.cwd();

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return (i > -1 && process.argv[i + 1]) ? process.argv[i + 1] : def;
}
const base = arg('--base', 'origin/main');
const head = arg('--head', 'HEAD');

function sh(cmd) {
  try { return execSync(cmd, { encoding: 'utf8' }).trim(); }
  catch { return ''; }
}
function clip(s, max) {
  if (!s) return '';
  return s.length <= max ? s : s.slice(0, max) + `\n\n[truncated ${s.length - max} chars]`;
}

// ---------- 0) 규칙 로딩 ----------
const prDescRoot = '.github/models/pr-desc';
const ctxRoot = join(prDescRoot, 'context');
const modulesDir = join(ctxRoot, 'modules');
const rulesPath = join(prDescRoot, 'rules.json');
let rules = { modules: [], fallbackModule: 'Misc' };
if (existsSync(rulesPath)) {
  try { rules = JSON.parse(readFileSync(rulesPath, 'utf8')); }
  catch { /* ignore parse error; keep defaults */ }
}
const moduleRuleMap = new Map();
for (const m of rules.modules || []) {
  moduleRuleMap.set(m.name, m);
}
const compiledPatterns = rules.modules.map(m => ({
  name: m.name,
  weight: Number(m.weight || 1),
  regs: (m.patterns || []).map(p => new RegExp(p))
}));

const globCache = new Map();
function globToRegExp(glob) {
  if (!globCache.has(glob)) {
    const escaped = glob.replace(/[-[\]{}()+?.,\\^$|#\s]/g, '\\$&');
    const regex = '^' + escaped
      .replace(/\\\*\\\*/g, '.*')
      .replace(/\\\*/g, '[^/]*')
      .replace(/\\\?/g, '.') + '$';
    globCache.set(glob, new RegExp(regex));
  }
  return globCache.get(glob);
}
function matchesGlob(glob, target) {
  if (!glob) return false;
  try { return globToRegExp(glob).test(target); }
  catch { return false; }
}

function locateDocFile(docPath) {
  if (!docPath) return null;
  const candidates = [];
  if (docPath.startsWith('/')) candidates.push(docPath);
  candidates.push(resolve(prDescRoot, docPath));
  candidates.push(resolve(workspaceRoot, docPath));
  for (const abs of candidates) {
    if (existsSync(abs)) {
      return { abs, rel: relative(workspaceRoot, abs) };
    }
  }
  return null;
}

function fallbackDocForModule(moduleName) {
  if (!existsSync(modulesDir)) return [];
  const key = moduleName.toLowerCase().replace(/\W/g, '');
  const candidates = readdirSync(modulesDir).filter(f => f.endsWith('.md'));
  const matched = candidates.find(f => f.toLowerCase().replace(/\W/g, '').includes(key));
  if (!matched) return [];
  const abs = join(modulesDir, matched);
  return [{ abs, rel: relative(workspaceRoot, abs) }];
}

function docCandidatesForModule(moduleName) {
  const rule = moduleRuleMap.get(moduleName);
  const configured = (rule && Array.isArray(rule.docs)) ? rule.docs : [];
  const located = configured.map(locateDocFile).filter(Boolean);
  if (located.length) return located;
  return fallbackDocForModule(moduleName);
}

function buildRiskAlerts(files) {
  const hints = rules.riskHints || {};
  const alerts = {};
  for (const [hint, patterns] of Object.entries(hints)) {
    const hits = new Set();
    for (const pattern of patterns || []) {
      for (const file of files) {
        if (matchesGlob(pattern, file.path)) {
          hits.add(file.path);
        }
      }
    }
    if (hits.size) alerts[hint] = Array.from(hits);
  }
  return alerts;
}

function diffExcerptForFile(pathname) {
  const safePath = pathname.replace(/"/g, '\\"');
  return clip(sh(`git diff -U3 ${base}...${head} -- "${safePath}"`), 2000);
}

function buildFileHighlights(files) {
  const limit = 12;
  return files
    .map(f => {
      const churn = churnMap.get(f.path) || { added: 0, removed: 0 };
      return { ...f, added: churn.added, removed: churn.removed, churn: churn.added + churn.removed };
    })
    .sort((a, b) => b.churn - a.churn)
    .slice(0, limit)
    .map(f => ({
      path: f.path,
      status: f.status,
      added: f.added,
      removed: f.removed,
      diffExcerpt: diffExcerptForFile(f.path)
    }));
}

function classifyModule(filepath) {
  for (const m of compiledPatterns) {
    if (m.regs.some(r => r.test(filepath))) return { module: m.name, weight: m.weight };
  }
  return { module: rules.fallbackModule || 'Misc', weight: 1 };
}

// ---------- 1) Overview / Modules 원문 문서 ----------
let overview = '';
const overviewPath = join(ctxRoot, 'overview.md');
if (existsSync(overviewPath)) {
  overview = readFileSync(overviewPath, 'utf8');
}
overview = clip(overview, 8000);

// ---------- 2) Git diff/numstat ----------
const nameStatusRaw = sh(`git diff --name-status -M -C ${base}...${head}`);
const statRaw = sh(`git diff --stat ${base}...${head}`);
const numstatRaw = sh(`git diff --numstat -M -C ${base}...${head}`);

// name-status 파싱 (status, path[, path2])
// M A D R100 old -> new 형태는 탭으로 분리
const changedFiles = [];
if (nameStatusRaw) {
  for (const line of nameStatusRaw.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    const status = parts[0]; // e.g., 'M', 'A', 'D', 'R100'
    const from = parts[1];
    const to = parts[2] || parts[1];
    changedFiles.push({ status, from, to, path: to });
  }
}

// numstat 파싱 (added removed path[, path2])
const churnMap = new Map(); // key: path(to), value: {added, removed}
if (numstatRaw) {
  for (const line of numstatRaw.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    if (parts.length < 3) continue;
    const added = parts[0] === '-' ? 0 : parseInt(parts[0], 10) || 0;
    const removed = parts[1] === '-' ? 0 : parseInt(parts[1], 10) || 0;
    const pth = parts[2].includes('\t') ? parts[3] : parts[2]; // rename의 경우 path\tpath2
    const path = parts[3] || parts[2];
    churnMap.set(path, { added, removed });
  }
}

// === A. build changedFiles FIRST ===
// parse name-status/numstat/etc. (existing code that fills changedFiles)
// ensure you have something like:
// const changedFiles = []; // declare before pushing to it
// ... push { path, status, additions, deletions, ... } into changedFiles

// === B. relevance-driven module docs (SAFE: changedFiles is ready) ===
function statusWeight(status) {
  // Rxxx, Cxxx 등은 리네임/복사로 간주
  if (status.startsWith('R') || status.startsWith('C')) return 2.0;
  if (status === 'D') return 2.5;
  if (status === 'A') return 1.5;
  // 기본 M
  return 1.0;
}

// ---------- 3) 모듈 그룹화 + 영향도 산출 ----------
const modulesAgg = new Map(); // name -> { files:[], score:0, weight, counts, lines }
const unmatched = [];
for (const f of changedFiles) {
  const cls = classifyModule(f.path);
  if (!cls || !cls.module) { unmatched.push(f.path); continue; }
  const key = cls.module;
  if (!modulesAgg.has(key)) {
    modulesAgg.set(key, {
      files: [],
      score: 0,
      baseWeight: cls.weight,
      count: 0,
      adds: 0,
      dels: 0,
      hasRename: false,
      hasDelete: false
    });
  }
  const agg = modulesAgg.get(key);
  agg.files.push({ path: f.path, status: f.status });
  agg.count += 1;

  const churn = churnMap.get(f.path) || { added: 0, removed: 0 };
  agg.adds += churn.added;
  agg.dels += churn.removed;
  if (f.status.startsWith('R') || f.status.startsWith('C')) agg.hasRename = true;
  if (f.status === 'D') agg.hasDelete = true;

  // 점수 = (모듈 가중치) * (상태 가중치) * (규모 가중치)
  const sizeFactor = Math.log10(1 + churn.added + churn.removed + 1); // 0~대략 4
  agg.score += cls.weight * statusWeight(f.status) * (1 + sizeFactor);
}

// 모듈별 최종 영향도 레벨 결정
function levelFromScore(s) {
  if (s >= 30) return 'High';
  if (s >= 12) return 'Medium';
  return 'Low';
}
function bumpForSignals(agg) {
  let bonus = 0;
  if (agg.hasDelete) bonus += 2.0;
  if (agg.hasRename) bonus += 1.0;
  if (agg.count >= 10) bonus += 1.5;
  const churn = agg.adds + agg.dels;
  if (churn >= 500) bonus += 2.0;
  else if (churn >= 200) bonus += 1.0;
  return bonus;
}
const moduleImpact = {};
for (const [name, agg] of modulesAgg.entries()) {
  const score = agg.score + bumpForSignals(agg);
  moduleImpact[name] = {
    impact: levelFromScore(score),
    score: Math.round(score * 10) / 10,
    files: agg.files,
    stats: { files: agg.count, added: agg.adds, removed: agg.dels,
             rename: agg.hasRename, delete: agg.hasDelete }
  };
}

const sortedModuleEntries = Object.entries(moduleImpact)
  .sort((a, b) => b[1].score - a[1].score);
const docSnippets = [];
let modulesDoc = '';
if (sortedModuleEntries.length) {
  const docBudget = 24000;
  const totalScore = sortedModuleEntries.reduce((sum, [, data]) => sum + Math.max(1, data.score), 0) || 1;
  let usedBudget = 0;
  for (const [moduleName, data] of sortedModuleEntries) {
    if (usedBudget >= docBudget) break;
    const candidates = docCandidatesForModule(moduleName);
    if (!candidates.length) continue;
    const moduleBudget = Math.max(2000, Math.round((Math.max(1, data.score) / totalScore) * docBudget));
    let moduleUsed = 0;
    for (const candidate of candidates) {
      if (usedBudget >= docBudget || moduleUsed >= moduleBudget) break;
      const raw = readFileSync(candidate.abs, 'utf8');
      const available = Math.max(0, Math.min(6000, moduleBudget - moduleUsed));
      if (!available) break;
      const excerpt = clip(raw, available);
      if (!excerpt) continue;
      docSnippets.push({ module: moduleName, path: candidate.rel, excerpt });
      modulesDoc += `\n\n## ${moduleName}: ${candidate.rel}\n${excerpt}`;
      const increment = excerpt.length;
      usedBudget += increment;
      moduleUsed += increment;
    }
  }
}
modulesDoc = clip(modulesDoc.trim(), 24000);

// extra reviewer signals (place AFTER changedFiles built)
function headerOrConfig(p){
 return /\.(h|hpp|hh|hxx|inc)$/.test(p) ||
 /(^|\/)(CMakeLists\.txt|configure|.*\.cmake|.*\.bazel|build\.gradle|settings\.gradle|package\.json)$/.test(p);
}
const apiSurfaceChanges = changedFiles.filter(f => headerOrConfig(f.path)).map(f => f.path);
const testFiles = changedFiles.filter(f => /(^|\/)(test|tests|testing|spec)\b|_test\.(cc|cpp|c|py|js|ts)$/.test(f.path)).map(f => f.path);
const concurrencySensitive= changedFiles.filter(f => /(thread|mutex|atomic|lock|concurrent|parallel)/i.test(f.path)).map(f => f.path);
const riskAlerts = buildRiskAlerts(changedFiles);
const fileHighlights = buildFileHighlights(changedFiles);

// ---------- 4) Diff/Commits 텍스트 ----------
const diff = clip(`### name-status\n${nameStatusRaw}\n\n### stat\n${statRaw}`, 8000);

const subjects = sh(`git log --pretty=%s ${base}..${head}`).split('\n').filter(Boolean);
const bodiesRaw = sh(`git log --pretty=%B ${base}..${head}`);
const bodies = bodiesRaw.split('\n\n').map(s=>s.trim()).filter(Boolean).slice(0,10).map(s=>clip(s,800));

const buckets = { feat:0, fix:0, refactor:0, test:0, docs:0, chore:0, other:0 };
for (const s of subjects) {
  const t = s.toLowerCase();
  if (t.startsWith('feat')) buckets.feat++;
  else if (t.startsWith('fix')) buckets.fix++;
  else if (t.startsWith('refactor')) buckets.refactor++;
  else if (t.startsWith('test')) buckets.test++;
  else if (t.startsWith('docs')) buckets.docs++;
  else if (t.startsWith('chore')) buckets.chore++;
  else buckets.other++;
}
const commits =
 `Total commits: ${subjects.length}\n` +
  Object.entries(buckets).map(([k,v]) => `- ${k}: ${v}`).join('\n') +
 (subjects.length ? `\n\nSamples:\n- ${subjects.slice(0,5).join('\n- ')}` : '') +
 (bodies.length ? `\n\nCommit bodies (top, clipped):\n- ${bodies.join('\n- ')}` : '');

// ---------- 5) 모듈 임팩트 요약 텍스트 (모델 힌트용) ----------
let moduleImpactSummary = '';
if (sortedModuleEntries.length) {
  const top = sortedModuleEntries.slice(0,5);
  moduleImpactSummary = top.map(([name, m]) =>
    `- ${name}: impact=${m.impact} (score=${m.score}, files=${m.stats.files}, +${m.stats.added}/-${m.stats.removed}${m.stats.rename?', rename':''}${m.stats.delete?', delete':''})`
  ).join('\n');
}

// ---------- 6) 출력 ----------
const out = {
  overview: clip(overview, 8000),
  modules: modulesDoc,
  docSnippets,
  diff,
  commits,
  // 새 필드들
  moduleImpact, // 모듈별 상세(머신 가독)
  moduleImpactSummary: moduleImpactSummary || '(no module impact detected)',
  reviewerSignals: {
    apiSurfaceChanges,
    testFiles,
    concurrencySensitive,
    riskAlerts
  },
  fileHighlights
};

process.stdout.write(JSON.stringify(out, null, 2));
