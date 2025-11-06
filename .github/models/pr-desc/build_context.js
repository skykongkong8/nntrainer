#!/usr/bin/env node
/**
 * Build PR context for prompt.yml
 * Writes:
 *  - pr_context.json
 *  - ctx_overview.md
 *  - ctx_modules.md
 *  - ctx_diff.md
 *  - ctx_commits.md
 */

const fs = require('fs');
const path = require('path');
const cp = require('child_process');

function argOf(name, def) {
  const idx = process.argv.indexOf(name);
  if (idx >= 0 && process.argv[idx + 1]) return process.argv[idx + 1];
  return def;
}

function readIfExists(p) {
  try {
    return fs.readFileSync(p, 'utf8');
  } catch {
    return '';
  }
}

function listModuleDocs(dir) {
  try {
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.md'));
    files.sort();
    const parts = [];
    for (const f of files) {
      const title = path.basename(f, '.md');
      const body = readIfExists(path.join(dir, f)).trim();
      if (body) {
        parts.push(`## ${title}\n\n${body}\n`);
      }
    }
    return parts.join('\n');
  } catch {
    return '';
  }
}

function sh(cmd, opts = {}) {
  return cp.execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], ...opts }).trim();
}

const base = argOf('--base', 'origin/main');
const head = argOf('--head', 'HEAD');

// Overview & modules
const overview = readIfExists(path.join('.github', 'models', 'pr-desc', 'context', 'overview.md')).trim();
const modules = listModuleDocs(path.join('.github', 'models', 'pr-desc', 'context', 'modules')).trim();

// Diff summaries
let diffList = '';
let diffStat = '';
try {
  diffList = sh(`git diff --name-status --find-renames --diff-filter=ACDMRTUXB ${base}..${head}`);
  diffStat = sh(`git diff --stat ${base}..${head}`);
} catch (e) {
  diffList = `! Failed to compute diff: ${e.message}`;
  diffStat = '';
}
const diffCombined = [
  '### Changed files (name-status)',
  '```',
  diffList || '(none)',
  '```',
  '',
  '### Diff stat',
  '```',
  diffStat || '(none)',
  '```',
].join('\n');

// Commits
let commits = '';
try {
  commits = sh(`git log --no-merges --pretty=format:"%h %ad %s (%an)" --date=short ${base}..${head}`);
} catch (e) {
  commits = `! Failed to get commits: ${e.message}`;
}

// Write split files for file_input
fs.writeFileSync('ctx_overview.md', overview || '(no overview context)');
fs.writeFileSync('ctx_modules.md', modules || '(no module docs)');
fs.writeFileSync('ctx_diff.md', diffCombined || '(no diff)');
fs.writeFileSync('ctx_commits.md', (commits || '(no commits)') + '\n');

// JSON bundle (for debugging/traceability)
const prContext = {
  base,
  head,
  generated_at: new Date().toISOString(),
  overview,
  modules,
  diff_markdown: diffCombined,
  commits_text: commits,
};

fs.writeFileSync('pr_context.json', JSON.stringify(prContext, null, 2));
console.log('Context written: pr_context.json, ctx_*.md');