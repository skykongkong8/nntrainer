'use strict';

const { readFileSync } = require('fs');

function safeString(value) {
  return typeof value === 'string' ? value : '';
}

function pickCommitSubjects(commitsText, limit = 3) {
  const text = safeString(commitsText);
  if (!text.trim()) return [];

  const lines = text.split(/\r?\n/);
  const subjects = [];
  let capture = false;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!capture && /^samples:/i.test(line)) {
      capture = true;
      continue;
    }
    if (!capture) continue;
    if (!line) continue;
    if (/^commit bodies/i.test(line)) break;
    if (/^total commits/i.test(line)) break;
    if (/^-\s+/.test(line)) {
      const subject = line.replace(/^-\s*/, '').trim();
      if (subject) subjects.push(subject);
      if (subjects.length >= limit) break;
    } else if (!/^\d/.test(line)) {
      break;
    }
  }

  if (!subjects.length) {
    // Fallback: grab first non-empty line that looks like a commit subject.
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line || line.startsWith('- ') || /^total commits/i.test(line)) continue;
      subjects.push(line);
      if (subjects.length >= limit) break;
    }
  }

  return subjects.slice(0, limit);
}

function buildWhatWhyBullets(commitsText) {
  const subjects = pickCommitSubjects(commitsText, 4);
  if (!subjects.length) return ['Changes detected but commit subjects were unavailable.'];
  return subjects.map((subject, idx) => {
    const intent = describeIntent(subject, idx);
    return `${subject}${intent ? ` — ${intent}` : ''}`;
  });
}

function describeIntent(subject, idx) {
  const lower = subject.toLowerCase();
  if (lower.startsWith('feat')) return 'introduces a feature highlighted in the commit history';
  if (lower.startsWith('fix')) return 'addresses a defect noted by the latest commits';
  if (lower.startsWith('refactor')) return 'refactors existing logic for maintainability';
  if (lower.startsWith('test')) return 'focuses on testing or validation improvements';
  if (lower.startsWith('docs')) return 'touches documentation or explanatory material';
  if (lower.startsWith('chore')) return 'performs repository chores or configuration updates';
  if (lower.startsWith('revert')) return 'reverts earlier work to restore stability';
  if (lower.includes('perf') || lower.includes('speed')) return 'targets performance characteristics';
  if (lower.includes('cleanup') || lower.includes('tidy')) return 'cleans up legacy code paths';
  if (idx === 0) return 'primary intent surfaced from the git history';
  if (idx === 1) return 'secondary theme captured from recent commits';
  return '';
}

function parseNameStatusSection(diffText) {
  const text = safeString(diffText);
  const start = text.indexOf('### name-status');
  if (start === -1) return new Map();
  const statStart = text.indexOf('### stat');
  const segment = text
    .slice(start + '### name-status'.length, statStart === -1 ? undefined : statStart)
    .trim();
  const map = new Map();
  if (!segment) return map;
  for (const line of segment.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const parts = line.split('\t');
    const status = parts[0] ? parts[0].trim() : 'M';
    const toPath = parts[2] || parts[1] || '';
    const path = toPath.trim();
    if (path) map.set(path, status);
  }
  return map;
}

function parseStatSection(diffText) {
  const text = safeString(diffText);
  const start = text.indexOf('### stat');
  if (start === -1) return [];
  const segment = text.slice(start + '### stat'.length).trim();
  if (!segment) return [];
  const entries = [];
  for (const line of segment.split(/\r?\n/)) {
    if (!line.trim()) continue;
    if (/files changed|insertions|deletions/.test(line)) break;
    const pipeIdx = line.indexOf('|');
    if (pipeIdx === -1) continue;
    const path = line.slice(0, pipeIdx).trim();
    const rest = line.slice(pipeIdx + 1).trim();
    const totalMatch = rest.match(/(\d+)/);
    const total = totalMatch ? Number(totalMatch[1]) : undefined;
    const changeBar = rest.slice(totalMatch ? rest.indexOf(totalMatch[1]) + totalMatch[1].length : 0).trim();
    const plus = (changeBar.match(/\+/g) || []).length;
    const minus = (changeBar.match(/-/g) || []).length;
    entries.push({ path, total, plus, minus });
  }
  return entries;
}

function normalizeHighlights(rawHighlights) {
  if (!rawHighlights) return [];
  const highlights = [];
  if (Array.isArray(rawHighlights)) {
    for (const item of rawHighlights) {
      if (!item) continue;
      if (typeof item === 'string') {
        const parsed = parseHighlightLine(item);
        if (parsed) highlights.push(parsed);
      } else if (typeof item === 'object') {
        const path = item.path || item.file || item.name;
        if (!path) continue;
        highlights.push({
          path,
          summary: item.summary || item.title || item.description || '',
          excerpt: item.excerpt || item.diff || item.body || ''
        });
      }
    }
    return highlights;
  }
  if (typeof rawHighlights === 'string') {
    for (const line of rawHighlights.split(/\r?\n/)) {
      const parsed = parseHighlightLine(line);
      if (parsed) highlights.push(parsed);
    }
  }
  return highlights;
}

function parseHighlightLine(line) {
  if (!line) return null;
  const trimmed = line.trim();
  if (!trimmed) return null;
  const colonIdx = trimmed.indexOf(':');
  if (colonIdx === -1) return { path: trimmed, summary: '', excerpt: '' };
  const path = trimmed.slice(0, colonIdx).trim();
  const summary = trimmed.slice(colonIdx + 1).trim();
  if (!path) return null;
  return { path, summary, excerpt: '' };
}

function describeStatus(status) {
  if (!status) return '';
  if (/^m/i.test(status)) return 'modified';
  if (/^a/i.test(status)) return 'added';
  if (/^d/i.test(status)) return 'deleted';
  if (/^r/i.test(status)) return 'renamed';
  if (/^c/i.test(status)) return 'copied';
  return status;
}

function formatPlusMinus(plus, minus) {
  const tokens = [];
  if (typeof plus === 'number' && plus > 0) tokens.push(`+${plus}`);
  if (typeof minus === 'number' && minus > 0) tokens.push(`−${minus}`);
  if (!tokens.length) return '';
  return tokens.join('/');
}

function buildFileHighlightSentences(context, limit = 8) {
  const diffText = safeString(context.diff);
  const statEntries = parseStatSection(diffText);
  const statusMap = parseNameStatusSection(diffText);
  const highlightMap = new Map();
  for (const highlight of normalizeHighlights(context.fileHighlights)) {
    highlightMap.set(highlight.path, highlight);
  }

  const seen = new Set();
  const sentences = [];
  const upsertSentence = (path, summary, plus, minus, status) => {
    if (!path || seen.has(path)) return;
    seen.add(path);
    const descBits = [];
    if (summary) descBits.push(summary);
    const pm = formatPlusMinus(plus, minus);
    const statusWord = describeStatus(status);
    const metaBits = [];
    if (pm) metaBits.push(pm);
    if (statusWord) metaBits.push(statusWord);
    const meta = metaBits.length ? ` (${metaBits.join(', ')})` : '';
    const description = descBits.length ? descBits.join(' ') : 'Updated file';
    sentences.push(`- ${path}: ${description}${meta}`);
  };

  for (const entry of statEntries) {
    const highlight = highlightMap.get(entry.path);
    const summary = highlight?.summary || highlight?.excerpt?.split(/\r?\n/)[0];
    upsertSentence(entry.path, summary, entry.plus, entry.minus, statusMap.get(entry.path));
  }

  for (const [path, highlight] of highlightMap.entries()) {
    upsertSentence(path, highlight.summary || highlight.excerpt, undefined, undefined, statusMap.get(path));
  }

  if (!sentences.length && statEntries.length) {
    for (const entry of statEntries.slice(0, limit)) {
      upsertSentence(entry.path, '', entry.plus, entry.minus, statusMap.get(entry.path));
    }
  }

  return sentences.slice(0, limit);
}

function describeModuleImpact(moduleImpact, limit = 5) {
  if (!moduleImpact || typeof moduleImpact !== 'object') return [];
  const entries = Object.entries(moduleImpact);
  if (!entries.length) return [];
  entries.sort((a, b) => (b[1]?.score || 0) - (a[1]?.score || 0));
  const sentences = [];
  for (const [name, data] of entries.slice(0, limit)) {
    const stats = data?.stats || {};
    const filesCount = stats.files || (data?.files ? data.files.length : 0) || 0;
    const filePhrase = filesCount ? `${filesCount} file${filesCount === 1 ? '' : 's'}` : 'files';
    const adds = stats.added || 0;
    const dels = stats.removed || 0;
    let trend = '';
    if (adds > dels * 2 && adds >= 10) trend = 'mostly additions';
    else if (dels > adds * 2 && dels >= 10) trend = 'mostly deletions';
    const extraSignals = [];
    if (stats.rename) extraSignals.push('rename');
    if (stats.delete) extraSignals.push('deletion');
    const extra = extraSignals.length ? ` with ${extraSignals.join(' & ')} changes` : '';
    const sampleFiles = (data?.files || []).slice(0, 2).map(f => f.path).filter(Boolean);
    const sampleText = sampleFiles.length ? ` (e.g., ${sampleFiles.join(', ')})` : '';
    const trendText = trend ? `, ${trend}` : '';
    sentences.push(`- ${name}: ${data?.impact || 'Low'} impact touching ${filePhrase} (+${adds}/−${dels}${trendText})${extra}${sampleText}.`);
  }
  return sentences;
}

function describeReviewerSignals(signals) {
  if (!signals || typeof signals !== 'object') return [];
  const lines = [];
  if (signals.apiSurfaceChanges?.length) {
    lines.push(`API/config touch points: ${signals.apiSurfaceChanges.slice(0, 5).join(', ')}`);
  }
  if (signals.testFiles?.length) {
    lines.push(`Tests exercised: ${signals.testFiles.slice(0, 5).join(', ')}`);
  }
  if (signals.concurrencySensitive?.length) {
    lines.push(`Concurrency-sensitive files: ${signals.concurrencySensitive.slice(0, 5).join(', ')}`);
  }
  return lines;
}

function buildFallback({ context = {}, errors = [], response = '' } = {}) {
  const lines = [];
  lines.push('# ⚠️ AI summary unavailable');
  lines.push('The guardrail rejected the generated description, so here is a synthesized summary built from git metadata.');
  lines.push('');

  lines.push('### 🎯 What & Why');
  for (const bullet of buildWhatWhyBullets(context.commits)) {
    lines.push(`- ${bullet}`);
  }
  lines.push('');

  const fileHighlights = buildFileHighlightSentences(context);
  if (fileHighlights.length) {
    lines.push('### 💡 Key Changes');
    for (const sentence of fileHighlights) lines.push(sentence);
    lines.push('');
  }

  const moduleStories = describeModuleImpact(context.moduleImpact);
  if (moduleStories.length) {
    lines.push('### 🧭 Module Impact');
    for (const sentence of moduleStories) lines.push(sentence);
    lines.push('');
  }

  const reviewerSignals = describeReviewerSignals(context.reviewerSignals);
  if (reviewerSignals.length) {
    lines.push('### 🔎 Extra Signals');
    for (const signal of reviewerSignals) lines.push(`- ${signal}`);
    lines.push('');
  }

  if (errors.length) {
    lines.push('### 🛡️ Validator output');
    for (const err of errors) {
      const msg = typeof err === 'string' ? err : err?.message || JSON.stringify(err);
      lines.push(`- ${msg}`);
    }
    lines.push('');
  }

  if (response) {
    lines.push('<details>');
    lines.push('<summary>Original model response</summary>');
    lines.push('');
    lines.push(response.trim());
    lines.push('');
    lines.push('</details>');
  }

  return lines.join('\n');
}

function readJson(file) {
  if (!file) return {};
  return JSON.parse(readFileSync(file, 'utf8'));
}

if (require.main === module) {
  const [, , responsePath, contextPath] = process.argv;
  const response = responsePath ? readFileSync(responsePath, 'utf8') : '';
  const context = contextPath ? readJson(contextPath) : {};
  process.stdout.write(`${buildFallback({ context, response })}\n`);
}

module.exports = {
  buildFallback,
  buildFileHighlightSentences,
  buildWhatWhyBullets,
  describeModuleImpact,
  describeReviewerSignals,
  pickCommitSubjects,
};
