#!/usr/bin/env node
// Validates AI responses before posting and falls back to a deterministic summary if needed.
const { readFileSync, writeFileSync, existsSync } = require('fs');

function arg(name, def) {
  const i = process.argv.indexOf(name);
  if (i === -1) return def;
  return process.argv[i + 1] || def;
}

function readMaybe(filePath) {
  if (!filePath || !existsSync(filePath)) return '';
  return readFileSync(filePath, 'utf8');
}

function parseContext(filePath) {
  if (!filePath || !existsSync(filePath)) return {};
  try {
    return JSON.parse(readFileSync(filePath, 'utf8'));
  } catch (err) {
    console.warn(`[validate_response] Failed to parse context JSON: ${err.message}`);
    return {};
  }
}

function normalize(text) {
  return (text || '').replace(/\r\n/g, '\n').trim();
}

function validateResponse(raw) {
  const normalized = normalize(raw);
  const errors = [];
  if (!normalized) {
    errors.push('Model returned an empty response.');
  }
  if (normalized && !normalized.startsWith('# ')) {
    errors.push('Response must start with an H1 title ("# ").');
  }
  const requiredHeadings = [
    '### 🎯 What & Why',
    '### 💡 Key Changes',
    '### 🔎 Pointers for Reviewer'
  ];
  let lastIndex = -1;
  for (const heading of requiredHeadings) {
    const match = normalized.match(new RegExp(`^${heading}`, 'm'));
    if (!match) {
      errors.push(`Missing section: ${heading}`);
      continue;
    }
    const idx = normalized.indexOf(match[0]);
    if (idx < lastIndex) {
      errors.push(`Section order is incorrect around ${heading}.`);
    }
    lastIndex = idx;
  }
  const wordCount = normalized.split(/\s+/).filter(Boolean).length;
  if (wordCount && wordCount < 80) {
    errors.push('Response is too short to describe the PR (min 80 words).');
  }
  const asciiChars = (normalized.match(/[a-z0-9\s.,;:!?"'`#\-_*()]/gi) || []).length;
  if (normalized && asciiChars / normalized.length < 0.7) {
    errors.push('Response appears to contain too many non-English characters.');
  }
  const fenceCount = (normalized.match(/```/g) || []).length;
  if (fenceCount % 2 !== 0) {
    errors.push('Code fences are not balanced.');
  }
  return {
    valid: errors.length === 0,
    errors,
    normalized
  };
}

function listOrFallback(items, emptyMsg, limit = 5) {
  if (!items || !items.length) return [`- ${emptyMsg}`];
  const sliced = items.slice(0, limit);
  return sliced.map(item => (item.startsWith('-') ? item : `- ${item}`));
}

function listPreview(items, limit = 5) {
  if (!items || !items.length) return 'none';
  const sliced = items.slice(0, limit);
  const suffix = items.length > limit ? '…' : '';
  return `${sliced.join(', ')}${suffix}`;
}

function buildFallback(context, errors) {
  const moduleImpactSummary = normalize(context.moduleImpactSummary);
  const moduleLines = moduleImpactSummary
    ? moduleImpactSummary.split('\n').filter(Boolean)
    : [];
  const docSnippets = Array.isArray(context.docSnippets) ? context.docSnippets : [];
  const docLines = docSnippets.slice(0, 3).map(snippet => `- ${snippet.module}: ${snippet.path}`);
  const fileHighlights = Array.isArray(context.fileHighlights) ? context.fileHighlights : [];
  const fileLines = fileHighlights.slice(0, 5).map(f => `- ${f.path} (${f.status}, +${f.added}/-${f.removed})`);
  const reviewerSignals = context.reviewerSignals || {};
  const reviewerLines = [];
  if (reviewerSignals.apiSurfaceChanges && reviewerSignals.apiSurfaceChanges.length) {
    reviewerLines.push(`- API/config files: ${listPreview(reviewerSignals.apiSurfaceChanges)}`);
  }
  if (reviewerSignals.testFiles && reviewerSignals.testFiles.length) {
    reviewerLines.push(`- Tests touched: ${listPreview(reviewerSignals.testFiles)}`);
  }
  if (reviewerSignals.concurrencySensitive && reviewerSignals.concurrencySensitive.length) {
    reviewerLines.push(`- Concurrency primitives: ${listPreview(reviewerSignals.concurrencySensitive)}`);
  }
  if (reviewerSignals.riskAlerts) {
    for (const [hint, paths] of Object.entries(reviewerSignals.riskAlerts)) {
      if (!paths || !paths.length) continue;
      reviewerLines.push(`- ${hint}: ${listPreview(paths)}`);
    }
  }
  if (!reviewerLines.length) {
    reviewerLines.push('- No concentrated hotspots detected; review the overall diff.');
  }
  const commitSummary = normalize(context.commits)
    .split('\n')
    .filter(Boolean)
    .slice(0, 6)
    .join('\n');

  const body = [
    '# Fallback PR description (AI output rejected)',
    '',
    '### 🎯 What & Why',
    ...listOrFallback(errors, 'Model output did not satisfy validation requirements.'),
    '- Auto-generated deterministic summary derived from git metadata.',
    '',
    '### 💡 Key Changes',
    ...(moduleLines.length ? moduleLines.map(line => (line.startsWith('-') ? line : `- ${line}`)) : ['- No module impact summary available.']),
    '',
    '**Docs referenced for context**',
    ...(docLines.length ? docLines : ['- No module documentation snippets were loaded.']),
    '',
    '**Representative files**',
    ...(fileLines.length ? fileLines : ['- File-level highlights unavailable.']),
    commitSummary ? `\n**Commit summary**\n${commitSummary}` : '',
    '',
    '### 🔎 Pointers for Reviewer',
    ...reviewerLines
  ]
    .filter(Boolean)
    .join('\n');

  return body;
}

function main() {
  const responsePath = arg('--response');
  const contextPath = arg('--context');
  const outputPath = arg('--output', 'pr_comment.md');

  const rawResponse = readMaybe(responsePath);
  const context = parseContext(contextPath);
  const validation = validateResponse(rawResponse);

  let finalText = validation.normalized;
  if (!validation.valid) {
    console.warn('[validate_response] AI response failed validation:');
    validation.errors.forEach(err => console.warn(` - ${err}`));
    finalText = buildFallback(context, validation.errors);
  } else {
    console.log('[validate_response] AI response passed validation.');
  }

  writeFileSync(outputPath, `${finalText.trim()}\n`);
  if (!validation.valid) {
    console.warn('[validate_response] Posted fallback PR description.');
  }
}

main();
