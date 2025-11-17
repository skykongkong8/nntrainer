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

function sanitizeResponse(text) {
  if (!text) return text;
  // Remove parentheticals that contain no meaningful tokens (e.g., "(, , and )").
  const withoutDangling = text.replace(/\(([^)]*)\)/g, (match, inner) => {
    const alphaNum = inner
      .replace(/[^0-9a-z]/gi, '')
      .replace(/(?:and|or)/gi, '')
      .trim();
    if (!alphaNum) return '';
    return match;
  });
  // Collapse stray double spaces introduced by removing the parentheticals.
  return withoutDangling
    .replace(/\s+,/g, ',')
    .replace(/ +/g, ' ');
}

function extractHeadings(text) {
  const headings = [];
  const lines = text.split('\n');
  let offset = 0;
  for (const line of lines) {
    if (/^#{2,6}\s*/.test(line)) {
      const normalized = line
        .replace(/^#{2,6}\s*/, '')
        .replace(/^[^A-Za-z0-9]+/, '')
        .replace(/[:*_`]/g, '')
        .trim()
        .toLowerCase();
      headings.push({ raw: line, normalized, offset });
    }
    offset += line.length + 1;
  }
  return headings;
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
  const headingDefs = [
    { label: 'What & Why', match: h => /what\s*&\s*why/.test(h) },
    { label: 'Key Changes', match: h => /key\s+changes/.test(h) },
    { label: 'Pointers for Reviewer', match: h => /(pointers?|reviewer)/.test(h) }
  ];
  const headings = extractHeadings(normalized);
  let lastOffset = -1;
  for (const def of headingDefs) {
    const match = headings.find(h => h.offset > lastOffset && def.match(h.normalized));
    if (!match) {
      errors.push(`Missing section: ${def.label}`);
      continue;
    }
    if (match.offset < lastOffset) {
      errors.push(`Section order is incorrect around ${def.label}.`);
    }
    lastOffset = match.offset;
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

function previewList(items, limit = 3) {
  if (!items || !items.length) return '';
  const filtered = items.filter(Boolean);
  if (!filtered.length) return '';
  const sliced = filtered.slice(0, limit);
  const suffix = filtered.length > limit ? ` …+${filtered.length - limit} more` : '';
  return `${sliced.join(', ')}${suffix}`;
}

function describeModuleImpact(moduleImpact) {
  const entries = Object.entries(moduleImpact || {});
  if (!entries.length) return [];
  return entries
    .sort((a, b) => (b[1].score || 0) - (a[1].score || 0))
    .slice(0, 5)
    .map(([name, data]) => {
      const stats = data.stats || {};
      const churn = (stats.added || 0) + (stats.removed || 0);
      const flags = [];
      if (stats.rename) flags.push('includes renames');
      if (stats.delete) flags.push('includes deletes');
      const flagText = flags.length ? ` (${flags.join(', ')})` : '';
      return `- ${name}: ${data.impact || 'Low'} impact across ${stats.files || 0} file(s) touching ~${churn} LOC${flagText}.`;
    });
}

function describeFileHighlights(fileHighlights) {
  if (!Array.isArray(fileHighlights) || !fileHighlights.length) return [];
  const describeStatus = status => {
    if (!status) return 'updated';
    if (status.startsWith('A')) return 'adds';
    if (status.startsWith('D')) return 'removes';
    if (status.startsWith('R')) return 'renames';
    return 'updates';
  };
  return fileHighlights.slice(0, 5).map(f => {
    const churn = `+${f.added || 0}/-${f.removed || 0}`;
    return `- ${f.path}: ${describeStatus(f.status)} (${churn})`;
  });
}

function describeDocs(docSnippets) {
  if (!Array.isArray(docSnippets) || !docSnippets.length) return [];
  return docSnippets.slice(0, 3).map(snippet => `- ${snippet.module}: ${snippet.path}`);
}

function buildPointers(reviewerSignals, fileHighlights) {
  const pointers = [];
  const addIf = (label, arr) => {
    const preview = previewList(arr);
    if (preview) pointers.push(`- ${label}: ${preview}`);
  };
  if (reviewerSignals) {
    addIf('Public/API headers touched', reviewerSignals.apiSurfaceChanges);
    addIf('Test files updated', reviewerSignals.testFiles);
    addIf('Concurrency primitives impacted', reviewerSignals.concurrencySensitive);
    if (reviewerSignals.riskAlerts) {
      for (const [hint, paths] of Object.entries(reviewerSignals.riskAlerts)) {
        addIf(hint, paths);
      }
    }
  }
  if (fileHighlights && fileHighlights.length) {
    const hotPaths = fileHighlights.slice(0, 2).map(f => f.path);
    addIf('Highest churn files', hotPaths);
  }
  if (!pointers.length) {
    pointers.push('- No concentrated hotspots detected; review the overall diff.');
  }
  return pointers;
}

function buildWhatWhy(context, errors) {
  const samples = Array.isArray(context.commitSamples) ? context.commitSamples : [];
  const whatLines = [];
  if (samples.length) {
    for (const sample of samples.slice(0, 3)) {
      whatLines.push(`- ${sample}`);
    }
  } else {
    const commitLines = normalize(context.commits)
      .split('\n')
      .filter(line => line.trim().startsWith('- '))
      .slice(0, 3);
    if (commitLines.length) {
      whatLines.push(...commitLines);
    }
  }
  if (!whatLines.length) {
    whatLines.push('- Review the diff for additional intent (commit summary unavailable).');
  }
  if (errors && errors.length) {
    whatLines.push(`- Guardrail rejected the initial AI draft: ${errors.join('; ')}`);
  }
  return whatLines;
}

function buildFallback(context, errors) {
  const moduleLines = describeModuleImpact(context.moduleImpact);
  const fileLines = describeFileHighlights(context.fileHighlights);
  const docLines = describeDocs(context.docSnippets);
  const pointers = buildPointers(context.reviewerSignals, context.fileHighlights);
  const whatLines = buildWhatWhy(context, errors);

  const keySection = [
    ...moduleLines,
    ...(fileLines.length ? ['', '**Notable files**', ...fileLines] : []),
    ...(docLines.length ? ['', '**Reference docs**', ...docLines] : [])
  ].filter(Boolean);

  const body = [
    '# Fallback PR description (AI output rejected)',
    '',
    '### 🎯 What & Why',
    ...whatLines,
    '',
    '### 💡 Key Changes',
    ...(keySection.length ? keySection : ['- No granular change data available.']),
    '',
    '### 🔎 Pointers for Reviewer',
    ...pointers
  ].join('\n');

  return body;
}

function main() {
  const responsePath = arg('--response');
  const contextPath = arg('--context');
  const outputPath = arg('--output', 'pr_comment.md');

  const rawResponse = readMaybe(responsePath);
  const context = parseContext(contextPath);
  const sanitized = sanitizeResponse(rawResponse);
  const validation = validateResponse(sanitized);

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
