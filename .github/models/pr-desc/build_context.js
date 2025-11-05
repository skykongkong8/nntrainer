// .github/models/pr-desc/build_context.js
#!/usr/bin/env node
const { execSync } = require('child_process');
const { readFileSync, readdirSync, existsSync } = require('fs');
const { join } = require('path');

const arg = (k, d) => { const i=process.argv.indexOf(k); return i>-1 && process.argv[i+1] ? process.argv[i+1] : d; };
const base = arg('--base','origin/main');
const head = arg('--head','HEAD');

const sh = c => { try { return execSync(c,{encoding:'utf8'}).trim(); } catch { return ''; } };
const clip = (s,m) => s && s.length>m ? s.slice(0,m)+`\n\n[truncated ${s.length-m} chars]` : (s||'');

const ctx = '.github/models/pr-desc/context';
let overview = existsSync(join(ctx,'overview.md')) ? readFileSync(join(ctx,'overview.md'),'utf8') : '';
let modules = '';
const mdir = join(ctx,'modules');
if (existsSync(mdir)) {
  for (const f of readdirSync(mdir).filter(f=>f.endsWith('.md')).sort().slice(0,5))
    modules += `\n\n## ${f}\n` + readFileSync(join(mdir,f),'utf8');
}
const diff = `### name-status
${sh(`git diff --name-status -M -C ${base}...${head}`)}

### stat
${sh(`git diff --stat ${base}...${head}`)}
`;
const subjects = sh(`git log --pretty=%s ${base}..${head}`).split('\n').filter(Boolean);
const buckets = { feat:0, fix:0, refactor:0, test:0, docs:0, chore:0, other:0 };
for (const s of subjects) {
  const t=s.toLowerCase();
  buckets[
    t.startsWith('feat')?'feat':
    t.startsWith('fix')?'fix':
    t.startsWith('refactor')?'refactor':
    t.startsWith('test')?'test':
    t.startsWith('docs')?'docs':
    t.startsWith('chore')?'chore':'other'
  ]++;
}
const commits = `총 커밋: ${subjects.length}
- feat: ${buckets.feat}
- fix: ${buckets.fix}
- refactor: ${buckets.refactor}
- test: ${buckets.test}
- docs: ${buckets.docs}
- chore: ${buckets.chore}
- other: ${buckets.other}

샘플:
- ${subjects.slice(0,5).join('\n- ')}`;

process.stdout.write(JSON.stringify({
  overview: clip(overview,8000),
  modules: clip(modules,12000),
  diff: clip(diff,8000),
  commits: clip(commits,4000)
}, null, 2));
