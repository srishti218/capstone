import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ArrowDownRight, Check, ChevronRight, CircleHelp, ClipboardList, Eye, EyeOff,
  FileClock, FlaskConical, Focus, Layers3, ListFilter, Minus, Plus, ScanLine, ShieldCheck, X,
} from 'lucide-react';
import './style.css';

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const details = await response.json().catch(() => ({}));
    throw new Error(details.error || `Request failed (${response.status}).`);
  }
  return response.json();
}

const statuses = { pending: 'Pending', approved: 'Approved', rejected: 'Rejected' };
const caseCode = (item) => item.demo ? 'DEMO-001' : item.id.slice(0, 8).toUpperCase();

function Status({ status }) {
  return <span className={`status ${status}`}><i />{statuses[status]}</span>;
}

function Worklist({ cases, current, onSelect, loaded }) {
  const [filter, setFilter] = useState('All');
  const displayed = cases.filter((item) => (
    filter === 'All' || (filter === 'Pending' ? item.status === 'pending' : item.status !== 'pending')
  ));
  return <aside className="worklist">
    <div className="kicker"><ListFilter size={15} /> CASE QUEUE</div>
    <div className="queue-title"><h2>Worklist</h2><span>{cases.length}</span></div>
    <div className="filters" aria-label="Filter worklist">
      {['All', 'Pending', 'Reviewed'].map((name) => <button key={name} aria-pressed={filter === name}
        onClick={() => setFilter(name)}>{name}</button>)}
    </div>
    <div className="case-list">
      {displayed.map((item) => <button key={item.id} onClick={() => onSelect(item.id)}
        aria-pressed={current?.id === item.id} className="case">
        <span className="case-icon">{item.modality === 'CT' ? <Layers3 size={19} /> : <ScanLine size={19} />}</span>
        <span className="case-text"><strong>{item.title}</strong><small>{caseCode(item)} · {item.modality} / {item.view}</small>
          <Status status={item.status} /></span><ChevronRight className="case-arrow" size={14} />
      </button>)}
      {!displayed.length && <p className="empty-copy">{loaded ? 'No cases in this view.' : 'Loading cases…'}</p>}
    </div>
    <div className="queue-footer"><span className="signal" /> {cases.some((item) => item.demo) ? 'Sample case available' : 'Local study workspace'}</div>
  </aside>;
}

function Scan({ result }) {
  const [overlay, setOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.8);
  const [zoom, setZoom] = useState(1);
  const [frame, setFrame] = useState(result.initial_frame ?? 0);
  const [badFrame, setBadFrame] = useState(null);
  const base = `/api/results/${result.id}/frames/${frame}`;
  return <section className="scan-panel" aria-label="DICOM preview">
    <div className="scan-toolbar">
      <span><Focus size={16} /> SCAN VIEWPORT</span>
      <div className="toolbar-actions">
        <button onClick={() => setOverlay(!overlay)} aria-pressed={overlay}>
          {overlay ? <Eye size={15} /> : <EyeOff size={15} />} <span>AI overlay</span>
        </button>
        <div className="zoom-tools">
          <button aria-label="Zoom out" disabled={zoom <= 1} onClick={() => setZoom(zoom - 0.25)}><Minus size={15} /></button>
          <button aria-label="Reset zoom" onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button>
          <button aria-label="Zoom in" disabled={zoom >= 2} onClick={() => setZoom(zoom + 0.25)}><Plus size={15} /></button>
        </div>
      </div>
    </div>
    <div className="scan-stage">
      <span className="scan-badge scan-top">{result.modality} <b>·</b> {result.view}</span>
      <span className="scan-badge scan-size">{result.width} × {result.height} PX</span>
      <div className="scan-clip">
        {badFrame !== frame ? <div className="scan-stack"
          style={{ transform: `scale(${zoom})`, '--ratio': result.width / result.height }}>
          <img key={base} src={`${base}/image`} alt={`${result.title} preview`} onError={() => setBadFrame(frame)} />
          {overlay && <img className="heatmap" src={`${base}/overlay`} alt="" style={{ opacity }} />}
        </div> : <p className="empty-copy">This scan frame could not be loaded.</p>}
      </div>
    </div>
    <div className="scan-footer">
      <span className="heat-key" style={{ visibility: overlay ? 'visible' : 'hidden' }}>
        <i className={result.kind === 'segmentation' ? 'mask' : ''} /> {result.overlay}
      </span>
      {result.frame_count > 1 && <label>Slice <input type="range" min="0" max={result.frame_count - 1} value={frame}
        onChange={(event) => { setBadFrame(null); setFrame(Number(event.target.value)); }} /></label>}
      <label>Opacity <input type="range" min="0" max="1" step="0.05" value={opacity}
        disabled={!overlay} onChange={(event) => setOpacity(Number(event.target.value))} /></label>
      <span className="frame-count">{frame + 1} / {result.frame_count}</span>
    </div>
  </section>;
}

function Review({ result, onReviewed }) {
  const [actor, setActor] = useState(sessionStorage.getItem('radreview-actor') || '');
  const [note, setNote] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const lastRequest = useRef(null);
  const score = result.confidence == null ? null : result.confidence * 100;

  async function decide(decision) {
    if (!actor.trim()) {
      setMessage('Enter your reviewer name.');
      return;
    }
    const fields = { actor: actor.trim(), note: note.trim(), decision };
    const key = JSON.stringify(fields);
    if (lastRequest.current?.key !== key) lastRequest.current = { key, id: crypto.randomUUID() };
    setBusy(true);
    setMessage('');
    try {
      const entry = await api(`/api/results/${result.id}/reviews`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...fields, request_id: lastRequest.current.id }),
      });
      sessionStorage.setItem('radreview-actor', actor.trim());
      lastRequest.current = null;
      setNote('');
      setMessage('Decision recorded in the audit log.');
      onReviewed(entry);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  return <div className="review-column">
    <section className="info-panel">
      <div className="kicker"><Activity size={16} /> MODEL OUTPUT</div>
      <div className="info-heading"><h2>AI assessment</h2>{result.demo && <span className="demo-tag">SAMPLE</span>}</div>
      <p className="finding"><ArrowDownRight size={17} />{result.label}</p>
      <div className="score-label">{result.score_label}</div>
      <div className="score">{score == null ? '—' : <>{score.toFixed(1)}<span>%</span></>}</div>
      {score != null && <div className="meter" role="meter" aria-label={result.score_label}
        aria-valuenow={score} aria-valuemin={0} aria-valuemax={100}><div style={{ width: `${score}%` }} /></div>}
      <dl className="details"><div><dt>Model</dt><dd title={result.checkpoint}>{result.model}</dd></div>
        <div><dt>Overlay</dt><dd>{result.overlay}</dd></div></dl>
      <p className="info-note"><CircleHelp size={15} />{result.demo ? 'The score and heatmap are illustrative.'
        : result.kind === 'classification' ? 'Grad-CAM highlights influence on the predicted class. Scores are uncalibrated.'
          : 'The score summarizes selected voxels. It is uncalibrated.'}</p>
    </section>
    <section className="info-panel review-panel">
      <div className="kicker"><ClipboardList size={16} /> HUMAN REVIEW</div>
      <div className="info-heading"><h2>Record decision</h2><Status status={result.status} /></div>
      <label>Reviewer<input placeholder="Enter reviewer name" value={actor} maxLength={80}
        onChange={(event) => setActor(event.target.value)} /></label>
      <label>Note <span>(optional)</span><textarea placeholder="Add a short observation…" value={note} maxLength={500}
        onChange={(event) => setNote(event.target.value)} /></label>
      <div className="decision-buttons">
        <button className="approve" disabled={busy} onClick={() => decide('approved')}><Check size={16} /> Approve</button>
        <button className="reject" disabled={busy} onClick={() => decide('rejected')}><X size={16} /> Reject</button>
      </div>
      <p className="info-note" role="status"><ShieldCheck size={15} />{message || 'Decisions are saved to the audit log.'}</p>
    </section>
  </div>;
}

function Audit({ entries, cases }) {
  return <section className="audit-panel">
    <div className="kicker"><FileClock size={16} /> REVIEW HISTORY</div>
    <h2>Audit log</h2><p>Each submitted decision has its own entry.</p>
    {entries.length ? <div className="table-scroll"><table>
      <thead><tr><th>Case</th><th>Decision</th><th>Reviewer</th><th>Time (UTC)</th><th>Note</th></tr></thead>
      <tbody>{entries.map((entry) => <tr key={entry.id}>
        <td>{caseCode(cases.find((item) => item.id === entry.result_id) || { id: entry.result_id })}</td>
        <td><Status status={entry.decision} /></td><td>{entry.actor}</td>
        <td><time dateTime={entry.created_at}>{entry.created_at.replace('T', ' ').replace('+00:00', '')}</time></td>
        <td>{entry.note || '—'}</td>
      </tr>)}</tbody>
    </table></div> : <div className="audit-empty"><FileClock size={26} /><strong>No decisions recorded</strong>
      <span>Approve or reject a result to create the first entry.</span></div>}
  </section>;
}

function App() {
  const [cases, setCases] = useState([]);
  const [selected, setSelected] = useState(null);
  const [entries, setEntries] = useState([]);
  const [tab, setTab] = useState('viewer');
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const revision = useRef(0);
  const current = cases.find((item) => item.id === selected) || cases[0];

  useEffect(() => {
    let canceled = false;
    let timer;
    async function refresh() {
      const snapshot = revision.current;
      try {
        const [results, audit] = await Promise.all([api('/api/results'), api('/api/audit')]);
        if (!canceled && snapshot === revision.current) {
          setCases(results);
          setEntries(audit);
          setLoaded(true);
          setError('');
        }
      } catch (failure) {
        if (!canceled) setError(failure.message);
      }
      if (!canceled) timer = setTimeout(refresh, 5000);
    }
    refresh();
    return () => { canceled = true; clearTimeout(timer); };
  }, []);

  function reviewed(entry) {
    revision.current += 1;
    setCases((previous) => previous.map((item) => item.id === entry.result_id ? { ...item, status: entry.decision } : item));
    setEntries((previous) => [entry, ...previous.filter((item) => item.id !== entry.id)].sort((a, b) => b.id - a.id));
  }

  return <>
    <header className="header">
      <a href="/" className="brand" aria-label="RadReview home">
        <span className="brand-icon"><ScanLine size={23} /></span><span>Rad<span>Review</span><small>IMAGING WORKSPACE</small></span>
      </a>
      <nav aria-label="Workspace">
        <button aria-current={tab === 'viewer' ? 'page' : undefined}
          onClick={() => setTab('viewer')}><Layers3 size={17} /> Study viewer</button>
        <button aria-current={tab === 'audit' ? 'page' : undefined}
          onClick={() => setTab('audit')}><FileClock size={17} /> Audit log</button>
      </nav>
      <span className="mode"><FlaskConical size={15} /> EDUCATION MODE</span>
    </header>
    <main>
      <div className="page-heading">
        <div><div className="eyebrow"><span className="signal" /> RADIOLOGY REVIEW WORKSPACE</div>
          <h1>{tab === 'viewer' ? 'Study review' : 'Decision history'}</h1>
          <p>{tab === 'viewer' ? 'Inspect scans, explore model overlays, and record your review.' : 'Review decisions across the local worklist.'}</p></div>
        {tab === 'viewer' && current && <div className="study-ref"><span>ACTIVE CASE</span><strong>{caseCode(current)}</strong></div>}
      </div>
      {error && <p className="error-banner" role="alert">{error}</p>}
      {tab === 'viewer' ? <div className="workspace">
        <Worklist cases={cases} current={current} loaded={loaded} onSelect={setSelected} />
        {current ? <><Scan key={current.id} result={current} /><Review key={current.id} result={current} onReviewed={reviewed} /></>
          : <section className="no-study"><ScanLine size={32} /><h2>{loaded ? 'No study results yet' : 'Loading workspace…'}</h2>
            {loaded && <p>Results appear after the worker processes an eligible Orthanc study.</p>}</section>}
      </div> : <Audit entries={entries} cases={cases} />}
      <footer><span><FlaskConical size={15} /> For education only, not clinical use.</span><span>RADREVIEW LAB <b>·</b> LOCAL WORKSPACE</span></footer>
    </main>
  </>;
}

createRoot(document.getElementById('root')).render(<App />);
