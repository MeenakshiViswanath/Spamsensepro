const SAMPLES = {
  spam: "URGENT: Your account has been suspended. Verify your password at http://secure-login.verify-now.info immediately. You have also won a $5000 prize - click now to claim!",
  ham: "Hi Rahul, the project review is scheduled on Friday at 5 pm in the conference room. Please find the updated timetable attached. Let me know if you have any questions."
};
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function sample(k){ $('text').value = SAMPLES[k]; }
function clearAll(){ $('text').value=''; $('result').style.display='none'; $('err').textContent=''; }

async function check(){
  const text = $('text').value.trim(); $('err').textContent='';
  if(!text){ $('err').textContent='Please paste some email text first.'; return; }
  $('btn').disabled = true; $('btn').textContent = 'Analyzing...';
  try{
    const r = await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const d = await r.json(); if(!r.ok) throw new Error(d.error||'Request failed');
    $('badge').textContent = d.label==='Spam' ? 'SPAM' : 'HAM (Not Spam)'; $('badge').className = 'badge '+d.label;
    $('conf').textContent = d.confidence+'% confidence'; $('sp').textContent = d.spam_probability;
    $('fill').style.width = d.spam_probability+'%';
    $('kw').innerHTML = d.keywords.map(k=>`<span class="chip ${k.signal}">${esc(k.word)}</span>`).join('') || '<span class="chip">none</span>';
    $('result').style.display='block';
  }catch(e){ $('err').textContent = e.message; }
  $('btn').disabled = false; $('btn').textContent = 'Analyze Email';
}

async function loadMetrics(){
  const m = await (await fetch('/api/metrics')).json(); if(!m.accuracy) return;
  const item=(l,v)=>`<div class="stat"><div class="n">${v}</div><div class="l">${l}</div></div>`;
  const pct=x=>(x*100).toFixed(1)+'%';
  $('metrics').innerHTML = item('Accuracy',pct(m.accuracy))+item('Precision',pct(m.precision))+item('Recall',pct(m.recall))+item('F1-score',pct(m.f1));
}

async function loadHistory(){
  const d = await (await fetch('/api/history')).json();
  $('total').textContent = d.stats.Spam + d.stats.Ham; $('spam').textContent = d.stats.Spam; $('ham').textContent = d.stats.Ham;
  $('rows').innerHTML = d.history.map(h=>`<tr><td>${h.id}</td><td class="txt">${esc(h.email_text.slice(0,200))}${h.email_text.length>200?'...':''}</td>
    <td><span class="badge ${h.label}" style="font-size:.85rem;padding:3px 12px">${h.label}</span></td><td>${h.confidence}%</td><td>${esc(h.created_at)}</td></tr>`).join('')
    || '<tr><td colspan="5" style="color:var(--muted)">No emails checked yet.</td></tr>';
}
async function clearHist(){ if(confirm('Delete all history?')){ await fetch('/api/history/clear',{method:'POST'}); loadHistory(); } }

if($('metrics')) loadMetrics();
if($('rows')) loadHistory();
