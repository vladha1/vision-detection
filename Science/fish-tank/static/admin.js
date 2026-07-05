const grid = document.getElementById('grid');
const countEl = document.getElementById('count');
const capacityFill = document.getElementById('capacityFill');
const statusEl = document.getElementById('status');
const MAX_FISH = 20;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const dropHint = document.getElementById('dropHint');
const submitBtn = document.getElementById('submitBtn');

fileInput.addEventListener('change', () => showPreview(fileInput.files[0]));

['dragover', 'dragenter'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); })
);
dropZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    showPreview(file);
  }
});

function showPreview(file) {
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  dropHint.textContent = file.name;
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = kind || '';
}

async function refresh() {
  const res = await fetch('/api/fish');
  const roster = await res.json();
  countEl.textContent = roster.length;
  capacityFill.style.width = `${Math.min(100, (roster.length / MAX_FISH) * 100)}%`;
  grid.innerHTML = '';

  if (roster.length === 0) {
    grid.innerHTML = '<div class="empty">No fish yet - add one above!</div>';
    return;
  }

  for (const entry of roster) {
    const card = document.createElement('div');
    card.className = 'card';

    if (entry.kind === 'image') {
      const img = document.createElement('img');
      img.src = '/sprites/' + entry.filename;
      card.appendChild(img);
    } else {
      const swatch = document.createElement('div');
      swatch.className = 'swatch';
      swatch.style.background = entry.color;
      card.appendChild(swatch);
    }

    const toggle = document.createElement('div');
    toggle.className = 'badge-toggle';
    for (const [value, label] of [['seek', 'Attract'], ['flee', 'Repel']]) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = value + (entry.temperament === value ? ' active' : '');
      btn.textContent = label;
      btn.onclick = () => setTemperament(entry.id, value);
      toggle.appendChild(btn);
    }
    card.appendChild(toggle);

    const del = document.createElement('button');
    del.className = 'delete';
    del.textContent = 'Delete';
    del.onclick = () => {
      if (confirm('Remove this fish from the tank?')) deleteFish(entry.id);
    };
    card.appendChild(del);

    grid.appendChild(card);
  }
}

async function setTemperament(id, temperament) {
  await fetch(`/api/fish/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ temperament }),
  });
  refresh();
}

async function deleteFish(id) {
  await fetch(`/api/fish/${id}`, { method: 'DELETE' });
  refresh();
}

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const data = new FormData(form);
  submitBtn.disabled = true;
  setStatus('Uploading...', '');
  try {
    const res = await fetch('/api/fish', { method: 'POST', body: data });
    const body = await res.json();
    if (res.ok) {
      setStatus('Fish added!', 'ok');
      form.reset();
      preview.style.display = 'none';
      dropHint.textContent = 'Tap to take a photo or choose one from your library';
      refresh();
    } else {
      setStatus('Error: ' + (body.error || res.status), 'err');
    }
  } catch (err) {
    setStatus('Upload failed: ' + err, 'err');
  } finally {
    submitBtn.disabled = false;
  }
});

refresh();
setInterval(refresh, 4000);
