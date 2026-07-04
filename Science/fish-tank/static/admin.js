const grid = document.getElementById('grid');
const countEl = document.getElementById('count');
const statusEl = document.getElementById('status');

async function refresh() {
  const res = await fetch('/api/fish');
  const roster = await res.json();
  countEl.textContent = roster.length;
  grid.innerHTML = '';
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

    const select = document.createElement('select');
    for (const [value, label] of [['seek', 'Attract'], ['flee', 'Repel']]) {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = label;
      if (entry.temperament === value) opt.selected = true;
      select.appendChild(opt);
    }
    select.onchange = () => setTemperament(entry.id, select.value);
    card.appendChild(select);

    const del = document.createElement('button');
    del.className = 'delete';
    del.textContent = 'Delete';
    del.onclick = () => deleteFish(entry.id);
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
}

async function deleteFish(id) {
  await fetch(`/api/fish/${id}`, { method: 'DELETE' });
  refresh();
}

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const data = new FormData(form);
  statusEl.textContent = 'Uploading...';
  try {
    const res = await fetch('/api/fish', { method: 'POST', body: data });
    const body = await res.json();
    if (res.ok) {
      statusEl.textContent = 'Added!';
      form.reset();
      refresh();
    } else {
      statusEl.textContent = 'Error: ' + (body.error || res.status);
    }
  } catch (e) {
    statusEl.textContent = 'Upload failed: ' + e;
  }
});

refresh();
setInterval(refresh, 4000);
