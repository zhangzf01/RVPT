(function(){
  const nav = document.getElementById('site-nav');
  const toggle = document.querySelector('.nav-toggle');
  const links = nav ? Array.from(nav.querySelectorAll('a[href^="#"]')) : [];

  if (toggle && nav) {
    toggle.addEventListener('click', () => {
      const isOpen = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(isOpen));
    });
  }

  // Smooth scroll and close mobile menu
  links.forEach(link => {
    link.addEventListener('click', (e) => {
      const id = link.getAttribute('href');
      if (!id || id.length < 2) return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      nav.classList.remove('open');
      if (toggle) toggle.setAttribute('aria-expanded', 'false');
      history.replaceState(null, '', id);
    });
  });

  // Active section highlighting (scroll spy)
  // Track intersection ratios and highlight the link for the section
  // that occupies the most space in the viewport.
  const sections = Array.from(document.querySelectorAll('main section[id]'))
    .filter(sec => links.some(a => a.getAttribute('href') === `#${sec.id}`));

  const ratioById = Object.fromEntries(sections.map(s => [s.id, 0]));

  const updateActive = () => {
    let bestId = null;
    let bestRatio = 0;
    for (const [id, r] of Object.entries(ratioById)) {
      if (r > bestRatio) { bestRatio = r; bestId = id; }
    }
    if (!bestId) return;
    const activeHref = `#${bestId}`;
    links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === activeHref));
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const id = entry.target.getAttribute('id');
      ratioById[id] = entry.intersectionRatio;
    });
    updateActive();
  }, { root: null, rootMargin: '0px 0px -35% 0px', threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] });

  sections.forEach(sec => observer.observe(sec));

  // Initialize once in case we load mid-page
  updateActive();

  // Footer year
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());

  // No slider needed now (single demo video placeholder)
  
  // Auto-resize embedded iframes to fit their content (opt-in via data-auto-height="true")
  const embeddedIframes = Array.from(document.querySelectorAll('iframe.embedded-iframe[data-auto-height="true"]'));
  const computeDocHeight = (doc) => {
    try {
      if (!doc) return 0;
      const body = doc.body;
      const html = doc.documentElement;
      if (!body || !html) return 0;
      // Use scrollHeight to allow shrink as content collapses
      return Math.max(body.scrollHeight, html.scrollHeight);
    } catch { return 0; }
  };
  const setIframeHeight = (iframe, height) => {
    if (!iframe) return;
    const target = Math.max(0, Math.floor(height || 0));
    if (target > 0) iframe.style.height = `${target}px`;
  };

  // Strategy 1: Listen for postMessage from subpages (works with file:// as well)
  window.addEventListener('message', (event) => {
    // Do not rely on origin; instead, verify the source matches one of our iframes
    const data = event.data;
    if (!data || typeof data !== 'object' || data.type !== 'subpage:height') return;
    const target = embeddedIframes.find((f) => {
      try { return f.contentWindow === event.source; } catch { return false; }
    });
    if (!target) return;
    const h = Number(data.height);
    if (Number.isFinite(h) && h > 0) setIframeHeight(target, h);
  });

  // Strategy 2: Same-origin direct measurement with ResizeObserver
  embeddedIframes.forEach((iframe) => {
    const attachObservers = () => {
      let ro = null; let mo = null;
      try {
        const doc = iframe.contentDocument;
        const body = doc && doc.body;
        if (!body) return;
        // Initial set
        setIframeHeight(iframe, computeDocHeight(doc));
        // ResizeObserver for layout changes
        ro = new ResizeObserver(() => setIframeHeight(iframe, computeDocHeight(doc)));
        ro.observe(body);
        // MutationObserver as a fallback for DOM changes
        mo = new MutationObserver(() => setIframeHeight(iframe, computeDocHeight(doc)));
        mo.observe(body, { attributes:true, childList:true, subtree:true, characterData:true });
        // A couple of delayed reads to catch late layout
        setTimeout(() => setIframeHeight(iframe, computeDocHeight(doc)), 50);
        setTimeout(() => setIframeHeight(iframe, computeDocHeight(doc)), 250);
      } catch (_) {
        // Cross-origin: skip direct access
      }

      // Cleanup on reload of iframe
      iframe.addEventListener('load', () => {
        if (ro) try { ro.disconnect(); } catch {}
        if (mo) try { mo.disconnect(); } catch {}
        // Re-attach for new document
        attachObservers();
      }, { once: true });
    };
    const docReady = () => {
      try {
        const rs = iframe.contentDocument && iframe.contentDocument.readyState;
        return rs === 'interactive' || rs === 'complete';
      } catch { return false; }
    };
    if (docReady()) attachObservers();
    else iframe.addEventListener('load', attachObservers, { once: true });
  });

  // Copy BibTeX button
  const copyBtn = document.querySelector('.bibtex-copy-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      const pre = document.querySelector('.bibtex-wrap pre.bibtex');
      if (!pre) return;
      const text = pre.textContent || '';
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = 'Copied';
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
      } catch (_) {
        // Fallback: select and copy
        const range = document.createRange();
        range.selectNodeContents(pre);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        try { document.execCommand('copy'); copyBtn.textContent = 'Copied'; } catch {}
        setTimeout(() => { copyBtn.textContent = 'Copy'; sel.removeAllRanges(); }, 1500);
      }
    });
  }
})();

