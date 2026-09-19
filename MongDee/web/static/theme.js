// Shared dark/light toggle for every page. Preference is per-browser
// (localStorage), not sent to the server — each page just needs a
// <button id="theme-toggle-btn"> in its top-bar and to load this after
// icons.js.
(function () {
    const STORAGE_KEY = 'mongdee-theme';

    function currentTheme() {
        try {
            return localStorage.getItem(STORAGE_KEY) || 'dark';
        } catch (e) {
            return 'dark';
        }
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const btn = document.getElementById('theme-toggle-btn');
        if (btn && typeof iconHtml === 'function') {
            btn.innerHTML = iconHtml(theme === 'light' ? 'moon' : 'sun', { size: 16 });
            btn.title = theme === 'light' ? 'สลับเป็นโหมดมืด' : 'สลับเป็นโหมดสว่าง';
        }
    }

    function toggleTheme() {
        const next = currentTheme() === 'light' ? 'dark' : 'light';
        try { localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* private mode etc. */ }
        applyTheme(next);
    }

    applyTheme(currentTheme());
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) btn.addEventListener('click', toggleTheme);
})();
