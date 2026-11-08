// The motion layer's head line (taste-skill motion.md), as a file so the page needs no inline script
// under its Content-Security-Policy. The 700 ms safety mark shows the hero even if motion.js never runs.
document.documentElement.classList.add('m-enter');
setTimeout(() => document.documentElement.classList.add('m-enter-done'), 700);
