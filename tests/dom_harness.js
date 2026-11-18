// DOM 打桩环境：在 Node 中直接执行页面的真实脚本，用于前端行为测试。
// 不引入任何第三方依赖；各页面测试共享本文件。
const fs = require('fs');
const path = require('path');

function loadPage(relPath) {
  const html = fs.readFileSync(path.join(__dirname, '..', 'app', 'static', relPath), 'utf8');
  const m = html.match(/<script>([\s\S]*)<\/script>/);
  if (!m) throw new Error('页面未找到内联脚本：' + relPath);
  return { html, js: m[1] };
}

function installDom() {
  const store = {};
  function el(id) {
    if (!store[id]) {
      const e = {
        id, checked: false, textContent: '', innerHTML: '',
        style: { display: 'none' }, dataset: {}, disabled: false,
        listeners: {},
        classList: { add() {}, remove() {} },
        addEventListener(ev, fn) { (this.listeners[ev] ??= []).push(fn); },
        appendChild() {},
      };
      let v = '0';
      Object.defineProperty(e, 'value', {   // 真实 DOM 中 input.value 恒为字符串
        get: () => v, set: x => { v = String(x); },
      });
      store[id] = e;
    }
    return store[id];
  }
  global.document = {
    getElementById: el,
    querySelector: () => el('tbody'),
    createElement: () => el('tr_' + Math.random()),
  };
  return { el, store };
}

function installFetch() {
  let impl = () => Promise.reject(new Error('fetch not stubbed'));
  global.fetch = (...a) => impl(...a);
  return {
    set(fn) { impl = fn; },
    get() { return impl; },
  };
}

// 每个场景一条断言；失败写入 process.exitCode。
function makeAssert() {
  return (name, cond, detail) => {
    console.log((cond ? '[PASS] ' : '[FAIL] ') + name + (detail ? ' — ' + detail : ''));
    if (!cond) process.exitCode = 1;
  };
}

module.exports = { loadPage, installDom, installFetch, makeAssert };
