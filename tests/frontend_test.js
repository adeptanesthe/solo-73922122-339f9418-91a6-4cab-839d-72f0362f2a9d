// 试配页前端撤回逻辑功能测试：DOM 打桩 + 可控 fetch，执行页面真实脚本，验证
//  1) 求解期间修改输入 → 旧建议立即撤下（结果区清空）
//  2) 迟到的旧响应即使成功返回也绝不恢复显示
//  3) 正常求解 → 结果正常渲染
const { loadPage, installDom, makeAssert } = require('./dom_harness');

const { js: pageJs } = loadPage('index.html');
installDom();
const assert = makeAssert();

const META = { feature_names: ['A','B','C','D'], total_grams: 1000, proba_threshold: 0.7,
  n_trees: 5, max_depth: 3, tree_depths: [3,3,3,3,3], train_seed: 1 };
const SAMPLES = { feasible: { original: [250,250,350,150], lower: [0,0,0,0],
  upper: [1000,1000,1000,1000], locked: [false,true,false,false] },
  infeasible: { original: [150,130,620,100], lower: [0,0,620,0],
  upper: [150,1000,1000,1000], locked: [false,false,false,false] } };
const OK_RESULT = { status: 'ok', suggestion: [390,250,210,150], probability: 0.7969,
  changed_count: 2, total_change: 280, solve_ms: 50, optimality: '已证明最优…',
  per_ingredient: [
    { name:'A', original:250, new:390, delta:140, locked:false },
    { name:'B', original:250, new:250, delta:0, locked:true },
    { name:'C', original:350, new:210, delta:-140, locked:false },
    { name:'D', original:150, new:150, delta:0, locked:false } ] };

// 页面脚本以全局 fetch 发起请求；所有请求经由 fetchImpl（场景中可替换）。
let fetchImpl = url => Promise.resolve({ json: async () => url.includes('meta') ? META : SAMPLES });
global.fetch = (url, opts) => fetchImpl(url, opts);

const driver = `
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const $ = id => document.getElementById(id);
  const assert = ${assert.toString()};

  await sleep(20);
  assert('启动后载入可行样例', $('orig0').value == 250);

  // --- 场景 1：正常求解渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_RESULT });
  await solve();
  assert('正常求解后 ok 面板显示', $('panelOk').style.display === 'block');
  assert('概率已渲染', $('okProb').textContent.includes('79.69'));

  // --- 场景 2：求解期间修改 → 立即撤下；迟到结果不恢复 ---
  let resolveStale;
  fetchImpl = () => new Promise(res => { resolveStale = () => res({ json: async () => OK_RESULT }); });
  const p = solve();                 // 发起求解（不等待）
  await sleep(5);
  onUserEdit();                      // 用户修改输入
  assert('修改后旧建议立即撤下', $('panelOk').style.display === 'none');
  assert('状态提示已撤下', $('status').textContent.includes('撤下'));
  resolveStale();                    // 迟到响应“绕过”abort 成功返回（竞态）
  await p;
  await sleep(5);
  assert('迟到结果未恢复 ok 面板', $('panelOk').style.display === 'none');
  assert('迟到结果未改写状态', $('status').textContent.includes('撤下'));

  // --- 场景 3：撤下后重新求解仍正常 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_RESULT });
  await solve();
  assert('重新求解后 ok 面板恢复', $('panelOk').style.display === 'block');

  // --- 场景 4：无解分支渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => ({ status: 'infeasible', reason: '已证明无解', solve_ms: 30 }) });
  await solve();
  assert('无解面板显示', $('panelInfeasible').style.display === 'block');
  assert('ok 面板保持隐藏', $('panelOk').style.display === 'none');
})();
`;

eval(pageJs + driver);
