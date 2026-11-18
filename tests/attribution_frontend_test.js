// 归因页面前端行为测试：DOM 打桩 + 可控 fetch，执行页面真实脚本，验证
//  1) 启动载入示例配方
//  2) 正常归因 → 报告渲染（两端概率、四项贡献、误差、优先复核项）
//  3) 计算期间修改配方 → 旧报告立即撤下
//  4) 迟到的旧响应即使成功返回也绝不恢复报告
//  5) 非法输入 → 前端即拒绝，不发请求、不出报告
//  6) 服务端返回非 ok → 不渲染报告
const { loadPage, installDom, makeAssert } = require('./dom_harness');

const { js: pageJs } = loadPage('attribution.html');
installDom();

const assert = makeAssert();

const META = { feature_names: ['A','B','C','D'], total_grams: 1000, proba_threshold: 0.7,
  n_trees: 5, max_depth: 3, tree_depths: [3,3,3,3,3], train_seed: 20241117 };
const SAMPLE = { original: [250,250,350,150], adjusted: [390,250,210,150] };

function report(overrides) {
  const contributions = [
    { index:0, name:'A', original:250, adjusted:390, delta:140, contribution:0.2233 },
    { index:1, name:'B', original:250, adjusted:250, delta:0, contribution:0.0960 },
    { index:2, name:'C', original:350, adjusted:210, delta:-140, contribution:0.3833 },
    { index:3, name:'D', original:150, adjusted:150, delta:0, contribution:0.0556 } ];
  const sum = contributions.reduce((a, c) => a + c.contribution, 0);
  const difference = 0.7969 - 0.0387;
  return Object.assign({
    status: 'ok', original: SAMPLE.original, adjusted: SAMPLE.adjusted,
    original_probability: 0.0387, adjusted_probability: 0.7969,
    difference, contributions, contribution_sum: sum, error: sum - difference,
    priority_index: 2, priority_name: 'C',
    intermediates: [
      { subset: [], grams: SAMPLE.original, probability: 0.0387 },
      { subset: [0,1,2,3], grams: SAMPLE.adjusted, probability: 0.7969 },
    ],
    compute_ms: 1.2,
  }, overrides || {});
}

let postCount = 0;
// 页面脚本以全局 fetch 发起请求；所有请求都经由 fetchImpl（可在场景中替换）。
let fetchImpl = url => {
  if (url.includes('meta')) return Promise.resolve({ json: async () => META });
  if (url.includes('sample')) return Promise.resolve({ json: async () => SAMPLE });
  return Promise.resolve({ json: async () => report() });
};
global.fetch = (url, opts) => {
  if (opts && opts.method === 'POST') postCount++;
  return fetchImpl(url, opts);
};

const driver = `
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const $ = id => document.getElementById(id);
  const plain = s => (s || '').replace(/<[^>]*>/g, '');
  const assert = ${assert.toString()};
  const withDraw = () => onUserEdit();

  await sleep(20);
  assert('启动后载入示例配方', $('orig0').value == 250 && $('adj0').value == 390);
  assert('启动时报告区为空', $('panelOk').style.display === 'none');

  // --- 场景 1：正常归因渲染 ---
  postCount = 0;
  await compute();
  assert('计算发出一次归因请求', postCount === 1);
  assert('报告面板显示', $('panelOk').style.display === 'block');
  assert('两端概率已渲染', $('okProbOrig').textContent.includes('3.87')
      && $('okProbAdj').textContent.includes('79.69'));
  assert('概率变化已渲染', $('okDiff').textContent.includes('+75.82'));
  assert('贡献表含 4 项且贡献带正负号', ($('okRows').innerHTML.match(/<tr>/g) || []).length === 4
      && $('okRows').innerHTML.includes('+22.33') && $('okRows').innerHTML.includes('+38.33'));
  assert('优先复核项已渲染', $('okPriority').textContent.includes('C'));
  assert('误差已渲染', $('okError').textContent.length > 0);
  assert('中间配方可展开复算', $('okIntermediates').innerHTML.includes('∅'));

  // --- 场景 2：计算期间修改 → 立即撤下；迟到结果不恢复 ---
  let resolveStale;
  postCount = 0;
  fetchImpl = () => new Promise(res => { resolveStale = () => res({ json: async () => report() }); });
  const p = compute();                       // 发起请求（不等待）
  await sleep(5);
  $('adj0').value = 400;
  withDraw();                                // 用户修改配方（真实页面由 input 事件触发 onUserEdit）
  assert('修改后旧报告立即撤下', $('panelOk').style.display === 'none');
  assert('状态提示已撤下', $('status').textContent.includes('撤下'));
  resolveStale();                            // 迟到响应“绕过” abort 成功返回
  await p;
  await sleep(5);
  assert('迟到结果未恢复报告', $('panelOk').style.display === 'none');
  assert('迟到结果未改写状态', $('status').textContent.includes('撤下'));

  // --- 场景 3：撤下后重新计算仍正常 ---
  $('adj0').value = 390;
  fetchImpl = url => {
    if (url.includes('meta')) return Promise.resolve({ json: async () => META });
    if (url.includes('sample')) return Promise.resolve({ json: async () => SAMPLE });
    return Promise.resolve({ json: async () => report() });
  };
  postCount = 0;
  await compute();
  assert('重新计算后报告恢复', $('panelOk').style.display === 'block');

  // --- 场景 4：非法输入 → 前端拒绝，不发请求、不出报告 ---
  postCount = 0;
  $('adj0').value = 391;                     // 调整后总重变为 1001
  await compute();
  assert('非法总重不发请求', postCount === 0);
  assert('非法输入显示 invalid 面板', $('panelInvalid').style.display === 'block');
  assert('非法输入不显示报告', $('panelOk').style.display === 'none');
  assert('非法提示提到总克数', $('invalidReason').textContent.includes('1000'));

  $('adj0').value = 390;
  $('orig1').value = '';                     // 空值
  await compute();
  assert('空值不发请求', postCount === 0);
  assert('空值不显示报告', $('panelOk').style.display === 'none');

  $('orig1').value = 250;

  // --- 场景 5：服务端返回非法/失败 → 不渲染报告 ---
  const bads = [
    { status: 'invalid', reason: '总克数必须等于 1000' },
    { status: 'internal_error', reason: 'x' },
    { status: 'ok', contributions: null },             // 残缺数据
    Object.assign(report(), { error: 'NaN' }),         // 非数值字段
    Object.assign(report(), { error: NaN }),           // NaN 字段
    Object.assign(report(), { priority_index: 9 }),    // 越界下标
    Object.assign(report(), { contributions: [       // 贡献缺字段
      { name:'A', original:250, adjusted:390, delta:140 },
      { name:'B', original:250, adjusted:250, delta:0, contribution:0.1 },
      { name:'C', original:350, adjusted:210, delta:-140, contribution:0.2 },
      { name:'D', original:150, adjusted:150, delta:0, contribution:0.1 }] }),
    Object.assign(report(), { intermediates: [{ subset: [0], grams: [1,2,3], probability: 0.5 }] }),
    Object.assign(report(), { intermediates: 'x' }),   // 中间配方类型错误
  ];
  for (const [k, bad] of bads.entries()) {
    fetchImpl = () => Promise.resolve({ json: async () => bad });
    await compute();
    assert('非 ok/残缺响应不渲染报告 #' + k,
        $('panelOk').style.display === 'none' && $('panelFailure').style.display === 'block');
  }

  // --- 场景 6：请求异常 → 失败面板，不残留旧报告 ---
  fetchImpl = () => Promise.reject(new Error('网络错误'));
  $('adj0').value = 350; $('adj1').value = 290;  // 总重仍 1000
  await compute();
  assert('请求异常显示失败面板', $('panelFailure').style.display === 'block'
      && $('panelOk').style.display === 'none');
})();
`;

eval(pageJs + driver);
