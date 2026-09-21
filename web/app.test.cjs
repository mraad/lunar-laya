const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
  const before = {x:500,y:30,vx:0,vy:-1,angle:0,fuel:80,time:0,status:'flying',score:0};
  const after = {...before,y:28,time:0.2,status:'landed',score:50};
  const decision = {proposed:{turn:0,throttle:0},executed:{turn:0,throttle:0},
    intervened:false,answers:null,prompt:'fixture',latency_ms:1};
  const summary = {...after,target:1,seed:3000,interventions:0,decisions:1,latency_p50_ms:1};
  const record = {schema_version:1,pilot:{mode:'baseline'},world:{terrain:[[0,20],[1000,20]],
    pads:[[150,240,30,2],[440,560,20,1],[775,825,40,4]]},
    episodes:[{summary,frames:[{before,after,decision}]}]};
  const run = {id:'0',label:'Guidance baseline',file:'run-0.json',mode:'baseline',summaries:[summary]};
  const elements = new Map(), listeners = {};
  const canvas = new Proxy({}, {get:()=>()=>{}});
  function element() { return {value:'0',textContent:'',style:{},children:[],dataset:{},width:1100,height:680,
    classList:{toggle(){}},append(...items){this.children.push(...items)},
    replaceChildren(){this.children=[]},getContext(){return canvas}}; }
  const context = vm.createContext({console,AbortController,location:{hash:'#flight'},
    document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},
      createElement:element,querySelectorAll(){return []}},
    window:{addEventListener(name,fn){listeners[name]=fn}},requestAnimationFrame(){},
    async fetch(url){return {ok:true,json:async()=>url==='manifest.json'?{version:1,runs:[run]}:record}}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'app.js'),'utf8'),context);
  await new Promise(setImmediate);
  assert.equal(elements.get('notice').textContent,'');
  assert.equal(elements.get('flight-status').textContent,'FLYING');
  assert.equal(elements.get('comparison-table').children.length,1);
  elements.get('timeline').value='1';elements.get('timeline').oninput();
  assert.equal(elements.get('flight-status').textContent,'LANDED');
  elements.get('restart').onclick();
  assert.equal(elements.get('flight-status').textContent,'FLYING');
  context.document.getElementById('speed').value='4';
  vm.runInContext('tick(0); tick(100)',context);
  assert.equal(elements.get('flight-status').textContent,'LANDED');
  context.location.hash='#compare';listeners.hashchange();
  assert.equal(elements.get('flight-page').hidden,true);
  assert.equal(elements.get('compare-page').hidden,false);
  assert.equal(elements.get('page-title').textContent,'Compare pilots');
  run.mode='laya';run.inference={bits:6};
  await vm.runInContext('loadRun(state.runs[0])',context);
  assert.equal(elements.get('engine-tag').textContent,'MLX / Q6');
  context.fetch=async()=>({ok:false,status:404});
  await vm.runInContext('loadRun(state.runs[0])',context);
  assert.match(elements.get('notice').textContent,/404/);
  assert.equal(elements.get('play').disabled,true);
  console.log('SPA loading, comparison, scrubbing, playback, routing and fetch errors: passed');
}
main().catch(error=>{console.error(error);process.exitCode=1});
