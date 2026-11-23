(() => {
'use strict';
const fileInput=document.getElementById('csvFile'), runBtn=document.getElementById('runBtn'), message=document.getElementById('message');
let rows=[];
const $=id=>document.getElementById(id);
function say(text,error=false){message.textContent=text;message.className='message'+(error?' error':'')}
function parseCSV(text){
 const lines=text.replace(/^\uFEFF/,'').trim().split(/\r?\n/);if(lines.length<80)throw Error('Please provide at least 80 rows of historical candles.');
 const split=line=>line.split(',').map(x=>x.trim().replace(/^"|"$/g,''));
 const headers=split(lines[0]).map(x=>x.toLowerCase());
 const idx={};['date','open','high','low','close'].forEach(k=>{idx[k]=headers.indexOf(k);if(idx[k]<0)throw Error('Missing required column: '+k);});
 const data=lines.slice(1).filter(Boolean).map((line,i)=>{const c=split(line);const o={date:c[idx.date],open:+c[idx.open],high:+c[idx.high],low:+c[idx.low],close:+c[idx.close]};if(!o.date||![o.open,o.high,o.low,o.close].every(Number.isFinite)||o.close<=0)throw Error('Invalid OHLC data on CSV row '+(i+2));return o;});
 for(let i=1;i<data.length;i++){if(data[i].date<data[i-1].date)throw Error('Sort CSV rows oldest to newest before importing.');}
 return data;
}
function features(data,i){
 const c=data[i].close, prev=data[i-1].close, p2=data[i-2].close,p5=data[i-5].close;
 const mean=n=>data.slice(i-n+1,i+1).reduce((s,r)=>s+r.close,0)/n;
 return [c/prev-1,prev/p2-1,c/p5-1,c/mean(5)-1,c/mean(10)-1];
}
function sigmoid(z){return 1/(1+Math.exp(-Math.max(-30,Math.min(30,z))));}
function fit(X,y){
 const n=X.length,m=X[0].length,mu=Array(m).fill(0),sd=Array(m).fill(0);
 for(let j=0;j<m;j++){mu[j]=X.reduce((s,r)=>s+r[j],0)/n;sd[j]=Math.sqrt(X.reduce((s,r)=>s+(r[j]-mu[j])**2,0)/n)||1;}
 const Z=X.map(r=>r.map((v,j)=>(v-mu[j])/sd[j])),w=Array(m+1).fill(0),lr=.08,epochs=900;
 for(let e=0;e<epochs;e++){const g=Array(m+1).fill(0);for(let i=0;i<n;i++){const p=sigmoid(w[0]+Z[i].reduce((s,v,j)=>s+v*w[j+1],0)),d=p-y[i];g[0]+=d;for(let j=0;j<m;j++)g[j+1]+=d*Z[i][j];}for(let j=0;j<w.length;j++)w[j]-=lr*g[j]/n;}
 return {predict:r=>sigmoid(w[0]+r.reduce((s,v,j)=>s+((v-mu[j])/sd[j])*w[j+1],0))};
}
function runBacktest(data){
 const samples=[];for(let i=10;i<data.length-1;i++){const f=features(data,i),ret=data[i+1].close/data[i].close-1;samples.push({x:f,y:ret>0?1:0,ret});}
 const cut=Math.floor(samples.length*.7);if(cut<30||samples.length-cut<15)throw Error('Not enough usable rows after feature calculation. Add more history.');
 const model=fit(samples.slice(0,cut).map(s=>s.x),samples.slice(0,cut).map(s=>s.y));
 const capital=Number($('capital').value),costPct=Number($('cost').value);if(!(capital>0)||!(costPct>=0&&costPct<=2))throw Error('Check starting balance and cost input.');
 const cost=costPct/100;let equity=1,peak=1,maxDD=0,correct=0,prevPos=0,changes=0;const curve=[1],trades=[];
 for(const s of samples.slice(cut)){const p=model.predict(s.x),pos=p>.55?1:p<.45?-1:0;if((p>=.5?1:0)===s.y)correct++;if(pos!==prevPos){if(pos!==0||prevPos!==0){equity*=1-cost*Math.abs(pos-prevPos);changes++;}prevPos=pos;}equity*=1+pos*s.ret;peak=Math.max(peak,equity);maxDD=Math.max(maxDD,1-equity/peak);curve.push(equity);trades.push({p,pos,ret:s.ret});}
 const net=equity-1,accuracy=correct/(samples.length-cut);
 return {net,maxDD,accuracy,n:samples.length-cut,curve,capital,final:capital*equity,changes,train:cut,test:samples.length-cut};
}
function draw(curve){window.lastCurve=curve;const canvas=$('equity'),ctx=canvas.getContext('2d'),w=canvas.clientWidth,h=180,dpr=window.devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const lo=Math.min(...curve),hi=Math.max(...curve),range=hi-lo||.001;ctx.strokeStyle='#294158';ctx.lineWidth=1;for(let k=1;k<4;k++){const y=h*k/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}ctx.strokeStyle='#83e0bd';ctx.lineWidth=2;ctx.beginPath();curve.forEach((v,i)=>{const x=i/(curve.length-1)*(w-12)+6,y=h-12-(v-lo)/range*(h-24);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
function render(r){const fmt=x=>(x*100).toFixed(2)+'%';const vals=[fmt(r.net),fmt(-r.maxDD),fmt(r.accuracy),r.n];$('metrics').innerHTML=vals.map((v,i)=>'<article class="metric"><span>'+['Net return','Max drawdown','Directional accuracy','Test observations'][i]+'</span><strong>'+v+'</strong><small>'+['After assumed costs','Peak-to-trough decline','Test-set direction hit rate','Out-of-sample rows'][i]+'</small></article>').join('');$('detailRows').innerHTML=[['Starting paper balance','₹'+r.capital.toLocaleString('en-IN'),'Simulation only'],['Ending paper balance','₹'+r.final.toLocaleString('en-IN'),'After modeled returns and costs'],['Training / test rows',r.train+' / '+r.test,'Chronological 70% / 30% split'],['Position changes',r.changes,'Each change incurs assumed cost'],['Model','Logistic regression','Features use past candles only']].map(row=>'<tr>'+row.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');$('resultLabel').textContent='Backtest complete';draw(r.curve);}
fileInput.addEventListener('change',async()=>{const f=fileInput.files[0];if(!f)return;try{rows=parseCSV(await f.text());$('fileName').textContent=f.name+' · '+rows.length.toLocaleString()+' rows';runBtn.disabled=false;say('CSV validated. Ready to run a chronological holdout backtest.');}catch(e){rows=[];runBtn.disabled=true;say(e.message,true);}});
runBtn.addEventListener('click',()=>{try{if(!rows.length)throw Error('Import a CSV first.');const r=runBacktest(rows);render(r);say('Completed locally. This is a historical simulation, not a prediction or live trade.');}catch(e){say(e.message,true);}});
window.addEventListener('resize',()=>{const curve=window.lastCurve;if(curve)draw(curve);});
})();