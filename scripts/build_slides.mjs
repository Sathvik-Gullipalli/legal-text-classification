import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { FileBlob, Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SKILL = process.env.PRESENTATION_SKILL_DIR;
if (!SKILL) throw new Error('Set PRESENTATION_SKILL_DIR to the installed Codex presentation skill.');
const PY = process.env.ARTIFACT_PYTHON;
if (!PY) throw new Error('Set ARTIFACT_PYTHON to the bundled artifact runtime Python.');
const {resolvePresentationFont, applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const family = resolvePresentationFont({fontFamily:'Arial'});
const read = async f => JSON.parse(await fs.readFile(path.join(ROOT,f),'utf8'));
const m = await read('results/metrics.json');
const d = await read('results/dataset_summary.json');
const selection = await read('results/selection.json');
const snapshot = await read('results/demo_snapshot.json');
const winner = selection.winner;
const names = {majority:'Majority baseline',tfidf:'TF-IDF + LR',lda:'LDA + LR',doc2vec:'Doc2Vec + LR'};
const ink='#19394A', muted='#556C77', accent='#247C91', gold='#C39343', paper='#F7F6F2';
const presentation = Presentation.create({slideSize:{width:1280,height:720}});

function text(slide, value, left, top, width, height, size=28, bold=false, color=ink) {
 const shape = slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 shape.text=value;
 shape.text.style={typeface:family,fontSize:size,bold,color,autoFit:'none'};
 return shape;
}
function slide(title, notes='') {
 const s=presentation.slides.add(); s.background.fill=paper;
 text(s,title,64,42,1152,94,44,true);
 text(s,String(presentation.slides.items.length),1180,663,36,24,16,false,muted);
 s.speakerNotes.textFrame.setText(notes);
 return s;
}
function table(s, values, left, top, width, height, widths, size=25) {
 const t=s.tables.add({rows:values.length,columns:values[0].length,left,top,width,height,values,columnWidths:widths});
 t.borders.assign({style:'solid',fill:'#D5DFE2',width:1});
 for(let r=0;r<values.length;r++) for(let c=0;c<values[0].length;c++) {
  const cell=t.getCell(r,c); cell.fill=r===0?ink:(r%2?'#FFFFFF':'#EAF0F1');
  cell.text.style={typeface:family,fontSize:size,bold:r===0,color:r===0?'#FFFFFF':ink};
 }
 return t;
}
function chart(s,categories,series,left,top,width,height,direction='column') {
 const rounded=series.map(s=>({...s,values:s.values.map(v=>Number(v.toFixed(4)))}));
 const ch=s.charts.add('bar',{position:{left,top,width,height},categories,series:rounded,
  barOptions:{direction,grouping:'clustered',gapWidth:100},hasLegend:series.length>1,
  legend:{position:'bottom',textStyle:{fontSize:20,typeface:family}},
  xAxis:{textStyle:{fontSize:20,typeface:family}},
  yAxis:{textStyle:{fontSize:20,typeface:family}},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:19,typeface:family}},
  chartFill:'none',plotAreaFill:'none'});
 applyPresentationChartFont(ch,{fontFamily:family}); return ch;
}
const sources='Sources: Krithika Iyer, Classification of Legal Text (Stanford CS229, 2020), https://cs229.stanford.edu/proj2020spr/report/Iyer.pdf. Corpus: https://github.com/bdewilde/textacy-data/releases/tag/supreme_court_py3_v1.0. Official labels: https://github.com/chartbeat-labs/textacy/blob/main/src/textacy/datasets/supreme_court.py. Our measured evidence: results/dataset_summary.json, results/metrics.json, results/selection.json and saved split manifest.';

{
 const s=presentation.slides.add(); s.background.fill=ink;
 text(s,'Classification\nof Legal Text',72,150,1120,200,68,true,'#FFFFFF');
 text(s,'Machine Learning mini-project\nTeam 9 / Problem 5',76,384,1050,80,30,false,'#C1DAE2');
 text(s,'Sai Sathvik Gullipalli  /  PES2UG24CS430\nPriyanka Mallikarjuna Reddy  /  PES2UG24CS378',76,536,1090,90,26,false,'#FFFFFF');
 s.speakerNotes.textFrame.setText('Introduce the assigned problem and both team members. Explain that we classify issue areas from opinion text. '+sources);
}
{
 const s=slide('Legal opinion classification',sources);
 text(s,'Input',68,182,200,44,27,true,accent);
 text(s,'A full English US Supreme Court opinion',68,238,1110,62,36);
 text(s,'Output',68,332,200,44,27,true,accent);
 text(s,'One broad issue area, such as Criminal Procedure or Civil Rights',68,388,1110,108,36);
 text(s,'Research question',68,540,280,44,27,true,accent);
 text(s,'How do lexical features, topics, and document embeddings compare?',68,588,1110,60,29);
 s.speakerNotes.textFrame.setText('Classification supports cataloging and search. Our target is the issue area, not the case outcome. The Stanford study motivates the comparison. We add a TF-IDF baseline and independently evaluate the adapted methods. '+sources);
}
{
 const s=slide('Dataset and category imbalance',sources);
 text(s,d.raw_records.toLocaleString(),68,170,330,72,58,true,accent);
 text(s,'original opinions',68,252,330,45,26);
 text(s,d.retained_records.toLocaleString(),68,332,330,72,58,true,accent);
 text(s,`${d.present_labels.length} supported issue areas`,68,414,370,70,26);
 text(s,'1946–2016\nPrivate Action excluded: one case\n46 records excluded overall',68,522,350,120,23,false,muted);
 const entries=Object.entries(d.class_counts).sort((a,b)=>a[1]-b[1]);
 const ch=chart(s,entries.map(x=>x[0]),[{name:'Opinions',values:entries.map(x=>x[1]),fill:accent}],445,154,770,476,'bar');
 ch.xAxis={numberFormatCode:'0',textStyle:{fontSize:23,typeface:family}};
 ch.yAxis={textStyle:{fontSize:23,typeface:family}};
 s.speakerNotes.textFrame.setText(`The corpus contains ${d.raw_records} originals. We remove missing labels, empty text, duplicates, conflicting groups, and a category with insufficient groups. ${d.retained_records} remain. The class imbalance motivates macro-F1. ${sources}`);
}
{
 const s=slide('Preparation and evaluation pipeline',sources);
 const v=[['Stage','Operation'],['Clean and group','Link shared citations and normalized text hashes'],['Split once','5,023 train / 1,675 validation / 1,675 test'],['Learn representations','Fit vocabularies, LDA, and Doc2Vec on training only'],['Select and freeze','Validation macro-F1, then accuracy and latency'],['Evaluate','One held-out comparison of frozen models']];
 table(s,v,68,174,1142,360,[300,842],26);
 text(s,'Seed 42. Case groups never cross partitions.',68,564,1100,64,31,true,accent);
 s.speakerNotes.textFrame.setText('The union-find data preparation catches transitive duplicate links. Case-group overlap is asserted to be zero. Ratios are approximate because groups move together. Validation chooses configurations. We save selection before transforming test opinions. '+sources);
}
{
 const s=slide('Three representations, one classifier',sources);
 table(s,[['Representation','Features','Training choice'],['TF-IDF','Unigrams and bigrams','Up to 30,000 features'],['LDA','Topic proportions','20 or 40 topics, count input'],['Doc2Vec','Dense document vectors','100 dimensions, 30 epochs']],68,188,1142,282,[275,380,487],27);
 text(s,'Logistic Regression: C = 0.1, 1, 10',68,516,1110,55,34,true,accent);
 text(s,'LDA uses counts. Doc2Vec uses inferred vectors consistently across all partitions.',68,592,1110,70,26,false,muted);
 s.speakerNotes.textFrame.setText('TF-IDF captures words and phrases. LDA discovers latent topics without labels. Doc2Vec learns a document embedding. LR learns the supervised mapping to issue areas. C controls inverse regularization. Counts are a documented adaptation from the reference. One-worker Doc2Vec and seeded inference support reproducibility. '+sources);
}
{
 const s=slide('Held-out model comparison',sources);
 const v=[['Model','Accuracy','Macro-F1'],...Object.entries(m).map(([n,x])=>[names[n],x.accuracy.toFixed(3),x.macro_f1.toFixed(3)])];
 table(s,v,68,183,558,330,[258,150,150],25);
 const cats=['Majority','TF-IDF','LDA','Doc2Vec'];
 const ch=chart(s,cats,[{name:'Accuracy',values:Object.values(m).map(x=>x.accuracy),fill:accent,valuesFormatCode:'0%'},{name:'Macro-F1',values:Object.values(m).map(x=>x.macro_f1),fill:gold,valuesFormatCode:'0%'}],650,160,558,380);
 ch.yAxis={min:0,max:1,numberFormatCode:'0%',textStyle:{fontSize:20,typeface:family}};
 ch.dataLabels={showValue:true,position:'outEnd',textStyle:{fontSize:17,typeface:family}};
 text(s,`Validation-selected model: ${names[winner]}`,68,560,1120,55,32,true,accent);
 text(s,'Macro-F1 gives each category equal weight. Test scores never select the model.',68,618,1100,52,24,false,muted);
 s.speakerNotes.textFrame.setText(`Validation selects ${winner} with macro-F1 ${selection.models[winner].macro_f1.toFixed(4)}. Test contains ${d.splits.test.records} complete opinions. We report accuracy and macro-F1 because classes are imbalanced. The chart and table use our own frozen test evidence, not scores copied from the paper. ${sources}`);
}
{
 const s=slide('Saved-model notebook demonstration',sources);
 text(s,`Example: ${snapshot.true_category}`,68,167,1110,52,30,true,accent);
 text(s,`TF-IDF prediction: ${snapshot.prediction.category}`,68,230,1110,66,37,true);
 table(s,[['Top category','Model probability'],...snapshot.prediction.top_three.map(r=>[r.category,(100*r.probability).toFixed(1)+'%'])],68,324,560,245,[365,195],25);
 text(s,'Similar training opinions',676,326,520,52,29,true,accent);
 text(s,snapshot.neighbors.slice(0,3).map(r=>r.case_name.length>56?r.case_name.slice(0,53)+'…':r.case_name).join('\n\n'),676,393,520,220,24);
 text(s,'No retraining. Probabilities are uncalibrated. Similarity does not establish legal authority.',68,626,1130,48,23,false,muted);
 s.speakerNotes.textFrame.setText(`Open 02_live_demo.ipynb, select the held-out Criminal Procedure example and TF-IDF, then press the button. Explain the top-three probabilities and influential terms. Switch models. Show five similar training opinions. The slide uses the actual saved-example output: ${snapshot.case_name}. ${sources}`);
}
{
 const s=slide('Conclusions and limitations',sources);
 text(s,`${names[winner]} reaches ${(100*m[winner].accuracy).toFixed(1)}% test accuracy`,68,178,1140,108,42,true,accent);
 text(s,'Our comparison supports classification and qualitative opinion retrieval on this historical corpus.',68,314,1130,108,32);
 text(s,'Future-case generalization needs a temporal evaluation.\nSmall categories and source-label noise limit reliability.\nShort passages differ from full opinions.\nThe US corpus does not establish performance on Indian law.',68,463,1130,166,26,false,muted);
 s.speakerNotes.textFrame.setText('Conclude using the validation-selected model and its measured held-out result. Explain why a grouped random split is not a future-case evaluation. Mention imbalance, noisy source alignment, and short-text mismatch. Potential extensions are temporal evaluation and probability calibration. '+sources);
}

const staging=path.join(ROOT,'.build/slides');
await fs.mkdir(staging,{recursive:true});
const candidate=path.join(staging,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const finalPath=path.join(ROOT,'submission/team9_presentation.pptx');
await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath,
 pythonExecutable:PY, integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...[4,5,6,7].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:8, requiredNativeTableOwnerSlides:[4,5,6,7],requiredNativeChartOwnerSlides:[3,6],
 materializeLiteralChartWorkbooks:true, fontPolicy:{basis:'design',families:[family]},verifyArtifactToolImport:true,
 receiptPath:path.join(staging,'validation.json')});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<finalDeck.slides.items.length;i++) {
 const slide=finalDeck.slides.items[i];
 const preview=await finalDeck.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(staging,`slide-${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));
}
console.log('Created and validated '+finalPath+' using '+family);
