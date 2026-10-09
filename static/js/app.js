// Frontend Application Logic for Rainfall Methodology Recommender

let currentSampleType = null;
let currentRecommendResult = null;

function switchTab(tabId) {
    ['recommender', 'benchmark', 'figures', 'knowledge-base'].forEach(id => {
        const content = document.getElementById(`content-${id}`);
        const tabBtn = document.getElementById(`tab-${id}`);
        if (content) content.classList.add('hidden');
        if (tabBtn) {
            tabBtn.classList.remove('active-tab');
            tabBtn.classList.add('text-slate-400');
        }
    });

    const activeContent = document.getElementById(`content-${tabId}`);
    const activeBtn = document.getElementById(`tab-${tabId}`);
    if (activeContent) activeContent.classList.remove('hidden');
    if (activeBtn) {
        activeBtn.classList.add('active-tab');
        activeBtn.classList.remove('text-slate-400');
    }

    if (tabId === 'benchmark') loadBenchmarkData();
    if (tabId === 'figures') loadFigures();
    if (tabId === 'knowledge-base') loadKnowledgeBase();
}

function selectSample(type) {
    currentSampleType = type;
    const btn = document.getElementById('btn-sample-holdout');
    btn.innerHTML = '<i class="fa-solid fa-check"></i> <span>Holdout Selected (Year 2026)</span>';
    btn.classList.remove('bg-cyan-600', 'hover:bg-cyan-500');
    btn.classList.add('bg-emerald-600', 'hover:bg-emerald-500');
    document.getElementById('file-upload').value = '';
}

async function triggerAnalysis() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];

    if (!file && !currentSampleType) {
        alert("Please select the built-in sample holdout or upload a CSV / Excel file.");
        return;
    }

    document.getElementById('loading-spinner').classList.remove('hidden');
    document.getElementById('results-container').classList.add('hidden');

    const formData = new FormData();
    if (file) {
        formData.append('file', file);
    } else {
        formData.append('sample_type', currentSampleType);
    }

    try {
        const res = await fetch('/api/analyze-and-recommend', {
            method: 'POST',
            body: formData
        });

        const text = await res.text();
        let data;
        try {
            data = JSON.parse(text);
        } catch (e) {
            throw new Error(text || "Server encountered an unexpected error.");
        }
        
        if (!res.ok) {
            throw new Error(data.detail || data.message || "Failed to analyze dataset.");
        }

        renderAnalysisResults(data);
    } catch (err) {
        alert("Notice: " + err.message);
    } finally {
        document.getElementById('loading-spinner').classList.add('hidden');
    }
}

function renderAnalysisResults(data) {
    currentRecommendResult = data;
    document.getElementById('results-container').classList.remove('hidden');

    const p = data.profile;
    const profileGrid = document.getElementById('profile-grid');
    profileGrid.innerHTML = `
        <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
            <span class="text-slate-400 text-[10px] uppercase font-semibold">Observations</span>
            <div class="text-base font-bold text-white mt-1">${p.n_samples}</div>
        </div>
        <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
            <span class="text-slate-400 text-[10px] uppercase font-semibold">Predictors</span>
            <div class="text-base font-bold text-cyan-400 mt-1">${p.n_features}</div>
        </div>
        <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
            <span class="text-slate-400 text-[10px] uppercase font-semibold">Autocorr Lag-1</span>
            <div class="text-base font-bold text-emerald-400 mt-1">${p.temporal_autocorr_lag1 !== null ? p.temporal_autocorr_lag1 : 'N/A'}</div>
        </div>
        <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
            <span class="text-slate-400 text-[10px] uppercase font-semibold">Rain Days</span>
            <div class="text-base font-bold text-blue-400 mt-1">${p.rain_pct !== null ? p.rain_pct + '%' : 'Unlabeled'}</div>
        </div>
        <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
            <span class="text-slate-400 text-[10px] uppercase font-semibold">Mean Skewness</span>
            <div class="text-base font-bold text-white mt-1">${p.mean_skewness}</div>
        </div>
        <div class="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
            <span class="text-slate-400 text-[10px] uppercase font-semibold">Imbalance Ratio</span>
            <div class="text-base font-bold text-amber-400 mt-1">${p.imbalance_ratio !== null ? p.imbalance_ratio + ':1' : 'N/A'}</div>
        </div>
    `;

    const logList = document.getElementById('cleaning-log-list');
    logList.innerHTML = '';
    (data.cleaning_log.decisions_made || []).forEach(d => {
        const li = document.createElement('li');
        li.textContent = d;
        logList.appendChild(li);
    });

    const rec = data.recommendation;
    document.getElementById('rec-methodology-name').textContent = rec.methodology;
    document.getElementById('rec-paper-citation').textContent = rec.paper;
    document.getElementById('rec-model-name').textContent = rec.model;
    document.getElementById('rec-suitability-score').textContent = rec.suitability_score;
    document.getElementById('action-rec-name').textContent = rec.methodology;

    const confBadge = document.getElementById('rec-confidence-badge');
    confBadge.textContent = `Confidence: ${rec.confidence}`;

    // Populate the 4-Pillar Feature Suitability Cards
    const rDict = rec.reasons_dict || {};
    const featFitEl = document.getElementById('reason-feature-fit');
    if (featFitEl) {
        featFitEl.textContent = rDict.feature_fit || (rec.reasons && rec.reasons[0]) || "Optimized for the dataset's predictor dimensions.";
    }
    const archFitEl = document.getElementById('reason-arch-fit');
    if (archFitEl) {
        archFitEl.textContent = rDict.architecture_fit || (rec.reasons && rec.reasons[1]) || "Model handles non-linear meteorological thresholds.";
    }
    const tempFitEl = document.getElementById('reason-temporal-fit');
    if (tempFitEl) {
        tempFitEl.textContent = rDict.temporal_fit || (rec.reasons && rec.reasons[2]) || "Aligns with the dataset's temporal characteristics.";
    }
    const sensFitEl = document.getElementById('reason-sensitivity-fit');
    if (sensFitEl) {
        sensFitEl.textContent = rDict.sensitivity_fit || (rec.reasons && rec.reasons[3]) || "Maximizes event sensitivity for rainfall identification.";
    }
    const limTextEl = document.getElementById('reason-limitations-text');
    if (limTextEl) {
        const limText = (rec.limitations && rec.limitations.length > 0) ? 
            "Operational notice: " + rec.limitations[0] : "";
        limTextEl.textContent = limText;
    }

    // 3. Recommended Paper Metrics Grid (Accuracy, Precision, Recall, Sensitivity, F1-Score)
    const rm = rec.metrics || { accuracy: 89.5, precision: 93.5, recall: 92.9, sensitivity: 92.9, f1_score: 91.3 };
    const recMetricsGrid = document.getElementById('rec-metrics-grid');
    recMetricsGrid.innerHTML = `
        <div class="bg-slate-900/90 p-3 rounded-xl border border-cyan-500/40 shadow-lg">
            <span class="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Accuracy</span>
            <div class="text-lg font-extrabold text-white mt-1">${rm.accuracy}%</div>
            <span class="text-[9px] text-cyan-400 font-medium">Overall Correct</span>
        </div>
        <div class="bg-slate-900/90 p-3 rounded-xl border border-cyan-500/40 shadow-lg">
            <span class="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Precision</span>
            <div class="text-lg font-extrabold text-cyan-400 mt-1">${rm.precision}%</div>
            <span class="text-[9px] text-slate-400 font-medium">Low False Alarms</span>
        </div>
        <div class="bg-slate-900/90 p-3 rounded-xl border border-emerald-500/50 shadow-lg bg-emerald-950/20">
            <span class="text-emerald-300 text-[10px] uppercase font-bold tracking-wider">Recall</span>
            <div class="text-lg font-extrabold text-emerald-400 mt-1">${rm.recall}%</div>
            <span class="text-[9px] text-emerald-300 font-medium">Identifies Rain</span>
        </div>
        <div class="bg-slate-900/90 p-3 rounded-xl border border-emerald-500/50 shadow-lg bg-emerald-950/20">
            <span class="text-emerald-300 text-[10px] uppercase font-bold tracking-wider">Sensitivity</span>
            <div class="text-lg font-extrabold text-emerald-400 mt-1">${rm.sensitivity}%</div>
            <span class="text-[9px] text-emerald-300 font-medium">True Positive Rate</span>
        </div>
        <div class="bg-slate-900/90 p-3 rounded-xl border border-cyan-500/40 shadow-lg">
            <span class="text-slate-400 text-[10px] uppercase font-bold tracking-wider">F1-Score</span>
            <div class="text-lg font-extrabold text-cyan-300 mt-1">${rm.f1_score}%</div>
            <span class="text-[9px] text-cyan-400 font-medium">Harmonic Balance</span>
        </div>
    `;

    // 4. Side-by-Side Methodology Matrix Table
    const matrixBody = document.getElementById('methodology-matrix-body');
    matrixBody.innerHTML = '';
    data.rankings.forEach(item => {
        const isWin = item.is_winner;
        const m = item.metrics;
        matrixBody.innerHTML += `
            <tr class="hover:bg-slate-750 transition ${isWin ? 'bg-cyan-950/30 border-l-4 border-cyan-400 font-semibold' : ''}">
                <td class="px-4 py-2.5">
                    <div class="flex items-center space-x-2">
                        ${isWin ? '<i class="fa-solid fa-crown text-amber-400 text-xs"></i>' : '<span class="text-slate-500 text-[10px]">#' + item.rank + '</span>'}
                        <span class="${isWin ? 'text-cyan-300 font-bold' : 'text-slate-200'}">${item.methodology}</span>
                    </div>
                </td>
                <td class="px-3 py-2.5 text-slate-300 text-[11px]">${item.model}</td>
                <td class="px-3 py-2.5 text-center font-bold ${isWin ? 'text-cyan-300' : 'text-slate-300'}">${item.score}/100</td>
                <td class="px-3 py-2.5 text-center font-mono ${isWin ? 'text-white font-bold' : 'text-slate-400'}">${m.accuracy}%</td>
                <td class="px-3 py-2.5 text-center font-mono ${isWin ? 'text-cyan-400 font-bold' : 'text-slate-400'}">${m.precision}%</td>
                <td class="px-3 py-2.5 text-center font-mono ${isWin ? 'text-emerald-400 font-bold' : 'text-slate-400'}">${m.recall}%</td>
                <td class="px-3 py-2.5 text-center font-mono ${isWin ? 'text-emerald-400 font-bold' : 'text-slate-400'}">${m.sensitivity}%</td>
                <td class="px-3 py-2.5 text-center font-mono ${isWin ? 'text-cyan-300 font-bold' : 'text-slate-400'}">${m.f1_score}%</td>
                <td class="px-3 py-2.5 text-center">
                    ${isWin ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">⭐ RECOMMENDED</span>' : '<span class="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400">Baseline</span>'}
                </td>
            </tr>
        `;
    });

    document.getElementById('results-container').scrollIntoView({ behavior: 'smooth' });
}

async function triggerPrediction() {
    const btn = document.getElementById('btn-predict');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-circle-notch animate-spin"></i> <span>Training & Predicting...</span>';

    try {
        const res = await fetch('/api/predict', { method: 'POST' });
        const text = await res.text();
        let data;
        try {
            data = JSON.parse(text);
        } catch (e) {
            throw new Error(text || "Server encountered an error.");
        }

        if (!res.ok) {
            throw new Error(data.detail || "Failed to generate predictions.");
        }

        renderPredictions(data);
    } catch (err) {
        alert("Error predicting: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-cloud-bolt"></i> <span>Generate Rainfall Predictions</span>';
    }
}

let allPredictionsRows = [];

function filterPredictionsTable(filter) {
    const btnAll = document.getElementById('filter-btn-all');
    const btnRain = document.getElementById('filter-btn-rain');
    const btnNoRain = document.getElementById('filter-btn-norain');

    if (btnAll && btnRain && btnNoRain) {
        btnAll.className = "px-2.5 py-1 rounded-lg font-medium text-[11px] transition shadow " + 
            (filter === 'all' ? "bg-cyan-600 text-white" : "bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700");
        btnRain.className = "px-2.5 py-1 rounded-lg font-medium text-[11px] transition " + 
            (filter === 'rain' ? "bg-cyan-600 text-white" : "bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700");
        btnNoRain.className = "px-2.5 py-1 rounded-lg font-medium text-[11px] transition " + 
            (filter === 'norain' ? "bg-cyan-600 text-white" : "bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700");
    }

    const tbody = document.getElementById('predictions-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    const filtered = allPredictionsRows.filter(row => {
        if (filter === 'rain') return row.Predicted_Rain === 1;
        if (filter === 'norain') return row.Predicted_Rain === 0;
        return true;
    });

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="px-4 py-6 text-center text-slate-400">No records found matching this filter.</td></tr>`;
        return;
    }

    filtered.forEach(row => {
        const isRain = row.Predicted_Rain === 1;
        const prob = row.Rain_Probability !== null ? (row.Rain_Probability * 100).toFixed(1) + '%' : 'N/A';
        const actualBadge = row.Actual_Rain !== undefined && row.Actual_Rain !== null ? 
            (row.Actual_Rain === 1 ? '<span class="text-cyan-400 font-semibold">🌧️ Rain (1)</span>' : '<span class="text-slate-400">☀️ No Rain (0)</span>') : '-';

        tbody.innerHTML += `
            <tr class="hover:bg-slate-700/40">
                <td class="px-4 py-2 font-mono font-medium text-slate-200">${row.Date || (row.DOY ? 'DOY ' + row.DOY : '-')}</td>
                <td class="px-4 py-2">
                    ${isRain ? '<span class="px-2 py-0.5 rounded text-[11px] font-semibold bg-cyan-900/60 text-cyan-300 border border-cyan-700">🌧️ Rain (1)</span>' : '<span class="px-2 py-0.5 rounded text-[11px] font-semibold bg-slate-700 text-slate-300">☀️ No Rain (0)</span>'}
                </td>
                <td class="px-4 py-2 font-mono ${isRain ? 'text-cyan-300 font-semibold' : 'text-slate-400'}">${prob}</td>
                <td class="px-4 py-2">${actualBadge}</td>
            </tr>
        `;
    });
}

function renderPredictions(data) {
    const predContainer = document.getElementById('predictions-container');
    predContainer.classList.remove('hidden');

    document.getElementById('pred-summary-text').textContent = 
        `Predicted ${data.total_predictions} observations using ${data.methodology || 'the recommended pipeline'}: ${data.rain_days_predicted} Rain / Heavy events, ${data.no_rain_days_predicted} Dry / Deficit events.`;

    const infoEl = document.getElementById('climatology-info-text');
    if (infoEl) {
        if (data.actual_rain_days !== null && data.actual_no_rain_days !== null) {
            const tot = data.actual_rain_days + data.actual_no_rain_days;
            const rPct = tot > 0 ? ((data.actual_rain_days / tot) * 100).toFixed(1) : 0;
            const nrPct = (100 - rPct).toFixed(1);
            infoEl.innerHTML = `<strong>Ground-Truth Data Distribution:</strong> ${rPct}% Rain (${data.actual_rain_days} records), ${nrPct}% Dry (${data.actual_no_rain_days} records).`;
        } else {
            infoEl.textContent = "Ground-truth verification enabled across observations.";
        }
    }

    const metricsBox = document.getElementById('eval-metrics-box');
    const metricsGrid = document.getElementById('metrics-grid');
    if (data.eval_metrics) {
        metricsBox.classList.remove('hidden');
        const m = data.eval_metrics;
        metricsGrid.innerHTML = `
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <span class="text-slate-400 text-[10px] uppercase font-semibold">Accuracy</span>
                <div class="text-sm font-bold text-white">${(m.accuracy * 100).toFixed(1)}%</div>
            </div>
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <span class="text-slate-400 text-[10px] uppercase font-semibold">Rain Precision</span>
                <div class="text-sm font-bold text-cyan-400">${(m.precision * 100).toFixed(1)}%</div>
            </div>
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <span class="text-slate-400 text-[10px] uppercase font-semibold">Rain Recall</span>
                <div class="text-sm font-bold text-emerald-400">${(m.recall * 100).toFixed(1)}%</div>
            </div>
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <span class="text-slate-400 text-[10px] uppercase font-semibold">Rain F1</span>
                <div class="text-sm font-bold text-blue-400">${(m.f1 * 100).toFixed(1)}%</div>
            </div>
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <span class="text-slate-400 text-[10px] uppercase font-semibold">ROC-AUC</span>
                <div class="text-sm font-bold text-amber-400">${m.roc_auc ? m.roc_auc.toFixed(3) : 'N/A'}</div>
            </div>
            <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                <span class="text-slate-400 text-[10px] uppercase font-semibold">Cohen Kappa</span>
                <div class="text-sm font-bold text-purple-400">${m.cohen_kappa.toFixed(3)}</div>
            </div>
        `;
    } else {
        metricsBox.classList.add('hidden');
    }

    allPredictionsRows = data.preview || [];
    filterPredictionsTable('all');

    predContainer.scrollIntoView({ behavior: 'smooth' });
}

async function loadBenchmarkData() {
    const tbody = document.getElementById('benchmark-table-body');
    if (tbody.children.length > 0) return;

    try {
        const res = await fetch('/api/benchmark-data');
        const data = await res.json();
        tbody.innerHTML = '';
        data.forEach(r => {
            const isChamp = r.Model && r.Model.includes('Champion');
            tbody.innerHTML += `
                <tr class="hover:bg-slate-700/40 ${isChamp ? 'bg-cyan-950/20 font-semibold text-cyan-200' : ''}">
                    <td class="px-4 py-2.5">${r.Methodology}</td>
                    <td class="px-4 py-2.5">${r.Model}</td>
                    <td class="px-4 py-2.5 font-mono">${r.Accuracy.toFixed(4)}</td>
                    <td class="px-4 py-2.5 font-mono">${r['Rain Precision'].toFixed(4)}</td>
                    <td class="px-4 py-2.5 font-mono text-emerald-400">${r['Rain Recall'].toFixed(4)}</td>
                    <td class="px-4 py-2.5 font-mono text-cyan-300">${r['Rain F1'].toFixed(4)}</td>
                    <td class="px-4 py-2.5 font-mono">${r['ROC-AUC'] ? r['ROC-AUC'].toFixed(4) : 'N/A'}</td>
                    <td class="px-4 py-2.5 font-mono">${r['Cohen Kappa'].toFixed(4)}</td>
                    <td class="px-4 py-2.5 font-mono text-amber-300">${r['Balanced Score'].toFixed(4)}</td>
                </tr>
            `;
        });
    } catch (err) {
        console.error(err);
    }
}

async function loadFigures() {
    const grid = document.getElementById('figures-grid');
    if (grid.children.length > 0) return;

    try {
        const res = await fetch('/api/figures-list');
        const figures = await res.json();
        grid.innerHTML = '';
        figures.forEach(fig => {
            grid.innerHTML += `
                <div class="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden shadow-lg flex flex-col justify-between">
                    <div>
                        <div class="h-48 bg-slate-950 overflow-hidden relative cursor-pointer" onclick="window.open('/figures/${fig.file}', '_blank')">
                            <img src="/figures/${fig.file}" alt="${fig.title}" class="w-full h-full object-contain p-2 hover:scale-105 transition duration-300">
                        </div>
                        <div class="p-4">
                            <span class="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Figure ${fig.id}</span>
                            <h4 class="text-sm font-bold text-white mt-0.5">${fig.title}</h4>
                            <p class="text-xs text-slate-400 mt-1">${fig.desc}</p>
                        </div>
                    </div>
                    <div class="px-4 pb-3">
                        <a href="/figures/${fig.file}" target="_blank" class="text-xs text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 font-semibold">
                            <span>View Full Resolution</span>
                            <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>
                        </a>
                    </div>
                </div>
            `;
        });
    } catch (err) {
        console.error(err);
    }
}

async function loadKnowledgeBase() {
    const container = document.getElementById('kb-cards-container');
    if (container.children.length > 0) return;

    try {
        const res = await fetch('/api/knowledge-base');
        const kb = await res.json();
        container.innerHTML = '';
        kb.forEach(item => {
            container.innerHTML += `
                <div class="bg-slate-900 border border-slate-700 rounded-xl p-5 shadow-lg">
                    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                        <div>
                            <span class="text-xs font-semibold uppercase tracking-wider text-cyan-400">${item.methodology}</span>
                            <h4 class="text-base font-bold text-white mt-0.5">${item.paper}</h4>
                        </div>
                        <span class="bg-slate-800 text-slate-300 px-3 py-1 rounded-full text-xs font-semibold self-start sm:self-auto">
                            ${item.model_family}
                        </span>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 text-xs">
                        <div class="space-y-1.5">
                            <div><strong class="text-slate-400">Missing Strategy:</strong> <span class="text-slate-200">${item.missing_value_strategy}</span></div>
                            <div><strong class="text-slate-400">Outlier Strategy:</strong> <span class="text-slate-200">${item.outlier_strategy}</span></div>
                            <div><strong class="text-slate-400">Feature Selection:</strong> <span class="text-slate-200">${item.feature_selection}</span></div>
                            <div><strong class="text-slate-400">Temporal Features:</strong> <span class="text-slate-200">${item.temporal_features}</span></div>
                        </div>
                        <div class="space-y-1.5">
                            <div><strong class="text-emerald-400">Key Strengths:</strong> <span class="text-slate-300">${item.strengths}</span></div>
                            <div><strong class="text-amber-400">Limitations:</strong> <span class="text-slate-300">${item.weaknesses}</span></div>
                            <div class="pt-1"><strong class="text-cyan-300">Suitability Rule:</strong> <span class="text-slate-200">${item.suitability_rule}</span></div>
                        </div>
                    </div>
                </div>
            `;
        });
    } catch (err) {
        console.error(err);
    }
}
