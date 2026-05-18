HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>GradSync Telemetry</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: monospace; padding: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: #161b22; padding: 20px; border-radius: 8px; border: 1px solid #30363d; }
        canvas { max-height: 250px; }
        h2 { margin-top: 0; color: #58a6ff; font-size: 1.2rem; }
    </style>
</head>
<body>
    <h1>GradSync Cluster Telemetry</h1>
    <div class="grid">
        <div class="card">
            <h2>Cluster VRAM (GB)</h2>
            <canvas id="vramChart"></canvas>
        </div>
        <div class="card">
            <h2>Training Loss</h2>
            <canvas id="lossChart"></canvas>
        </div>
        <div class="card">
            <h2>Forward Pass Time (ms)</h2>
            <canvas id="fwChart"></canvas>
        </div>
        <div class="card">
            <h2>Backward Pass Time (ms)</h2>
            <canvas id="bwChart"></canvas>
        </div>
    </div>
    <script>
        const colors = ['#ff7b72', '#d2a8ff', '#79c0ff', '#a5d6ff', '#3fb950', '#f2cc60', '#ff9e64', '#e3b341'];
        
        function createBaseChart(ctxId, isLog = false) {
            return new Chart(document.getElementById(ctxId), {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: { 
                    responsive: true, animation: false, 
                    scales: { x: { display: false }, y: isLog ? { type: 'logarithmic' } : {} } 
                }
            });
        }

        const vramChart = createBaseChart('vramChart');
        const fwChart = createBaseChart('fwChart');
        const bwChart = createBaseChart('bwChart');
        
        const lossChart = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#3fb950', tension: 0.2, spanGaps: true }] },
            options: { responsive: true, animation: false, scales: { x: { display: false }, y: { type: 'logarithmic' } } }
        });

        function updateNodeDatasets(chart, dataDict, labels) {
            Object.keys(dataDict).forEach(nodeId => {
                let ds = chart.data.datasets.find(d => d.label === `Node ${nodeId}`);
                if (!ds) {
                    const c = colors[parseInt(nodeId) % colors.length] || '#ffffff';
                    ds = { label: `Node ${nodeId}`, data: Array(Math.max(0, labels.length - 1)).fill(null), borderColor: c, tension: 0.2, spanGaps: true };
                    chart.data.datasets.push(ds);
                }
            });
            chart.data.datasets.forEach(ds => {
                const nId = ds.label.replace('Node ', '');
                ds.data.push(dataDict[nId] !== undefined ? dataDict[nId] : null);
            });
        }

        const ws = new WebSocket("ws://" + location.host + "/ws");
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const step = data.step;
            
            if (!vramChart.data.labels.includes(step)) {
                vramChart.data.labels.push(step);
                lossChart.data.labels.push(step);
                fwChart.data.labels.push(step);
                bwChart.data.labels.push(step);
                
                const plotLoss = (data.loss && data.loss > 0) ? data.loss : null;
                lossChart.data.datasets[0].data.push(plotLoss);
            }
            
            updateNodeDatasets(vramChart, data.vram, vramChart.data.labels);
            updateNodeDatasets(fwChart, data.fw_time, fwChart.data.labels);
            updateNodeDatasets(bwChart, data.bw_time, bwChart.data.labels);
            
            vramChart.update(); lossChart.update(); fwChart.update(); bwChart.update();
        };
    </script>
</body>
</html>
"""