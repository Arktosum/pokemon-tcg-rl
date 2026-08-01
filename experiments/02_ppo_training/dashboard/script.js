// Common Chart.js options for dark theme
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.font.family = "'Inter', sans-serif";

const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 }, // Disable animation for live updates to prevent flickering
    plugins: {
        legend: {
            position: 'top',
            labels: { boxWidth: 12, usePointStyle: true }
        },
        tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(15, 23, 42, 0.9)',
            titleColor: '#f8fafc',
            bodyColor: '#e2e8f0',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1
        }
    },
    scales: {
        x: { type: 'linear', grid: { display: false } },
        y: { beginAtZero: false }
    },
    elements: {
        point: { radius: 0, hitRadius: 10, hoverRadius: 4 },
        line: { tension: 0.2, borderWidth: 2 }
    }
};

// Initialize Charts
const winRateCtx = document.getElementById('winRateChart').getContext('2d');
const smoothedWinRateCtx = document.getElementById('smoothedWinRateChart').getContext('2d');
const avgWinRateCtx = document.getElementById('avgWinRateChart').getContext('2d');
const rewardCtx = document.getElementById('rewardChart').getContext('2d');
const lengthCtx = document.getElementById('lengthChart').getContext('2d');
const lossCtx = document.getElementById('lossChart').getContext('2d');

const opponentColors = {
    'random_agent': '#3b82f6', // blue
    'RandomBot': '#3b82f6', // blue (old name)
    'greedy_agent': '#10b981', // green
    'tactical_agent': '#f59e0b', // yellow
    'setup_agent': '#8b5cf6', // purple
    'self_play_agent': '#ef4444', // red
    'aggro_agent': '#ec4899', // pink
    'heuristic_agent': '#0ea5e9' // sky
};

// Global Chart Instances
let charts = {};

function initCharts() {
    charts.winRate = new Chart(winRateCtx, {
        type: 'scatter',
        data: { datasets: [] },
        options: {
            ...commonOptions,
            scales: {
                y: { ...commonOptions.scales.y, min: 0, max: 100, title: { display: true, text: 'Win Rate %' } },
                x: { ...commonOptions.scales.x, title: { display: true, text: 'Update Count' } }
            }
        }
    });

    charts.smoothedWinRate = new Chart(smoothedWinRateCtx, {
        type: 'line',
        data: { datasets: [] },
        options: {
            ...commonOptions,
            scales: {
                y: { ...commonOptions.scales.y, min: 0, max: 100, title: { display: true, text: 'Moving Avg Win Rate %' } },
                x: { ...commonOptions.scales.x, title: { display: true, text: 'Update Count' } }
            }
        }
    });

    charts.avgWinRate = new Chart(avgWinRateCtx, {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            ...commonOptions,
            scales: {
                y: { ...commonOptions.scales.y, min: 0, max: 100, title: { display: true, text: 'Overall Win Rate %' } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    charts.reward = new Chart(rewardCtx, {
        type: 'line',
        data: { datasets: [] },
        options: {
            ...commonOptions,
            scales: {
                y: { ...commonOptions.scales.y, title: { display: true, text: 'Avg Reward' } },
                x: { ...commonOptions.scales.x }
            }
        }
    });

    charts.length = new Chart(lengthCtx, {
        type: 'line',
        data: { datasets: [] },
        options: {
            ...commonOptions,
            scales: {
                y: { ...commonOptions.scales.y, beginAtZero: true, title: { display: true, text: 'Avg Episode Length' } },
                x: { ...commonOptions.scales.x }
            }
        }
    });

    charts.loss = new Chart(lossCtx, {
        type: 'line',
        data: { datasets: [] },
        options: {
            ...commonOptions,
            scales: {
                y: { ...commonOptions.scales.y, type: 'logarithmic', title: { display: true, text: 'Loss / Entropy' } },
                x: { ...commonOptions.scales.x }
            }
        }
    });
}

function processMetrics(data) {
    if (!data || data.length === 0) return;

    // Fix historical naming inconsistency
    data = data.map(d => {
        if (d.opponent === 'RandomBot') d.opponent = 'random_agent';
        return d;
    });

    // Update summary cards
    const latest = data[data.length - 1];
    document.getElementById('summary-steps').textContent = latest.total_steps ? latest.total_steps.toLocaleString() : '--';
    document.getElementById('summary-update').textContent = latest.update_count || '--';

    // Group data by opponent for Win Rate, Reward, Length
    const opponents = [...new Set(data.map(d => d.opponent).filter(Boolean))];
    
    const datasetsWin = [];
    const datasetsSmoothed = [];
    const datasetsReward = [];
    const datasetsLength = [];
    
    const avgWinRateLabels = [];
    const avgWinRateData = [];
    const avgWinRateColors = [];

    opponents.forEach(opp => {
        const oppData = data.filter(d => d.opponent === opp);
        const color = opponentColors[opp] || '#cbd5e1';
        
        const mappedData = oppData.map(d => ({ x: d.update_count, y: d.win_rate }));
        datasetsWin.push({ label: opp, data: mappedData, type: 'line', spanGaps: true, radius: 3, borderColor: color, backgroundColor: color });
        
        const windowSize = 5;
        const smoothedData = [];
        for (let i = 0; i < mappedData.length; i++) {
            let sum = 0;
            let count = 0;
            for (let j = Math.max(0, i - windowSize + 1); j <= i; j++) {
                sum += mappedData[j].y;
                count++;
            }
            smoothedData.push({ x: mappedData[i].x, y: sum / count });
        }
        datasetsSmoothed.push({ label: opp, data: smoothedData, type: 'line', spanGaps: true, radius: 0, tension: 0.4, borderColor: color, backgroundColor: color });

        if (mappedData.length > 0) {
            const overallAvg = mappedData.reduce((acc, curr) => acc + curr.y, 0) / mappedData.length;
            avgWinRateLabels.push(opp);
            avgWinRateData.push(overallAvg);
            avgWinRateColors.push(color);
        }

        datasetsReward.push({ label: opp, data: oppData.map(d => ({ x: d.update_count, y: d.avg_reward })), borderColor: color, backgroundColor: color });
        datasetsLength.push({ label: opp, data: oppData.map(d => ({ x: d.update_count, y: d.avg_length })), borderColor: color, backgroundColor: color });
    });

    // Update opponent-based charts
    charts.winRate.data = { datasets: datasetsWin };
    charts.winRate.update();

    charts.smoothedWinRate.data = { datasets: datasetsSmoothed };
    charts.smoothedWinRate.update();

    charts.avgWinRate.data = {
        labels: avgWinRateLabels,
        datasets: [{
            data: avgWinRateData,
            backgroundColor: avgWinRateColors,
            borderColor: avgWinRateColors,
            borderWidth: 1
        }]
    };
    charts.avgWinRate.update();

    charts.reward.data = { datasets: datasetsReward };
    charts.reward.update();

    charts.length.data = { datasets: datasetsLength };
    charts.length.update();

    // Overall metrics for Loss & Entropy
    const actorLossData = data.map(d => ({ x: d.update_count, y: Math.abs(d.actor_loss) }));
    const criticLossData = data.map(d => ({ x: d.update_count, y: d.critic_loss }));
    const entropyData = data.map(d => ({ x: d.update_count, y: d.entropy }));

    charts.loss.data = {
        datasets: [
            { label: '|Actor Loss|', data: actorLossData, borderColor: '#3b82f6' },
            { label: 'Critic Loss', data: criticLossData, borderColor: '#ef4444' },
            { label: 'Entropy', data: entropyData, borderColor: '#10b981' }
        ]
    };
    charts.loss.update();
}

async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        processMetrics(data);
        document.getElementById('connection-status').textContent = 'Live';
        document.getElementById('connection-status').style.color = 'var(--accent-green)';
        document.querySelector('.dot').style.backgroundColor = 'var(--accent-green)';
    } catch (error) {
        console.error("Error fetching metrics:", error);
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.getElementById('connection-status').style.color = '#ef4444';
        document.querySelector('.dot').style.backgroundColor = '#ef4444';
    }
}

// Init and start polling
initCharts();
fetchMetrics();
setInterval(fetchMetrics, 5000);
