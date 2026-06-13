// Global State
let rawData = [];
let processedData = [];
let scaledData = [];
let pcaData = [];
let clusters = {};
let scalerParams = {};
let personas = [];

// Current Database view page
let currentPage = 1;
const rowsPerPage = 15;
let filteredData = [];

// Chart references to update/destroy
let charts = {
    pca: null,
    distribution: null,
    categorySpending: null,
    incomeSpend: null
};

// Initialize Application on DOM Load
window.addEventListener('DOMContentLoaded', () => {
    loadDataset();
});

// 1. Dataset loading and parsing
function loadDataset() {
    document.getElementById('app-loader').style.display = 'flex';
    
    // Fetch the csv file from the server
    fetch('smartcart_customers.csv')
        .then(response => response.text())
        .then(csvText => {
            Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    rawData = results.data;
                    processAndClusterData(4); // Default 4 clusters
                },
                error: function(err) {
                    console.error("Error parsing CSV:", err);
                    alert("Failed to load customer dataset.");
                }
            });
        })
        .catch(err => {
            console.error("Error fetching CSV:", err);
            alert("Please start the local server to run this application.");
        });
}

// 2. Preprocessing & Feature Engineering
function processAndClusterData(k) {
    // A. Clean and Impute
    // Calculate median income (excluding nulls/undefined)
    const incomes = rawData.map(d => d.Income).filter(v => v !== null && v !== undefined && !isNaN(v));
    incomes.sort((a, b) => a - b);
    const medianIncome = incomes[Math.floor(incomes.length / 2)];
    
    // B. Map Dt_Customer to Tenure (reference date is max customer joining date)
    const parsedDates = rawData.map(d => {
        if (!d.Dt_Customer) return new Date(2012, 0, 1);
        const parts = d.Dt_Customer.toString().split('-');
        if (parts.length === 3) {
            const day = parseInt(parts[0], 10);
            const month = parseInt(parts[1], 10) - 1;
            const year = parseInt(parts[2], 10);
            return new Date(year, month, day);
        }
        return new Date(d.Dt_Customer);
    });
    const maxDate = new Date(Math.max(...parsedDates));
    
    // C. Process rows
    processedData = rawData.map((d, index) => {
        const row = { ...d };
        row.Income = (d.Income === null || d.Income === undefined || isNaN(d.Income)) ? medianIncome : d.Income;
        row.Age = 2026 - (d.Year_Birth || 1970);
        
        const date = parsedDates[index];
        row.Customer_Tenure_Days = Math.round((maxDate - date) / (1000 * 60 * 60 * 24));
        
        row.Total_Spending = (d.MntWines || 0) + (d.MntFruits || 0) + (d.MntMeatProducts || 0) + 
                             (d.MntFishProducts || 0) + (d.MntSweetProducts || 0) + (d.MntGoldProds || 0);
        
        row.Total_Children = (d.Kidhome || 0) + (d.Teenhome || 0);
        
        // Education mapping
        let edu = d.Education;
        if (edu === 'Basic' || edu === '2n Cycle') edu = 'Undergraduate';
        else if (edu === 'Graduation') edu = 'Graduate';
        else if (edu === 'Master' || edu === 'PhD') edu = 'Postgraduate';
        else edu = 'Graduate'; // Fallback
        
        row.Education_Mapped = edu;
        row.Education_Graduate = edu === 'Graduate' ? 1 : 0;
        row.Education_Postgraduate = edu === 'Postgraduate' ? 1 : 0;
        row.Education_Undergraduate = edu === 'Undergraduate' ? 1 : 0;
        
        // Marital mapping
        let marital = d.Marital_Status;
        let livingWith = 'Alone';
        if (marital === 'Married' || marital === 'Together') livingWith = 'Partner';
        
        row.Living_With = livingWith;
        row.Living_With_Alone = livingWith === 'Alone' ? 1 : 0;
        row.Living_With_Partner = livingWith === 'Partner' ? 1 : 0;
        
        row.Complain = d.Complain || 0;
        row.Response = d.Response || 0;
        
        return row;
    });

    // B. Scaling
    const featuresList = [
        'Income', 'Recency', 'NumDealsPurchases', 'NumWebPurchases', 
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 
        'Complain', 'Response', 'Age', 'Customer_Tenure_Days', 
        'Total_Spending', 'Total_Children', 
        'Education_Graduate', 'Education_Postgraduate', 'Education_Undergraduate', 
        'Living_With_Alone', 'Living_With_Partner'
    ];

    // Compute means and standard deviations
    scalerParams = {};
    featuresList.forEach(feat => {
        const vals = processedData.map(d => d[feat]);
        const mean = vals.reduce((sum, v) => sum + v, 0) / vals.length;
        const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
        const std = Math.sqrt(variance) || 1.0;
        scalerParams[feat] = { mean, std };
    });

    // Standardize data
    scaledData = processedData.map(d => {
        return featuresList.map(feat => {
            const params = scalerParams[feat];
            return (d[feat] - params.mean) / params.std;
        });
    });

    // C. PCA (2 Components)
    pcaData = runPCA(scaledData, 2);

    // D. K-Means
    const kMeansResult = runKMeans(scaledData, k);
    
    // Assign clusters
    processedData.forEach((d, i) => d.cluster = kMeansResult.assignments[i]);
    clusters = kMeansResult;
    
    generatePersonas(k);
    updateKPIs();
    renderCharts();
    initProfiler();
    filterDatabase();
    
    document.getElementById('app-loader').style.opacity = '0';
    setTimeout(() => {
        document.getElementById('app-loader').style.display = 'none';
    }, 500);
}

function runPCA(matrix, numComponents = 2) {
    const n = matrix.length;
    const m = matrix[0].length;
    
    // Calculate covariance matrix (m x m)
    const cov = Array.from({ length: m }, () => new Array(m).fill(0));
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) {
                sum += matrix[k][i] * matrix[k][j];
            }
            cov[i][j] = sum / (n - 1);
        }
    }
    
    // Power iteration with deflation
    const eigenvectors = [];
    let currentCov = cov.map(row => [...row]);
    
    for (let c = 0; c < numComponents; c++) {
        // Initialize random vector
        let u = Array.from({ length: m }, () => Math.random() - 0.5);
        let norm = Math.sqrt(u.reduce((sum, v) => sum + v * v, 0));
        u = u.map(v => v / norm);
        
        for (let iter = 0; iter < 100; iter++) {
            let uNew = new Array(m).fill(0);
            for (let i = 0; i < m; i++) {
                for (let j = 0; j < m; j++) {
                    uNew[i] += currentCov[i][j] * u[j];
                }
            }
            let normNew = Math.sqrt(uNew.reduce((sum, v) => sum + v * v, 0));
            uNew = uNew.map(v => v / normNew);
            
            // Check convergence
            let diff = 0;
            for (let i = 0; i < m; i++) {
                diff += Math.abs(u[i] - uNew[i]);
            }
            u = uNew;
            if (diff < 1e-8) break;
        }
        
        eigenvectors.push(u);
        
        // Calculate eigenvalue: lambda = u^T * C * u
        let temp = new Array(m).fill(0);
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < m; j++) {
                temp[i] += currentCov[i][j] * u[j];
            }
        }
        let eigenvalue = u.reduce((sum, v, idx) => sum + v * temp[idx], 0);
        
        // Deflate matrix: C = C - lambda * u * u^T
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < m; j++) {
                currentCov[i][j] -= eigenvalue * u[i] * u[j];
            }
        }
    }
    
    // Project matrix onto top eigenvectors
    return matrix.map(row => {
        return eigenvectors.map(ev => {
            return row.reduce((sum, val, idx) => sum + val * ev[idx], 0);
        });
    });
}

function runKMeans(data, k, maxIter = 100) {
    const n = data.length;
    const m = data[0].length;
    
    // K-Means++ initialization
    let centroids = [];
    centroids.push(data[Math.floor(Math.random() * n)]);
    
    for (let c = 1; c < k; c++) {
        let distances = data.map(point => {
            let minDist = Infinity;
            for (let centroid of centroids) {
                let dist = 0;
                for (let j = 0; j < m; j++) {
                    dist += Math.pow(point[j] - centroid[j], 2);
                }
                if (dist < minDist) minDist = dist;
            }
            return minDist;
        });
        
        let sum = distances.reduce((a, b) => a + b, 0);
        let randVal = Math.random() * sum;
        let runningSum = 0;
        let selectedIdx = 0;
        for (let i = 0; i < distances.length; i++) {
            runningSum += distances[i];
            if (runningSum >= randVal) {
                selectedIdx = i;
                break;
            }
        }
        centroids.push(data[selectedIdx]);
    }
    
    let assignments = new Array(n).fill(-1);
    let changed = true;
    let iter = 0;
    
    while (changed && iter < maxIter) {
        changed = false;
        iter++;
        
        // Assignment step
        for (let i = 0; i < n; i++) {
            let minDist = Infinity;
            let closest = -1;
            for (let c = 0; c < k; c++) {
                let dist = 0;
                for (let j = 0; j < m; j++) {
                    dist += Math.pow(data[i][j] - centroids[c][j], 2);
                }
                if (dist < minDist) {
                    minDist = dist;
                    closest = c;
                }
            }
            if (assignments[i] !== closest) {
                assignments[i] = closest;
                changed = true;
            }
        }
        
        // Update step
        let newCentroids = Array.from({ length: k }, () => new Array(m).fill(0));
        let counts = new Array(k).fill(0);
        for (let i = 0; i < n; i++) {
            let clusterId = assignments[i];
            counts[clusterId]++;
            for (let j = 0; j < m; j++) {
                newCentroids[clusterId][j] += data[i][j];
            }
        }
        
        for (let c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (let j = 0; j < m; j++) {
                    centroids[c][j] = newCentroids[c][j] / counts[c];
                }
            }
        }
    }
    
    return { centroids, assignments };
}

// 4. Generate Dynamic Personas and marketing strategies
function generatePersonas(k) {
    personas = [];
    const clusterStats = [];
    
    for (let c = 0; c < k; c++) {
        const clusterRows = processedData.filter(d => d.cluster === c);
        const avgSpending = clusterRows.reduce((sum, d) => sum + d.Total_Spending, 0) / clusterRows.length;
        const avgIncome = clusterRows.reduce((sum, d) => sum + d.Income, 0) / clusterRows.length;
        const avgAge = clusterRows.reduce((sum, d) => sum + d.Age, 0) / clusterRows.length;
        const avgTenure = clusterRows.reduce((sum, d) => sum + d.Customer_Tenure_Days, 0) / clusterRows.length;
        const avgKids = clusterRows.reduce((sum, d) => sum + d.Total_Children, 0) / clusterRows.length;
        
        clusterStats.push({
            id: c,
            size: clusterRows.length,
            avgSpending,
            avgIncome,
            avgAge,
            avgTenure,
            avgKids
        });
    }
    
    // Sort clusters by average total spending to assign descriptive personas consistently
    clusterStats.sort((a, b) => b.avgSpending - a.avgSpending);
    
    const personaTemplates = [
        {
            name: "Elite Spenders",
            class: "persona-c0",
            badge: "badge-c0",
            desc: "High income individuals with high spending capacity. They spend extensively on luxury categories like Wines and Meats, and mostly shop in stores or through catalogs. Typically have few or no children.",
            strategy: "Target with premium loyalty rewards, exclusive access to limited-edition products, and quality-focused (non-discount) marketing."
        },
        {
            name: "Loyal Spenders",
            class: "persona-c1",
            badge: "badge-c1",
            desc: "Middle-to-high income earners with moderate-to-high spending and the highest customer tenure. They are consistent, long-term customers who buy regularly.",
            strategy: "Engage with VIP retention campaigns, milestone rewards, and newsletters highlighting new collections across all channels."
        },
        {
            name: "Frugal Families",
            class: "persona-c2",
            badge: "badge-c2",
            desc: "Moderate income household managers, typically with children. They have average spending habits and are careful about where they allocate budget.",
            strategy: "Promote family-sized deals, practical household bundle discount campaigns, and back-to-school promotional packages."
        },
        {
            name: "Budget Shoppers",
            class: "persona-c3",
            badge: "badge-c3",
            desc: "Low income shoppers who spend very little and are highly price-sensitive. They frequently buy only when deals and discounts are offered, and have larger families on average.",
            strategy: "Heavy targeting with discount coupons, clearance items, BOGO (Buy One Get One) deals, and reminders about cost savings."
        }
    ];

    clusterStats.forEach((stat, rank) => {
        // Fallback for K > 4
        const template = personaTemplates[rank % personaTemplates.length];
        
        personas.push({
            clusterId: stat.id,
            name: template.name + (k > 4 ? ` (Group ${rank + 1})` : ''),
            class: template.class,
            badge: template.badge,
            desc: template.desc,
            strategy: template.strategy,
            stats: stat
        });
    });
}

// 5. Update KPI metrics
function updateKPIs() {
    document.getElementById('kpi-total-cust').innerText = processedData.length.toLocaleString();
    
    const totalIncome = processedData.reduce((sum, d) => sum + d.Income, 0);
    const avgIncome = totalIncome / processedData.length;
    document.getElementById('kpi-avg-income').innerText = '$' + Math.round(avgIncome).toLocaleString();
    
    const totalSpending = processedData.reduce((sum, d) => sum + d.Total_Spending, 0);
    const avgSpending = totalSpending / processedData.length;
    document.getElementById('kpi-avg-spending').innerText = '$' + Math.round(avgSpending).toLocaleString();
    
    // Find largest cluster
    const clusterCounts = {};
    processedData.forEach(d => {
        clusterCounts[d.cluster] = (clusterCounts[d.cluster] || 0) + 1;
    });
    let maxClusterId = 0;
    let maxCount = 0;
    for (const [id, count] of Object.entries(clusterCounts)) {
        if (count > maxCount) {
            maxCount = count;
            maxClusterId = parseInt(id);
        }
    }
    const largestPersona = personas.find(p => p.clusterId === maxClusterId);
    document.getElementById('kpi-largest-segment').innerText = largestPersona ? largestPersona.name : `-`;
}

// 6. Navigation Control
function showSection(sectionId) {
    // Hide all views
    document.querySelectorAll('.tab-content').forEach(view => {
        view.classList.remove('active');
    });
    
    // Show target view
    document.getElementById(sectionId + '-view').classList.add('active');
    
    // Update active class on Nav links
    document.querySelectorAll('.nav-links li').forEach(item => {
        item.classList.remove('active');
    });
    
    const map = {
        dashboard: 'nav-dash',
        profiler: 'nav-prof',
        predictor: 'nav-pred',
        database: 'nav-db',
        about: 'nav-about'
    };
    document.getElementById(map[sectionId]).classList.add('active');
    
    // Update Page Header Titles
    const titles = {
        dashboard: { main: "Segmentation Dashboard", sub: "Overview of SmartCart customer segments" },
        profiler: { main: "Segment Profiler", sub: "Deep dive profiles of each customer persona" },
        predictor: { main: "Predict Segment", sub: "Classify a new customer into a segment" },
        database: { main: "Customer Database", sub: "Full analytical table of SmartCart customers" },
        about: { main: "About the Project", sub: "Goal and architecture overview of SmartCart" }
    };
    
    document.getElementById('view-title').innerText = titles[sectionId].main;
    document.getElementById('view-subtitle').innerText = titles[sectionId].sub;
}

// 7. Light/Dark Theme toggle
function toggleTheme() {
    const body = document.body;
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    body.setAttribute('data-theme', newTheme);
    document.getElementById('theme-icon').innerHTML = newTheme === 'light' ? '&#9790;' : '&#9788;';
    
    // Re-render charts to update grid colors
    renderCharts();
}

// 8. Render Charts using Chart.js CDN
function renderCharts() {
    const isDark = document.body.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#9ca3af' : '#4b5563';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)';

    // A. Destroy existing charts
    for (const key in charts) {
        if (charts[key]) charts[key].destroy();
    }

    // B. Color mappings for clusters
    const segmentColors = ['#818cf8', '#34d399', '#fbbf24', '#fb7185', '#a78bfa', '#ec4899'];
    
    // C. PCA 2D Scatter Chart
    const pcaCtx = document.getElementById('pcaChart').getContext('2d');
    const datasets = Array.from({ length: personas.length }, (_, idx) => {
        const clusterId = personas[idx].clusterId;
        const pts = pcaData.filter((_, i) => processedData[i].cluster === clusterId)
                           .map((proj, i) => ({ x: proj[0], y: proj[1] }));
        return {
            label: personas[idx].name,
            data: pts,
            backgroundColor: segmentColors[idx % segmentColors.length],
            pointRadius: 4,
            pointHoverRadius: 6
        };
    });

    charts.pca = new Chart(pcaCtx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: gridColor }, ticks: { color: textColor } },
                y: { grid: { color: gridColor }, ticks: { color: textColor } }
            },
            plugins: {
                legend: { labels: { color: textColor, font: { family: 'Inter' } } }
            }
        }
    });

    // D. Segment Distribution Pie Chart
    const distCtx = document.getElementById('distributionChart').getContext('2d');
    charts.distribution = new Chart(distCtx, {
        type: 'doughnut',
        data: {
            labels: personas.map(p => p.name),
            datasets: [{
                data: personas.map(p => p.stats.size),
                backgroundColor: segmentColors.slice(0, personas.length),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: textColor, font: { family: 'Inter' } } }
            }
        }
    });

    // E. Average Category Spending Chart (Bar Chart)
    const categoryCtx = document.getElementById('categorySpendingChart').getContext('2d');
    const categories = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'];
    const barDatasets = personas.map((p, idx) => {
        const clusterRows = processedData.filter(d => d.cluster === p.clusterId);
        const data = categories.map(cat => {
            const key = cat === 'Wines' ? 'MntWines' : 
                        cat === 'Fruits' ? 'MntFruits' : 
                        cat === 'Meat' ? 'MntMeatProducts' : 
                        cat === 'Fish' ? 'MntFishProducts' : 
                        cat === 'Sweets' ? 'MntSweetProducts' : 'MntGoldProds';
            return clusterRows.reduce((sum, d) => sum + (d[key] || 0), 0) / clusterRows.length;
        });

        return {
            label: p.name,
            data: data,
            backgroundColor: segmentColors[idx % segmentColors.length],
            borderRadius: 6
        };
    });

    charts.categorySpending = new Chart(categoryCtx, {
        type: 'bar',
        data: { labels: categories, datasets: barDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: gridColor }, ticks: { color: textColor } },
                y: { grid: { color: gridColor }, ticks: { color: textColor } }
            },
            plugins: {
                legend: { position: 'bottom', labels: { color: textColor, font: { family: 'Inter' } } }
            }
        }
    });

    // F. Income vs Total Spending Scatter plot
    const isCtx = document.getElementById('incomeSpendChart').getContext('2d');
    const isDatasets = personas.map((p, idx) => {
        const clusterRows = processedData.filter(d => d.cluster === p.clusterId);
        const pts = clusterRows.map(d => ({ x: d.Total_Spending, y: d.Income }));
        return {
            label: p.name,
            data: pts,
            backgroundColor: segmentColors[idx % segmentColors.length],
            pointRadius: 3,
            pointHoverRadius: 5
        };
    });

    charts.incomeSpend = new Chart(isCtx, {
        type: 'scatter',
        data: { datasets: isDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Total Spending ($)', color: textColor }, grid: { color: gridColor }, ticks: { color: textColor } },
                y: { title: { display: true, text: 'Annual Income ($)', color: textColor }, grid: { color: gridColor }, ticks: { color: textColor } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// 9. Segment Profiler Setup
function initProfiler() {
    const profilerTabs = document.getElementById('profiler-tabs');
    profilerTabs.innerHTML = '';
    
    // Inject Tab buttons
    personas.forEach((p, idx) => {
        const btn = document.createElement('button');
        btn.className = `tab-btn ${idx === 0 ? 'active' : ''}`;
        btn.innerText = p.name;
        btn.setAttribute('onclick', `switchProfilerTab(${idx})`);
        profilerTabs.appendChild(btn);
    });

    switchProfilerTab(0);
}

function switchProfilerTab(index) {
    // Update active tab button style
    const tabButtons = document.querySelectorAll('#profiler-tabs .tab-btn');
    tabButtons.forEach((btn, idx) => {
        if (idx === index) btn.classList.add('active');
        else btn.classList.remove('active');
    });

    const p = personas[index];
    const wrapper = document.getElementById('profiler-content-wrapper');
    
    // Profile metrics calculation
    const clusterRows = processedData.filter(d => d.cluster === p.clusterId);
    
    const countKids = clusterRows.reduce((sum, d) => sum + (d.Kidhome || 0), 0) / clusterRows.length;
    const countTeens = clusterRows.reduce((sum, d) => sum + (d.Teenhome || 0), 0) / clusterRows.length;
    
    // Education averages
    const gradPct = Math.round((clusterRows.reduce((sum, d) => sum + d.Education_Graduate, 0) / clusterRows.length) * 100);
    const postGradPct = Math.round((clusterRows.reduce((sum, d) => sum + d.Education_Postgraduate, 0) / clusterRows.length) * 100);
    const underGradPct = Math.round((clusterRows.reduce((sum, d) => sum + d.Education_Undergraduate, 0) / clusterRows.length) * 100);

    // Living With averages
    const partnerPct = Math.round((clusterRows.reduce((sum, d) => sum + d.Living_With_Partner, 0) / clusterRows.length) * 100);
    const alonePct = Math.round((clusterRows.reduce((sum, d) => sum + d.Living_With_Alone, 0) / clusterRows.length) * 100);

    wrapper.innerHTML = `
        <div class="card ${p.class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="font-family: Outfit, sans-serif; font-size: 1.5rem;">${p.name}</h2>
                <span class="badge ${p.badge}">Cluster ${p.clusterId}</span>
            </div>
            
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem; line-height: 1.6;">${p.desc}</p>
            
            <div class="kpi-container" style="margin-bottom: 2rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">
                <div class="kpi-card" style="background: rgba(0,0,0,0.1); padding: 1rem;">
                    <span class="kpi-title" style="font-size: 0.75rem;">Avg Income</span>
                    <span class="kpi-value" style="font-size: 1.35rem;">$${Math.round(p.stats.avgIncome).toLocaleString()}</span>
                </div>
                <div class="kpi-card" style="background: rgba(0,0,0,0.1); padding: 1rem;">
                    <span class="kpi-title" style="font-size: 0.75rem;">Avg Spending</span>
                    <span class="kpi-value" style="font-size: 1.35rem;">$${Math.round(p.stats.avgSpending).toLocaleString()}</span>
                </div>
                <div class="kpi-card" style="background: rgba(0,0,0,0.1); padding: 1rem;">
                    <span class="kpi-title" style="font-size: 0.75rem;">Average Age</span>
                    <span class="kpi-value" style="font-size: 1.35rem;">${Math.round(p.stats.avgAge)} Years</span>
                </div>
                <div class="kpi-card" style="background: rgba(0,0,0,0.1); padding: 1rem;">
                    <span class="kpi-title" style="font-size: 0.75rem;">Avg Kids/Teens</span>
                    <span class="kpi-value" style="font-size: 1.35rem;">${countKids.toFixed(1)} / ${countTeens.toFixed(1)}</span>
                </div>
            </div>

            <div class="dashboard-grid">
                <div class="col-6" style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <h3 style="font-size: 1rem; color: var(--primary-color);">Education Distribution</h3>
                    <div style="display: flex; justify-content: space-between;"><span>Graduate:</span> <strong>${gradPct}%</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span>Postgraduate:</span> <strong>${postGradPct}%</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span>Undergraduate:</span> <strong>${underGradPct}%</strong></div>
                </div>
                <div class="col-6" style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <h3 style="font-size: 1rem; color: var(--primary-color);">Social Dynamics</h3>
                    <div style="display: flex; justify-content: space-between;"><span>Living with Partner:</span> <strong>${partnerPct}%</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span>Living Alone:</span> <strong>${alonePct}%</strong></div>
                </div>
            </div>

            <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--panel-border);">
                <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem; font-family: Outfit, sans-serif;">Actionable Marketing Recommendation</h3>
                <p style="color: var(--text-secondary); line-height: 1.5;">${p.strategy}</p>
            </div>
        </div>
    `;
}

// 10. Predict customer cluster membership
function predictCustomer(e) {
    e.preventDefault();
    
    // Extract input values
    const birthYear = parseInt(document.getElementById('pred-birth').value);
    const income = parseFloat(document.getElementById('pred-income').value);
    const education = document.getElementById('pred-education').value;
    const marital = document.getElementById('pred-marital').value;
    const kidhome = parseInt(document.getElementById('pred-kidhome').value);
    const teenhome = parseInt(document.getElementById('pred-teenhome').value);
    const recency = parseInt(document.getElementById('pred-recency').value);
    const wines = parseFloat(document.getElementById('pred-wines').value);
    const fruits = parseFloat(document.getElementById('pred-fruits').value);
    const meat = parseFloat(document.getElementById('pred-meat').value);
    const fish = parseFloat(document.getElementById('pred-fish').value);
    const sweets = parseFloat(document.getElementById('pred-sweets').value);
    const gold = parseFloat(document.getElementById('pred-gold').value);
    const deals = parseInt(document.getElementById('pred-deals').value);
    const web = parseInt(document.getElementById('pred-web').value);
    const catalog = parseInt(document.getElementById('pred-catalog').value);
    const store = parseInt(document.getElementById('pred-store').value);
    const webvisits = parseInt(document.getElementById('pred-webvisits').value);

    // Calculate engineered fields for new row
    const age = 2026 - birthYear;
    
    // Impute tenure to average tenure of dataset
    const avgTenure = processedData.reduce((sum, d) => sum + d.Customer_Tenure_Days, 0) / processedData.length;
    const totalSpending = wines + fruits + meat + fish + sweets + gold;
    const totalChildren = kidhome + teenhome;

    const edu_grad = education === 'Graduate' ? 1 : 0;
    const edu_post = education === 'Postgraduate' ? 1 : 0;
    const edu_under = education === 'Undergraduate' ? 1 : 0;

    const living_alone = marital === 'Alone' ? 1 : 0;
    const living_partner = marital === 'Partner' ? 1 : 0;

    const newCustomerObj = {
        Income: income,
        Recency: recency,
        NumDealsPurchases: deals,
        NumWebPurchases: web,
        NumCatalogPurchases: catalog,
        NumStorePurchases: store,
        NumWebVisitsMonth: webvisits,
        Complain: 0,
        Response: 0,
        Age: age,
        Customer_Tenure_Days: avgTenure,
        Total_Spending: totalSpending,
        Total_Children: totalChildren,
        Education_Graduate: edu_grad,
        Education_Postgraduate: edu_post,
        Education_Undergraduate: edu_under,
        Living_With_Alone: living_alone,
        Living_With_Partner: living_partner
    };

    const featuresList = [
        'Income', 'Recency', 'NumDealsPurchases', 'NumWebPurchases', 
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 
        'Complain', 'Response', 'Age', 'Customer_Tenure_Days', 
        'Total_Spending', 'Total_Children', 
        'Education_Graduate', 'Education_Postgraduate', 'Education_Undergraduate', 
        'Living_With_Alone', 'Living_With_Partner'
    ];

    // Standardize input
    const scaledInput = featuresList.map(feat => {
        const params = scalerParams[feat];
        return (newCustomerObj[feat] - params.mean) / params.std;
    });

    // Run custom K-Means assignment algorithm against calculated centroids
    const centroids = getKMeansCentroids(scaledData, personas.length);
    let minDist = Infinity;
    let predictedCluster = -1;

    for (let c = 0; c < centroids.length; c++) {
        let dist = 0;
        for (let j = 0; j < scaledInput.length; j++) {
            dist += Math.pow(scaledInput[j] - centroids[c][j], 2);
        }
        if (dist < minDist) {
            minDist = dist;
            predictedCluster = c;
        }
    }

    // Lookup predicted cluster persona details
    const persona = personas.find(p => p.clusterId === predictedCluster);

    // Update Output
    document.getElementById('pred-cluster-title').innerText = `Predicted Segment: ${persona.name}`;
    document.getElementById('pred-cluster-badge').className = `badge ${persona.badge}`;
    document.getElementById('pred-cluster-badge').innerText = `Cluster ${persona.clusterId}`;
    document.getElementById('pred-cluster-desc').innerText = persona.desc;
    document.getElementById('pred-cluster-strategy').innerText = persona.strategy;
    document.getElementById('prediction-result').style.display = 'flex';
}

function getKMeansCentroids(data, k) {
    // We compute centroids from the labeled processedData
    const centroids = Array.from({ length: k }, () => new Array(data[0].length).fill(0));
    const counts = new Array(k).fill(0);
    
    for (let i = 0; i < data.length; i++) {
        const clusterId = processedData[i].cluster;
        counts[clusterId]++;
        for (let j = 0; j < data[0].length; j++) {
            centroids[clusterId][j] += data[i][j];
        }
    }
    
    for (let c = 0; c < k; c++) {
        if (counts[c] > 0) {
            for (let j = 0; j < data[0].length; j++) {
                centroids[c][j] = centroids[c][j] / counts[c];
            }
        }
    }
    return centroids;
}

// 10. Database Table rendering and search/pagination
function filterDatabase() {
    const searchVal = document.getElementById('db-search').value.toLowerCase();
    const clusterFilter = document.getElementById('db-cluster-filter').value;
    
    filteredData = processedData.filter(d => {
        const matchSearch = d.ID.toString().includes(searchVal) || d.Education_Mapped.toLowerCase().includes(searchVal);
        const matchCluster = clusterFilter === 'all' || d.cluster.toString() === clusterFilter;
        return matchSearch && matchCluster;
    });

    currentPage = 1;
    renderDatabaseTable();
}

function renderDatabaseTable() {
    const tbody = document.getElementById('database-table-body');
    tbody.innerHTML = '';

    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    const pageRows = filteredData.slice(start, end);

    pageRows.forEach(d => {
        const tr = document.createElement('tr');
        const persona = personas.find(p => p.clusterId === d.cluster);
        
        tr.innerHTML = `
            <td>#${d.ID}</td>
            <td>${d.Age}</td>
            <td>${d.Education_Mapped}</td>
            <td>${d.Living_With}</td>
            <td>$${d.Income.toLocaleString()}</td>
            <td>${d.Total_Children}</td>
            <td>$${d.Total_Spending.toLocaleString()}</td>
            <td>${d.Customer_Tenure_Days}</td>
            <td><span class="badge ${persona.badge}">${persona.name}</span></td>
        `;
        tbody.appendChild(tr);
    });

    // Update pagination status details
    const totalRows = filteredData.length;
    const from = totalRows === 0 ? 0 : start + 1;
    const to = Math.min(end, totalRows);
    document.getElementById('db-pagination-info').innerText = `Showing ${from} to ${to} of ${totalRows.toLocaleString()} customers`;
    
    // Toggle button disabled states
    document.getElementById('btn-prev-page').disabled = currentPage === 1;
    document.getElementById('btn-next-page').disabled = end >= totalRows;
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        renderDatabaseTable();
    }
}

function nextPage() {
    if (currentPage * rowsPerPage < filteredData.length) {
        currentPage++;
        renderDatabaseTable();
    }
}



